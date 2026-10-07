//! Exact stored-sample reads through the shared RITK image-I/O contract.

mod error;
mod volumes;

pub use error::{NrrdSpatialMetadataField, NrrdStoredReadError};

use ritk_image_io::{ImageReadBudget, SeriesAxis, StoredSeries, StoredVolume};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use super::diffusion::scheme_from_header;
use super::header::{open_nrrd_header_reader, NrrdHeader, NrrdHeaderError};
use super::volume::{
    parse_nrrd_raw, parse_nrrd_read_plan, read_nrrd_payload, NrrdReadPlan, NrrdReadPurpose, RawNrrd,
};
use crate::axes::AcquisitionAxis;

/// Read one NRRD volume without changing its stored sample type or bits.
///
/// A 2-D file is represented as a one-slice volume. Every 4-D file has an
/// acquisition axis and must use [`read_nrrd_stored_series`], including files
/// whose axis contains one entry.
///
/// # Errors
///
/// Returns a [`NrrdStoredReadError`] that identifies an invalid header field,
/// unsupported encoding or geometry, truncated payload, sample decoding
/// failure, allocation failure, or a violation of the shared stored-volume
/// contract. `budget` bounds encoded payload bytes, decoded output bytes,
/// gzip-expanded payload bytes (including a declared byte skip), and series
/// volume count before payload allocation. A multi-volume acquisition returns
/// [`NrrdStoredReadError::AcquisitionAxisRequiresSeries`].
pub fn read_nrrd_stored<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<StoredVolume, NrrdStoredReadError> {
    let parsed = parse_nrrd_raw(path, budget, NrrdReadPurpose::StoredVolume)?;
    volumes::decode_stored_volumes(parsed)?
        .into_iter()
        .next()
        .ok_or(NrrdStoredReadError::InvalidVolumeRange { volume_index: 0 })
}

/// Read a NRRD volume sequence without discarding acquisition-axis meaning.
///
/// Both a leading, interleaved acquisition axis and a trailing, contiguous
/// acquisition axis are preserved in acquisition order. The returned series
/// carries its declared `list` or diffusion meaning; an undeclared axis is
/// marked unspecified. Unsupported axis kinds fail rather than becoming lists.
/// `budget` bounds encoded payload bytes, decoded output bytes, and the
/// returned volume count before sample storage is allocated.
///
/// # Errors
///
/// Returns a [`NrrdStoredReadError`] that identifies an invalid header field,
/// unsupported encoding or geometry, truncated payload, sample decoding
/// failure, allocation failure, or a violation of the shared stored-volume
/// contract. `budget` bounds encoded bytes, decoded output, gzip-expanded
/// bytes (including a declared byte skip), and series volume count before
/// payload allocation.
pub fn read_nrrd_stored_series<P: AsRef<Path>>(
    path: P,
    budget: ImageReadBudget,
) -> Result<StoredSeries, NrrdStoredReadError> {
    stored_series(parse_nrrd_raw(path, budget, NrrdReadPurpose::StoredSeries)?)
}

fn stored_series(mut parsed: RawNrrd) -> Result<StoredSeries, NrrdStoredReadError> {
    let axis = parsed
        .series_axis
        .take()
        .ok_or(NrrdStoredReadError::MissingSeriesAxis)?;
    let volumes = volumes::decode_stored_volumes(parsed)?;
    StoredSeries::new(volumes, axis).map_err(|source| NrrdStoredReadError::StoredSeries { source })
}

/// One opened NRRD source with its parsed header and unread payload stream.
pub(crate) struct NrrdReadSession {
    path: PathBuf,
    reader: BufReader<std::fs::File>,
    header: NrrdHeader,
}

impl NrrdReadSession {
    /// Opens one source and parses its header without reopening the path.
    pub(crate) fn open(path: &Path) -> Result<Self, NrrdHeaderError> {
        let path = path.to_path_buf();
        let (reader, header) = open_nrrd_header_reader(&path)?;
        Ok(Self {
            path,
            reader,
            header,
        })
    }

    /// Returns the parsed header while the same source remains open.
    pub(crate) fn header(&self) -> &NrrdHeader {
        &self.header
    }

    /// Prepares the stored series metadata without reading its voxel payload.
    pub(crate) fn prepare_stored_series(
        mut self,
        budget: ImageReadBudget,
    ) -> Result<PreparedNrrdSeries, NrrdStoredReadError> {
        let plan = parse_nrrd_read_plan(
            &mut self.reader,
            &self.header,
            budget,
            NrrdReadPurpose::StoredSeries,
        )?;
        Ok(PreparedNrrdSeries {
            path: self.path,
            reader: self.reader,
            plan,
        })
    }
}

pub(crate) struct PreparedNrrdSeries {
    path: PathBuf,
    reader: BufReader<std::fs::File>,
    plan: NrrdReadPlan,
}

impl PreparedNrrdSeries {
    /// Returns the parsed stored-series layout before payload decoding.
    pub(crate) fn plan(&self) -> &NrrdReadPlan {
        &self.plan
    }

    /// Reads the voxel payload from the same open source and builds its series.
    pub(crate) fn into_stored_series(
        mut self,
        budget: ImageReadBudget,
    ) -> Result<StoredSeries, NrrdStoredReadError> {
        let parsed = read_nrrd_payload(&self.path, &mut self.reader, self.plan, budget)?;
        stored_series(parsed)
    }
}

pub(crate) fn has_diffusion_metadata(header: &NrrdHeader) -> bool {
    header.key_values.iter().any(|(key, value)| {
        (key.eq_ignore_ascii_case("modality") && value.eq_ignore_ascii_case("DWMRI"))
            || key.to_ascii_uppercase().starts_with("DWMRI_")
    })
}

pub(super) fn stored_series_axis(
    header: &NrrdHeader,
    acquisition: AcquisitionAxis,
) -> Result<SeriesAxis, NrrdStoredReadError> {
    let has_diffusion = has_diffusion_metadata(header);
    if !has_diffusion && let Some(measurement_frame) = header.fields.get("measurement frame") {
        return Err(NrrdStoredReadError::UnsupportedMeasurementFrame {
            measurement_frame: measurement_frame.clone(),
        });
    }
    if has_diffusion {
        if acquisition == AcquisitionAxis::Absent {
            return Err(NrrdStoredReadError::DiffusionRequiresAcquisitionAxis);
        }
        if let Some(kind) = acquisition_kind(header, acquisition)
            && !kind.eq_ignore_ascii_case("list")
        {
            return Err(NrrdStoredReadError::UnsupportedAcquisitionKind {
                kind: kind.to_owned(),
            });
        }
        let scheme = scheme_from_header(header)
            .map_err(|source| NrrdStoredReadError::DiffusionScheme { source })?;
        return Ok(SeriesAxis::Diffusion(scheme));
    }
    if acquisition == AcquisitionAxis::Absent {
        return Ok(SeriesAxis::SingleVolume);
    }
    match acquisition_kind(header, acquisition) {
        Some(kind) if kind.eq_ignore_ascii_case("list") => Ok(SeriesAxis::List),
        Some(kind) => Err(NrrdStoredReadError::UnsupportedAcquisitionKind {
            kind: kind.to_owned(),
        }),
        None => Ok(SeriesAxis::Unspecified),
    }
}

pub(super) fn acquisition_axis_index(axis: AcquisitionAxis) -> usize {
    match axis {
        AcquisitionAxis::Absent | AcquisitionAxis::Slowest => 3,
        AcquisitionAxis::Fastest => 0,
    }
}

pub(super) fn acquisition_kind(header: &NrrdHeader, axis: AcquisitionAxis) -> Option<&str> {
    let index = acquisition_axis_index(axis);
    header.fields.get("kinds")?.split_whitespace().nth(index)
}
