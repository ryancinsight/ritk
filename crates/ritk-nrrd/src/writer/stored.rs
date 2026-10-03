//! Exact stored-sample writes through the shared RITK image-I/O contract.

use ritk_codecs::{ByteOrder, SampleError, SampleType, SampleWriteError};
use ritk_diffusion_scheme::GradientFrame;
use ritk_image_io::{
    validate_coordinate_map, validate_physical_geometry, SeriesAxis, StoredSeries, StoredVolume,
    VolumeError,
};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;

use super::{write_nrrd_header, write_nrrd_series_header, HeaderBuffer, SeriesLayout};

/// A stored volume cannot be represented as a valid NRRD output.
#[derive(Debug, Error)]
pub enum NrrdStoredWriteError {
    /// NRRD has no field for the volume's stored-to-real intensity transform.
    #[error("NRRD cannot represent non-identity intensity calibration")]
    UnsupportedCalibration,
    /// A stored sample representation has no NRRD type spelling.
    #[error("NRRD cannot represent stored sample type {sample_type:?}")]
    UnsupportedSampleType {
        /// The sample type without a corresponding NRRD element type.
        sample_type: SampleType,
    },
    /// A physical geometry cannot be represented by NRRD space directions.
    #[error("NRRD stored-volume geometry is invalid: {source}")]
    PhysicalGeometry {
        /// Shared image-I/O geometry-contract failure.
        #[source]
        source: VolumeError,
    },
    /// A series has no volume to write.
    #[error("an NRRD stored series must contain at least one volume")]
    EmptySeries,
    /// A diffusion gradient frame cannot be encoded in a NRRD physical frame.
    #[error("NRRD stored writer cannot encode diffusion frame {frame:?}")]
    UnsupportedGradientFrame {
        /// Frame declared by the diffusion scheme.
        frame: GradientFrame,
    },
    /// A nonzero diffusion weighting underflows when represented by NRRD gradients.
    #[error("NRRD cannot represent diffusion weighting at volume {index}")]
    UnrepresentableDiffusionWeighting {
        /// Acquisition-order volume index.
        index: usize,
    },
    /// The acquisition-axis meaning has no NRRD series representation.
    #[error("NRRD cannot encode this acquisition-axis meaning")]
    UnsupportedAxisMeaning,
    /// A volume has a different shape from the first volume in the series.
    #[error("NRRD series volume {index} has shape {actual:?}; expected {expected:?}")]
    ShapeMismatch {
        /// Volume index in acquisition order.
        index: usize,
        /// Required depth, row, column shape.
        expected: [usize; 3],
        /// Supplied depth, row, column shape.
        actual: [usize; 3],
    },
    /// A volume has different physical geometry from the first volume.
    #[error("NRRD series volume {index} has different physical geometry")]
    GeometryMismatch {
        /// Volume index in acquisition order.
        index: usize,
    },
    /// A volume has a different non-affine coordinate map from the first.
    #[error("NRRD series volume {index} has a different coordinate map")]
    CoordinateMapMismatch {
        /// Volume index in acquisition order.
        index: usize,
    },
    /// A volume has a different stored sample representation from the first.
    #[error("NRRD series volume {index} has a different stored sample type")]
    SampleTypeMismatch {
        /// Volume index in acquisition order.
        index: usize,
    },
    /// The serialized header exceeds the reader's bounded header size.
    #[error("NRRD output header exceeds {maximum_bytes} bytes (at least {header_bytes} bytes)")]
    HeaderTooLarge {
        /// Lower bound on serialized header length; excess bytes are not retained.
        header_bytes: usize,
        /// Largest header length accepted by the reader.
        maximum_bytes: usize,
    },
    /// The serialized header would exceed the reader's entry-count bound.
    #[error(
        "NRRD output header has {entries} metadata entries; the reader limit is {maximum_entries}"
    )]
    HeaderTooManyEntries {
        /// Number of fields and key/value entries the writer would emit.
        entries: usize,
        /// Largest entry count accepted by the reader.
        maximum_entries: usize,
    },
    /// Sample encoding could not produce a complete NRRD payload.
    #[error("NRRD stored sample encoding failed: {source}")]
    SampleEncoding {
        /// Exact sample-codec failure.
        #[source]
        source: SampleError,
    },
    /// NRRD file creation, writing, or flushing failed.
    #[error("NRRD output failed: {0}")]
    Io(#[from] std::io::Error),
}

/// Write a stored volume without changing its sample type, values, or bits.
///
/// The writer emits raw little-endian payloads and preserves spatial metadata
/// and the coordinate map. Calibration must be identity because NRRD has no
/// standard field for the transforms represented by `IntensityCalibration`.
///
/// # Errors
///
/// Returns [`NrrdStoredWriteError::UnsupportedCalibration`] or
/// [`NrrdStoredWriteError::UnsupportedSampleType`] before creating the output
/// file. Other variants describe sample encoding and filesystem failures.
pub fn write_nrrd_stored<P: AsRef<Path>>(
    path: P,
    volume: &StoredVolume,
) -> Result<(), NrrdStoredWriteError> {
    validate_calibration(volume)?;
    validate_physical_geometry(volume.metadata())
        .map_err(|source| NrrdStoredWriteError::PhysicalGeometry { source })?;
    let element_type = nrrd_type_name(volume.samples().sample_type())?;
    validate_coordinate_map(volume.coordinate_map(), volume.shape())
        .map_err(|source| NrrdStoredWriteError::PhysicalGeometry { source })?;
    let mut header = HeaderBuffer::new();
    let header_result = write_nrrd_header(
        &mut header,
        volume.shape(),
        volume.metadata().spacing(),
        volume.metadata().origin(),
        volume.metadata().direction(),
        element_type,
        volume.coordinate_map(),
    );
    ensure_header_result(&header, header_result)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(header.bytes())?;
    write_sample_payload(volume, &mut writer)?;
    writer.flush()?;
    Ok(())
}

/// Write a stored acquisition series as a NRRD file.
///
/// Each volume must share shape, sample type, physical geometry, and
/// coordinate mapping. The writer preserves the axis meaning. A
/// [`SeriesAxis::SingleVolume`] uses the canonical 3-D representation; other
/// axes remain 4-D even when they contain one entry. Diffusion schemes are
/// serialized in LPS physical coordinates.
///
/// # Errors
///
/// Returns a semantic error before creating the output file when the series is
/// a volume differs from the first, calibration is non-identity, or the axis
/// metadata cannot be represented.
pub fn write_nrrd_stored_series<P: AsRef<Path>>(
    path: P,
    series: &StoredSeries,
) -> Result<(), NrrdStoredWriteError> {
    let volumes = series.volumes();
    let Some((first, rest)) = volumes.split_first() else {
        return Err(NrrdStoredWriteError::EmptySeries);
    };
    validate_series_axis(series.axis())?;
    validate_series_header_entries(series.axis(), first.coordinate_map())?;
    validate_calibration(first)?;
    validate_physical_geometry(first.metadata())
        .map_err(|source| NrrdStoredWriteError::PhysicalGeometry { source })?;
    let sample_type = first.samples().sample_type();
    let element_type = nrrd_type_name(sample_type)?;
    for (offset, volume) in rest.iter().enumerate() {
        let index = offset + 1;
        validate_calibration(volume)?;
        if volume.shape() != first.shape() {
            return Err(NrrdStoredWriteError::ShapeMismatch {
                index,
                expected: first.shape(),
                actual: volume.shape(),
            });
        }
        if volume.metadata() != first.metadata() {
            return Err(NrrdStoredWriteError::GeometryMismatch { index });
        }
        if volume.coordinate_map() != first.coordinate_map() {
            return Err(NrrdStoredWriteError::CoordinateMapMismatch { index });
        }
        if volume.samples().sample_type() != sample_type {
            return Err(NrrdStoredWriteError::SampleTypeMismatch { index });
        }
    }

    if matches!(series.axis(), SeriesAxis::SingleVolume) {
        return write_nrrd_stored(path, first);
    }

    let mut header = HeaderBuffer::new();
    let header_result = write_nrrd_series_header(
        &mut header,
        first.shape(),
        volumes.len(),
        first.metadata().spacing(),
        first.metadata().origin(),
        first.metadata().direction(),
        element_type,
        first.coordinate_map(),
        SeriesLayout::AcquisitionSlowest,
        series.axis(),
    );
    ensure_header_result(&header, header_result)?;
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    writer.write_all(header.bytes())?;
    for volume in volumes {
        write_sample_payload(volume, &mut writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_sample_payload<W: Write>(
    volume: &StoredVolume,
    writer: &mut W,
) -> Result<(), NrrdStoredWriteError> {
    volume
        .samples()
        .write_to(writer, ByteOrder::LeastSignificantByteFirst)
        .map_err(|source| match source {
            SampleWriteError::Sample(source) => NrrdStoredWriteError::SampleEncoding { source },
            SampleWriteError::Io(source) => NrrdStoredWriteError::Io(source),
            // `SampleWriteError` is `#[non_exhaustive]` for forward compatibility;
            // the two variants above are exhaustive for the current `ritk-codecs`
            // release, so this arm is unreachable until a new variant is added
            // upstream (which then extends this mapping instead of panicking).
            _ => unreachable!("exhaustive SampleWriteError mapping covers all current variants"),
        })
}

fn validate_series_axis(axis: &SeriesAxis) -> Result<(), NrrdStoredWriteError> {
    let SeriesAxis::Diffusion(scheme) = axis else {
        return Ok(());
    };
    if scheme.frame() != GradientFrame::Lps {
        return Err(NrrdStoredWriteError::UnsupportedGradientFrame {
            frame: scheme.frame(),
        });
    }
    let maximum = scheme
        .directions()
        .iter()
        .map(|entry| entry.weighting().seconds_per_square_millimeter())
        .fold(0.0_f64, f64::max);
    for (index, entry) in scheme.directions().iter().enumerate() {
        let weighting = entry.weighting().seconds_per_square_millimeter();
        if weighting > 0.0 && (weighting / maximum).sqrt() == 0.0 {
            return Err(NrrdStoredWriteError::UnrepresentableDiffusionWeighting { index });
        }
    }
    Ok(())
}

fn validate_series_header_entries(
    axis: &SeriesAxis,
    coordinate_map: &ritk_spatial::CoordinateMap,
) -> Result<(), NrrdStoredWriteError> {
    let diffusion_entries = match axis {
        SeriesAxis::Diffusion(scheme) => {
            scheme
                .len()
                .checked_add(2)
                .ok_or(NrrdStoredWriteError::HeaderTooManyEntries {
                    entries: usize::MAX,
                    maximum_entries: crate::reader::MAX_HEADER_ENTRIES,
                })?
        }
        SeriesAxis::SingleVolume | SeriesAxis::Unspecified | SeriesAxis::List => 0,
        _ => {
            return Err(NrrdStoredWriteError::UnsupportedAxisMeaning);
        }
    };
    let coordinate_map_entry = if matches!(coordinate_map, ritk_spatial::CoordinateMap::Cartesian) {
        0
    } else {
        1
    };
    let entries = 10_usize
        .checked_add(coordinate_map_entry)
        .and_then(|entries| entries.checked_add(diffusion_entries))
        .ok_or(NrrdStoredWriteError::HeaderTooManyEntries {
            entries: usize::MAX,
            maximum_entries: crate::reader::MAX_HEADER_ENTRIES,
        })?;
    let maximum_entries = crate::reader::MAX_HEADER_ENTRIES;
    if entries > maximum_entries {
        return Err(NrrdStoredWriteError::HeaderTooManyEntries {
            entries,
            maximum_entries,
        });
    }
    Ok(())
}

fn validate_calibration(volume: &StoredVolume) -> Result<(), NrrdStoredWriteError> {
    if !volume.calibration().is_identity() {
        return Err(NrrdStoredWriteError::UnsupportedCalibration);
    }
    Ok(())
}

fn ensure_header_result(
    header: &HeaderBuffer,
    result: std::io::Result<()>,
) -> Result<(), NrrdStoredWriteError> {
    if header.exceeded_limit() {
        let maximum_bytes = crate::reader::MAX_HEADER_BYTES;
        return Err(NrrdStoredWriteError::HeaderTooLarge {
            header_bytes: maximum_bytes.saturating_add(1),
            maximum_bytes,
        });
    }
    result?;
    Ok(())
}

fn nrrd_type_name(sample_type: SampleType) -> Result<&'static str, NrrdStoredWriteError> {
    let name = match sample_type {
        SampleType::U8 => "unsigned char",
        SampleType::I8 => "signed char",
        SampleType::U16 => "unsigned short",
        SampleType::I16 => "short",
        SampleType::U32 => "unsigned int",
        SampleType::I32 => "int",
        SampleType::U64 => "unsigned long long",
        SampleType::I64 => "long long",
        SampleType::F32 => "float",
        SampleType::F64 => "double",
        _ => {
            return Err(NrrdStoredWriteError::UnsupportedSampleType { sample_type });
        }
    };
    Ok(name)
}
