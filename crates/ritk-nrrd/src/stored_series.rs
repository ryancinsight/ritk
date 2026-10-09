//! Preflighted NRRD documents from shared stored samples.
//!
//! NRRD's stored-series writer used to validate inside `NrrdDocument::new` and
//! again inside `write_to`, so the shared conversion preflight in
//! `ritk_image_io` had no NRRD implementation to route through and every
//! caller rediscovered the same constraints. [`NrrdStoredSeriesTarget`] states
//! those constraints once, as a [`ConversionAdapter`], and both
//! [`NrrdDocument::from_stored_series`] and the document's own write guard run
//! it before any destination exists.

use crate::document::{validate_document_metadata, NrrdDocument, NrrdDocumentError};
use crate::writer::{
    nrrd_type_name, validate_calibration, validate_series_axis, validate_series_header_entries,
    write_nrrd_header_with_metadata, write_nrrd_series_header_with_metadata, HeaderBuffer,
    NrrdStoredWriteError, SeriesLayout,
};
use ritk_codecs::SampleType;
use ritk_diffusion_scheme::GradientFrame;
use ritk_image_io::{
    prepare_conversion, validate_coordinate_map, validate_physical_geometry, ConversionAdapter,
    ConversionFeature, ConversionLocation, ConversionPrepareError, ConversionRejection,
    ConversionTarget, FormatMetadataLoss, SeriesAxis, StoredSeries, VolumeError,
};
use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
use thiserror::Error;

/// Source identifier reported when a document is validated without one.
pub(crate) const NRRD_DOCUMENT_SOURCE: &str = "stored";

/// NRRD's encodable semantic categories.
///
/// NRRD carries a raw element type, a full 3-D affine per spatial axis, and an
/// arbitrary non-spatial axis, so every fixed-width stored sample type and
/// every coordinate map RITK models survives a round trip. Calibration does
/// not: NRRD has no field for a stored-to-real transform, so only the identity
/// category is declared and any other calibration is reported as an
/// unsupported feature at its volume.
const NRRD_FEATURES: &[ConversionFeature] = &[
    ConversionFeature::SampleType(SampleType::U8),
    ConversionFeature::SampleType(SampleType::I8),
    ConversionFeature::SampleType(SampleType::U16),
    ConversionFeature::SampleType(SampleType::I16),
    ConversionFeature::SampleType(SampleType::U32),
    ConversionFeature::SampleType(SampleType::I32),
    ConversionFeature::SampleType(SampleType::U64),
    ConversionFeature::SampleType(SampleType::I64),
    ConversionFeature::SampleType(SampleType::F32),
    ConversionFeature::SampleType(SampleType::F64),
    ConversionFeature::PhysicalGeometry,
    ConversionFeature::CartesianCoordinates,
    ConversionFeature::CurvilinearArrayCoordinates,
    ConversionFeature::PhasedArray3DCoordinates,
    ConversionFeature::SliceSeriesCoordinates,
    ConversionFeature::IdentityCalibration,
    ConversionFeature::SingleVolumeAxis,
    ConversionFeature::ListAxis,
    ConversionFeature::UnspecifiedAxis,
    ConversionFeature::DiffusionAxis,
];

/// The NRRD stored-series target, carrying the caller's metadata records.
struct NrrdStoredSeriesTarget<'a> {
    comments: &'a [String],
    records: &'a [(String, String)],
}

/// Target-owned header bytes a validated NRRD write emits verbatim.
struct NrrdStoredSeriesPlan {
    header: Vec<u8>,
}

impl NrrdStoredSeriesPlan {
    /// Returns the serialized header the adapter validated.
    fn header(&self) -> &[u8] {
        &self.header
    }
}

impl ConversionTarget for NrrdStoredSeriesTarget<'_> {
    const FORMAT: &'static str = "nrrd";
    const FEATURES: &'static [ConversionFeature] = NRRD_FEATURES;
}

impl ConversionAdapter for NrrdStoredSeriesTarget<'_> {
    type Plan = NrrdStoredSeriesPlan;
    type Rejection = NrrdStoredSeriesRejection;

    fn prepare(&self, series: &StoredSeries) -> Result<Self::Plan, Self::Rejection> {
        let volumes = series.volumes();
        let Some(first) = volumes.first() else {
            return Err(NrrdStoredSeriesRejection::EmptySeries);
        };
        validate_series_axis(series.axis()).map_err(map_axis_rejection)?;
        validate_series_header_entries(series.axis(), first.coordinate_map())
            .map_err(map_header_rejection)?;
        validate_calibration(first)
            .map_err(|_| NrrdStoredSeriesRejection::UnsupportedCalibration { volume_index: 0 })?;
        validate_physical_geometry(first.metadata()).map_err(|source| {
            NrrdStoredSeriesRejection::PhysicalGeometry {
                volume_index: 0,
                source,
            }
        })?;
        validate_coordinate_map(first.coordinate_map(), first.shape()).map_err(|source| {
            NrrdStoredSeriesRejection::CoordinateMap {
                volume_index: 0,
                source,
            }
        })?;
        let sample_type = first.samples().sample_type();
        let element_type = nrrd_type_name(sample_type)
            .map_err(|_| NrrdStoredSeriesRejection::UnsupportedSampleType { sample_type })?;
        for (volume_index, volume) in volumes.iter().enumerate().skip(1) {
            validate_calibration(volume)
                .map_err(|_| NrrdStoredSeriesRejection::UnsupportedCalibration { volume_index })?;
            if volume.shape() != first.shape() {
                return Err(NrrdStoredSeriesRejection::ShapeMismatch {
                    volume_index,
                    expected: first.shape(),
                    actual: volume.shape(),
                });
            }
            if volume.metadata() != first.metadata() {
                return Err(NrrdStoredSeriesRejection::GeometryMismatch { volume_index });
            }
            if volume.coordinate_map() != first.coordinate_map() {
                return Err(NrrdStoredSeriesRejection::CoordinateMapMismatch { volume_index });
            }
            if volume.samples().sample_type() != sample_type {
                return Err(NrrdStoredSeriesRejection::SampleTypeMismatch { volume_index });
            }
        }
        let metadata = first.metadata();
        let header = build_series_header(
            first.shape(),
            volumes.len(),
            metadata.spacing(),
            metadata.origin(),
            metadata.direction(),
            element_type,
            first.coordinate_map(),
            series.axis(),
            self.comments,
            self.records,
        )?;
        Ok(NrrdStoredSeriesPlan { header })
    }
}

/// Serializes the NRRD header, rejecting one that would exceed reader bounds.
///
/// The document writer and the stored-series adapter both emit through this
/// function, so the emitted field set has one definition. A header that the
/// bounded `HeaderBuffer` truncated, or whose entry count exceeds the reader's
/// limit, is rejected before any destination is opened.
pub(crate) fn build_series_header(
    shape: [usize; 3],
    volume_count: usize,
    spacing: &Spacing<3>,
    origin: &Point<3>,
    direction: &Direction<3>,
    element_type: &str,
    coordinate_map: &CoordinateMap,
    axis: &SeriesAxis,
    comments: &[String],
    records: &[(String, String)],
) -> Result<Vec<u8>, NrrdStoredSeriesRejection> {
    let mut header = HeaderBuffer::new();
    let result = if matches!(axis, SeriesAxis::SingleVolume) {
        write_nrrd_header_with_metadata(
            &mut header,
            shape,
            spacing,
            origin,
            direction,
            element_type,
            coordinate_map,
            comments,
            records,
        )
    } else {
        write_nrrd_series_header_with_metadata(
            &mut header,
            shape,
            volume_count,
            spacing,
            origin,
            direction,
            element_type,
            coordinate_map,
            SeriesLayout::AcquisitionSlowest,
            axis,
            comments,
            records,
        )
    };
    if header.exceeded_limit() || result.is_err() {
        return Err(header_too_large());
    }
    let entries = header
        .bytes()
        .split(|byte| *byte == b'\n')
        .skip(1)
        .take_while(|line| !line.is_empty())
        .count();
    if entries > crate::reader::MAX_HEADER_ENTRIES {
        return Err(NrrdStoredSeriesRejection::HeaderTooManyEntries {
            entries,
            maximum_entries: crate::reader::MAX_HEADER_ENTRIES,
        });
    }
    Ok(header.into_bytes())
}

/// Runs the stored-series preflight and returns the header it validated.
pub(crate) fn prepare_document_header(
    series: &StoredSeries,
    comments: &[String],
    records: &[(String, String)],
) -> Result<Vec<u8>, NrrdStoredSeriesError> {
    let target = NrrdStoredSeriesTarget { comments, records };
    let prepared = prepare_conversion(&target, NRRD_DOCUMENT_SOURCE, series, std::iter::empty())
        .map_err(NrrdStoredSeriesError::Preparation)?;
    Ok(prepared.plan().header().to_vec())
}

impl NrrdDocument {
    /// Builds a NRRD document from stored samples and metadata.
    ///
    /// Every input-dependent constraint NRRD imposes is checked here, before
    /// any destination exists: the series must be non-empty, every volume must
    /// share the first volume's shape, sample type, physical geometry, and
    /// coordinate map, calibration must be identity, and the serialized header
    /// must fit the reader's bounds. `metadata_losses` lets the source adapter
    /// report fields `StoredSeries` cannot carry.
    ///
    /// # Errors
    ///
    /// Returns the scoped capability report or the target's typed rejection
    /// before output is opened, or a metadata conflict in `comments` and
    /// `records`.
    pub fn from_stored_series(
        source_format: &'static str,
        series: StoredSeries,
        comments: Vec<String>,
        records: Vec<(String, String)>,
        metadata_losses: impl IntoIterator<Item = FormatMetadataLoss>,
    ) -> Result<Self, NrrdStoredSeriesError> {
        validate_document_metadata(
            &comments,
            &records,
            matches!(series.axis(), SeriesAxis::Diffusion(_)),
        )?;
        let target = NrrdStoredSeriesTarget {
            comments: &comments,
            records: &records,
        };
        // The preflight runs before the document exists so a rejection cannot
        // leave a destination behind. Its plan is not retained: `write_to`
        // rebuilds the header through the same target at output time, so the
        // header has one producer and this call only gates construction.
        drop(
            prepare_conversion(&target, source_format, &series, metadata_losses)
                .map_err(NrrdStoredSeriesError::Preparation)?,
        );
        Ok(NrrdDocument::from_validated(series, comments, records))
    }
}

fn header_too_large() -> NrrdStoredSeriesRejection {
    let maximum_bytes = crate::reader::MAX_HEADER_BYTES;
    NrrdStoredSeriesRejection::HeaderTooLarge {
        header_bytes: maximum_bytes.saturating_add(1),
        maximum_bytes,
    }
}

fn map_axis_rejection(error: NrrdStoredWriteError) -> NrrdStoredSeriesRejection {
    match error {
        NrrdStoredWriteError::UnsupportedGradientFrame { frame } => {
            NrrdStoredSeriesRejection::UnsupportedGradientFrame { frame }
        }
        NrrdStoredWriteError::UnrepresentableDiffusionWeighting { index } => {
            NrrdStoredSeriesRejection::UnrepresentableDiffusionWeighting {
                volume_index: index,
            }
        }
        _ => NrrdStoredSeriesRejection::UnsupportedAxisMeaning,
    }
}

fn map_header_rejection(error: NrrdStoredWriteError) -> NrrdStoredSeriesRejection {
    match error {
        NrrdStoredWriteError::HeaderTooManyEntries {
            entries,
            maximum_entries,
        } => NrrdStoredSeriesRejection::HeaderTooManyEntries {
            entries,
            maximum_entries,
        },
        _ => NrrdStoredSeriesRejection::UnsupportedAxisMeaning,
    }
}

/// A stored-series value that NRRD cannot encode.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NrrdStoredSeriesRejection {
    /// The series has no volume to write.
    #[error("an NRRD stored series must contain at least one volume")]
    EmptySeries,
    /// A volume's stored-to-real transform has no NRRD representation.
    #[error("volume {volume_index} has a non-identity intensity calibration")]
    UnsupportedCalibration {
        /// The zero-based volume that cannot be represented.
        volume_index: usize,
    },
    /// A stored sample representation has no NRRD element type.
    #[error("NRRD cannot represent stored sample type {sample_type:?}")]
    UnsupportedSampleType {
        /// The sample representation without an NRRD element type.
        sample_type: SampleType,
    },
    /// A physical grid cannot be represented by NRRD space directions.
    #[error("volume {volume_index} has geometry NRRD cannot represent: {source}")]
    PhysicalGeometry {
        /// The zero-based volume whose geometry is invalid.
        volume_index: usize,
        /// Shared image-I/O geometry-contract failure.
        #[source]
        source: VolumeError,
    },
    /// A coordinate map is inconsistent with the volume shape.
    #[error("volume {volume_index} has an invalid coordinate map: {source}")]
    CoordinateMap {
        /// The zero-based volume whose coordinate map is invalid.
        volume_index: usize,
        /// Shared image-I/O coordinate-map failure.
        #[source]
        source: VolumeError,
    },
    /// The acquisition-axis meaning has no NRRD representation.
    #[error("NRRD cannot encode this acquisition-axis meaning")]
    UnsupportedAxisMeaning,
    /// A diffusion frame cannot be encoded in an NRRD physical frame.
    #[error("NRRD cannot encode diffusion frame {frame:?}")]
    UnsupportedGradientFrame {
        /// Frame declared by the diffusion scheme.
        frame: GradientFrame,
    },
    /// A nonzero diffusion weighting underflows when represented by gradients.
    #[error("volume {volume_index} has a diffusion weighting NRRD cannot represent")]
    UnrepresentableDiffusionWeighting {
        /// The zero-based volume whose weighting underflows.
        volume_index: usize,
    },
    /// A later volume has a different spatial shape.
    #[error("volume {volume_index} has shape {actual:?}, expected {expected:?}")]
    ShapeMismatch {
        /// The zero-based volume that differs.
        volume_index: usize,
        /// The first volume's depth, row, column shape.
        expected: [usize; 3],
        /// The rejected depth, row, column shape.
        actual: [usize; 3],
    },
    /// A later volume has a different physical grid.
    #[error("volume {volume_index} has geometry different from volume 0")]
    GeometryMismatch {
        /// The zero-based volume that differs.
        volume_index: usize,
    },
    /// A later volume has a different coordinate map.
    #[error("volume {volume_index} has a coordinate map different from volume 0")]
    CoordinateMapMismatch {
        /// The zero-based volume that differs.
        volume_index: usize,
    },
    /// A later volume uses a different stored sample representation.
    #[error("volume {volume_index} uses a different stored sample type")]
    SampleTypeMismatch {
        /// The zero-based volume that differs.
        volume_index: usize,
    },
    /// The serialized header exceeds the reader's bounded header size.
    #[error("NRRD output header exceeds {maximum_bytes} bytes (at least {header_bytes} bytes)")]
    HeaderTooLarge {
        /// Lower bound on the serialized header length.
        header_bytes: usize,
        /// Largest header length accepted by the reader.
        maximum_bytes: usize,
    },
    /// The serialized header exceeds the reader's entry-count bound.
    #[error(
        "NRRD output header has {entries} metadata entries; the reader limit is {maximum_entries}"
    )]
    HeaderTooManyEntries {
        /// Number of fields and key/value entries the writer would emit.
        entries: usize,
        /// Largest entry count accepted by the reader.
        maximum_entries: usize,
    },
}

impl ConversionRejection for NrrdStoredSeriesRejection {
    fn location(&self) -> ConversionLocation {
        match self {
            Self::EmptySeries
            | Self::UnsupportedSampleType { .. }
            | Self::UnsupportedAxisMeaning
            | Self::UnsupportedGradientFrame { .. }
            | Self::HeaderTooLarge { .. }
            | Self::HeaderTooManyEntries { .. } => ConversionLocation::Series,
            Self::UnsupportedCalibration { volume_index }
            | Self::UnrepresentableDiffusionWeighting { volume_index }
            | Self::PhysicalGeometry { volume_index, .. }
            | Self::CoordinateMap { volume_index, .. }
            | Self::ShapeMismatch { volume_index, .. }
            | Self::GeometryMismatch { volume_index }
            | Self::CoordinateMapMismatch { volume_index }
            | Self::SampleTypeMismatch { volume_index } => ConversionLocation::Volume {
                volume_index: *volume_index,
            },
        }
    }
}

/// Failure to build a validated NRRD document from stored image values.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NrrdStoredSeriesError {
    /// Preflight found a declared loss or a target-specific rejection.
    #[error("NRRD conversion preflight failed: {0}")]
    Preparation(#[source] ConversionPrepareError<NrrdStoredSeriesRejection>),
    /// The caller's comments or records conflict with generated NRRD fields.
    ///
    /// Boxed so this error and [`NrrdDocumentError`] do not form an infinitely
    /// sized pair: each type reports a failure of the other.
    #[error(transparent)]
    Document(Box<NrrdDocumentError>),
}

impl From<NrrdDocumentError> for NrrdStoredSeriesError {
    fn from(error: NrrdDocumentError) -> Self {
        Self::Document(Box::new(error))
    }
}

#[cfg(test)]
mod tests;
