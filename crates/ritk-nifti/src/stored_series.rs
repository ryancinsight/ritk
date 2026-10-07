//! Preflighted NIfTI documents from shared stored samples.

use crate::document::{NiftiDocumentError, MAX_DOCUMENT_BYTES};
use crate::header::{
    HeaderAxis, HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader,
};
use crate::spatial::{sform_from_internal_lps_metadata, sform_handedness};
use crate::{NiftiDocument, NiftiVersion};
use ritk_codecs::{ByteOrder, SampleError, SampleType};
use ritk_image_io::{
    prepare_conversion, ConversionAdapter, ConversionFeature, ConversionLocation,
    ConversionPrepareError, ConversionRejection, ConversionTarget, FormatMetadataLoss,
    IntensityCalibration, LinearCalibration, SeriesAxis, StoredSeries,
};
use std::collections::TryReserveError;
use thiserror::Error;

const NIFTI_FEATURES: &[ConversionFeature] = &[
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
    ConversionFeature::IdentityCalibration,
    ConversionFeature::LinearCalibration,
    ConversionFeature::PerFrameLinearCalibration,
    ConversionFeature::SingleVolumeAxis,
    ConversionFeature::ListAxis,
    ConversionFeature::UnspecifiedAxis,
];

struct NiftiStoredSeriesTarget {
    version: NiftiVersion,
}

struct NiftiStoredSeriesPlan {
    header: NiftiHeader,
    document_bytes: usize,
}

impl ConversionTarget for NiftiStoredSeriesTarget {
    const FORMAT: &'static str = "nifti";
    const FEATURES: &'static [ConversionFeature] = NIFTI_FEATURES;
}

impl ConversionAdapter for NiftiStoredSeriesTarget {
    type Plan = NiftiStoredSeriesPlan;
    type Rejection = NiftiStoredSeriesRejection;

    fn prepare(&self, series: &StoredSeries) -> Result<Self::Plan, Self::Rejection> {
        let first = series
            .volumes()
            .first()
            .expect("invariant: StoredSeries contains at least one volume");
        let shape = first.shape();
        let sample_type = first.samples().sample_type();
        let scaling = scaling_for_volume(first.calibration(), 0)?;

        for (volume_index, volume) in series.volumes().iter().enumerate().skip(1) {
            if volume.shape() != shape {
                return Err(NiftiStoredSeriesRejection::ShapeMismatch {
                    volume_index,
                    expected: shape,
                    actual: volume.shape(),
                });
            }
            let actual_sample_type = volume.samples().sample_type();
            if actual_sample_type != sample_type {
                return Err(NiftiStoredSeriesRejection::SampleTypeMismatch {
                    volume_index,
                    expected: sample_type,
                    actual: actual_sample_type,
                });
            }
            if !same_geometry(first, volume) {
                return Err(NiftiStoredSeriesRejection::GeometryMismatch { volume_index });
            }
            let volume_scaling = scaling_for_volume(volume.calibration(), volume_index)?;
            if volume_scaling != scaling {
                return Err(NiftiStoredSeriesRejection::CalibrationMismatch { volume_index });
            }
        }

        let [depth, rows, columns] = shape;
        let dimensions = HeaderDims {
            nx: columns,
            ny: rows,
            nz: depth,
        };
        let axis = match series.axis() {
            SeriesAxis::SingleVolume => HeaderAxis::Volume,
            SeriesAxis::List | SeriesAxis::Unspecified => HeaderAxis::Acquisition,
            SeriesAxis::Diffusion(_) => {
                return Err(NiftiStoredSeriesRejection::UnsupportedAxis);
            }
            _ => return Err(NiftiStoredSeriesRejection::UnsupportedAxis),
        };
        let metadata = first.metadata();
        let origin = metadata.origin().as_slice();
        let spacing = metadata.spacing().to_array();
        let direction = metadata.direction().0;
        let direction_row_major = [
            direction[(0, 0)],
            direction[(0, 1)],
            direction[(0, 2)],
            direction[(1, 0)],
            direction[(1, 1)],
            direction[(1, 2)],
            direction[(2, 0)],
            direction[(2, 1)],
            direction[(2, 2)],
        ];
        let sform = sform_from_internal_lps_metadata(
            [origin[0], origin[1], origin[2]],
            spacing,
            direction_row_major,
        );
        let version = match self.version {
            NiftiVersion::One => HeaderVersion::One,
            NiftiVersion::Two => HeaderVersion::Two,
        };
        validate_volume_count(self.version, series.volumes().len())?;
        let mut header = NiftiHeader::new_with_version(
            version,
            dimensions,
            series.volumes().len(),
            axis,
            NiftiDatatype::try_from(sample_type)
                .map_err(|_| NiftiStoredSeriesRejection::UnsupportedSampleType { sample_type })?,
            HeaderSpatial {
                pixdim: [1.0, spacing[2], spacing[1], spacing[0], 1.0, 1.0, 1.0, 1.0],
                srow_x: sform.x,
                srow_y: sform.y,
                srow_z: sform.z,
            },
        )
        .map_err(NiftiStoredSeriesRejection::HeaderEncoding)?;
        header.scl_slope = scaling.slope();
        header.scl_inter = scaling.intercept();
        header
            .validate_for_encoding()
            .map_err(NiftiStoredSeriesRejection::HeaderEncoding)?;
        if header.sform_code > 0 {
            let encoded_header = NiftiHeader::parse(&header.encode())
                .map_err(NiftiStoredSeriesRejection::HeaderEncoding)?;
            if sform_handedness([
                encoded_header.srow_x,
                encoded_header.srow_y,
                encoded_header.srow_z,
            ])
            .is_none()
            {
                return Err(NiftiStoredSeriesRejection::HeaderEncoding(anyhow::anyhow!(
                    "NIfTI sform becomes singular after header encoding"
                )));
            }
        }

        let voxel_count = shape
            .into_iter()
            .try_fold(1_usize, |count, dimension| count.checked_mul(dimension));
        let payload_bytes = voxel_count
            .and_then(|voxels| voxels.checked_mul(series.volumes().len()))
            .and_then(|voxels| voxels.checked_mul(sample_type.byte_width()))
            .ok_or(NiftiStoredSeriesRejection::DocumentSizeOverflow)?;
        let document_bytes = header
            .vox_offset
            .checked_add(payload_bytes)
            .ok_or(NiftiStoredSeriesRejection::DocumentSizeOverflow)?;
        let document_bytes_u64 =
            u64::try_from(document_bytes).expect("invariant: document byte length fits u64");
        if document_bytes_u64 > MAX_DOCUMENT_BYTES {
            return Err(NiftiStoredSeriesRejection::DocumentSizeLimit {
                actual: document_bytes_u64,
                maximum: MAX_DOCUMENT_BYTES,
            });
        }

        Ok(NiftiStoredSeriesPlan {
            header,
            document_bytes,
        })
    }
}

impl NiftiDocument {
    /// Construct a NIfTI document from stored series samples and metadata.
    ///
    /// The returned document owns its bytes and can be written or transcoded
    /// without an intermediate file. The source identifier and declared losses
    /// let a format adapter report fields not retained by [`StoredSeries`].
    /// NIfTI-1 stores spatial and scaling fields as 32-bit floats; NIfTI-2
    /// stores them as 64-bit floats.
    ///
    /// # Errors
    ///
    /// Returns the scoped capability report or target rejection before any
    /// destination exists, a bounded-allocation failure, a sample encoding
    /// error, or a malformed constructed document.
    pub fn from_stored_series(
        source_format: &'static str,
        series: &StoredSeries,
        version: NiftiVersion,
        metadata_losses: impl IntoIterator<Item = FormatMetadataLoss>,
    ) -> Result<Self, NiftiStoredSeriesError> {
        let target = NiftiStoredSeriesTarget { version };
        let prepared = prepare_conversion(&target, source_format, series, metadata_losses)
            .map_err(NiftiStoredSeriesError::Preparation)?;
        let plan = prepared.plan();
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(plan.document_bytes)
            .map_err(NiftiStoredSeriesError::Allocation)?;
        bytes.extend_from_slice(&plan.header.encode());
        bytes.resize(plan.header.vox_offset, 0);
        for volume in prepared.series().volumes() {
            volume
                .samples()
                .write_to(&mut bytes, ByteOrder::LeastSignificantByteFirst)
                .map_err(NiftiStoredSeriesError::SampleEncoding)?;
        }
        if bytes.len() != plan.document_bytes {
            return Err(NiftiStoredSeriesError::PayloadLengthMismatch {
                expected: plan.document_bytes,
                actual: bytes.len(),
            });
        }
        Self::from_owned_bytes(bytes).map_err(NiftiStoredSeriesError::Document)
    }
}

fn same_geometry(
    first: &ritk_image_io::StoredVolume,
    candidate: &ritk_image_io::StoredVolume,
) -> bool {
    first.metadata().origin().as_slice() == candidate.metadata().origin().as_slice()
        && first.metadata().spacing().to_array() == candidate.metadata().spacing().to_array()
        && first.metadata().direction().iter().copied().eq(candidate
            .metadata()
            .direction()
            .iter()
            .copied())
}

fn validate_volume_count(
    version: NiftiVersion,
    volume_count: usize,
) -> Result<(), NiftiStoredSeriesRejection> {
    let representable = match version {
        NiftiVersion::One => i16::try_from(volume_count).is_ok(),
        NiftiVersion::Two => i64::try_from(volume_count).is_ok(),
    };
    if representable {
        Ok(())
    } else {
        Err(NiftiStoredSeriesRejection::VolumeCountOutOfRange {
            version,
            volume_count,
        })
    }
}

fn scaling_for_volume(
    calibration: &IntensityCalibration,
    volume_index: usize,
) -> Result<LinearCalibration, NiftiStoredSeriesRejection> {
    let scaling = match calibration {
        IntensityCalibration::Identity => {
            LinearCalibration::new(1.0, 0.0).expect("identity calibration is finite")
        }
        IntensityCalibration::Linear(linear) => *linear,
        IntensityCalibration::PerFrameLinear(frames) => {
            let Some(first) = frames.first().copied() else {
                return Err(NiftiStoredSeriesRejection::EmptyFrameCalibration { volume_index });
            };
            for (frame_index, frame) in frames.iter().copied().enumerate().skip(1) {
                if frame != first {
                    return Err(NiftiStoredSeriesRejection::FrameCalibrationMismatch {
                        volume_index,
                        frame_index,
                    });
                }
            }
            first
        }
        IntensityCalibration::ModalityLookup(_) => {
            return Err(NiftiStoredSeriesRejection::UnsupportedCalibration { volume_index });
        }
    };
    if scaling.slope() == 0.0 {
        return Err(NiftiStoredSeriesRejection::ZeroSlopeCalibration { volume_index });
    }
    Ok(scaling)
}

/// A stored-series value that the selected NIfTI version cannot encode.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NiftiStoredSeriesRejection {
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
    /// A later volume uses another fixed-width sample representation.
    #[error("volume {volume_index} uses {actual:?} samples, expected {expected:?}")]
    SampleTypeMismatch {
        /// The zero-based volume that differs.
        volume_index: usize,
        /// The first volume's sample representation.
        expected: SampleType,
        /// The rejected sample representation.
        actual: SampleType,
    },
    /// The codec mapping does not support the stored sample type.
    #[error("NIfTI cannot encode stored sample type {sample_type:?}")]
    UnsupportedSampleType {
        /// The stored scalar representation rejected by the codec mapping.
        sample_type: SampleType,
    },
    /// A later volume has a different physical grid.
    #[error("volume {volume_index} has geometry different from volume 0")]
    GeometryMismatch {
        /// The zero-based volume that differs.
        volume_index: usize,
    },
    /// A later volume has a different intensity scale.
    #[error("volume {volume_index} has calibration different from volume 0")]
    CalibrationMismatch {
        /// The zero-based volume that differs.
        volume_index: usize,
    },
    /// One frame in a volume uses a different scale than the first frame.
    #[error("volume {volume_index} frame {frame_index} has a different calibration")]
    FrameCalibrationMismatch {
        /// The zero-based volume containing the differing frame.
        volume_index: usize,
        /// The zero-based frame that differs.
        frame_index: usize,
    },
    /// A per-frame calibration has no frame entries.
    #[error("volume {volume_index} has an empty per-frame calibration")]
    EmptyFrameCalibration {
        /// The zero-based volume containing the calibration.
        volume_index: usize,
    },
    /// A constant transform cannot use NIfTI's zero-slope representation.
    #[error("volume {volume_index} has zero-slope calibration, which NIfTI treats as disabled")]
    ZeroSlopeCalibration {
        /// The zero-based volume containing the calibration.
        volume_index: usize,
    },
    /// A nonlinear lookup calibration has no NIfTI scalar scaling equivalent.
    #[error("volume {volume_index} uses a modality lookup table")]
    UnsupportedCalibration {
        /// The zero-based volume containing the calibration.
        volume_index: usize,
    },
    /// The series axis has no supported NIfTI dimension representation.
    #[error("NIfTI cannot preserve the stored series axis")]
    UnsupportedAxis,
    /// The acquisition axis count does not fit the selected NIfTI header.
    #[error("NIfTI-{version:?} cannot encode an acquisition axis with {volume_count} volumes")]
    VolumeCountOutOfRange {
        /// Header version selected for the output document.
        version: NiftiVersion,
        /// Number of stored volumes on the acquisition axis.
        volume_count: usize,
    },
    /// A value needed for the NIfTI header exceeds its selected version.
    #[error("NIfTI header cannot represent the stored series: {0}")]
    HeaderEncoding(#[source] anyhow::Error),
    /// Computing the document's encoded size overflowed `usize`.
    #[error("NIfTI stored-series document size overflows usize")]
    DocumentSizeOverflow,
    /// The complete document exceeds the bounded NIfTI document size.
    #[error("NIfTI stored-series document has {actual} bytes; maximum is {maximum}")]
    DocumentSizeLimit {
        /// Computed uncompressed document size.
        actual: u64,
        /// Largest supported uncompressed document size.
        maximum: u64,
    },
}

impl ConversionRejection for NiftiStoredSeriesRejection {
    fn location(&self) -> ConversionLocation {
        match self {
            Self::FrameCalibrationMismatch {
                volume_index,
                frame_index,
            } => ConversionLocation::Frame {
                volume_index: *volume_index,
                frame_index: *frame_index,
            },
            Self::ShapeMismatch { volume_index, .. }
            | Self::SampleTypeMismatch { volume_index, .. }
            | Self::GeometryMismatch { volume_index }
            | Self::CalibrationMismatch { volume_index }
            | Self::EmptyFrameCalibration { volume_index }
            | Self::ZeroSlopeCalibration { volume_index }
            | Self::UnsupportedCalibration { volume_index } => ConversionLocation::Volume {
                volume_index: *volume_index,
            },
            Self::UnsupportedSampleType { .. }
            | Self::UnsupportedAxis
            | Self::VolumeCountOutOfRange { .. }
            | Self::DocumentSizeOverflow
            | Self::DocumentSizeLimit { .. } => ConversionLocation::Series,
            Self::HeaderEncoding(_) => ConversionLocation::Volume { volume_index: 0 },
        }
    }
}

#[cfg(test)]
#[path = "stored_series_tests.rs"]
mod tests;

/// Failure to build a validated NIfTI document from stored image values.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum NiftiStoredSeriesError {
    /// Preflight found a declared loss or target-specific rejection.
    #[error("NIfTI conversion preflight failed: {0}")]
    Preparation(#[source] ConversionPrepareError<NiftiStoredSeriesRejection>),
    /// The output document allocation could not be reserved.
    #[error("cannot reserve NIfTI stored-series output: {0}")]
    Allocation(#[source] TryReserveError),
    /// A sample codec failed while writing exact stored values.
    #[error("cannot encode stored NIfTI samples: {0}")]
    SampleEncoding(#[source] SampleError),
    /// The encoded payload length differed from the validated plan.
    #[error("NIfTI payload length is {actual}, expected {expected}")]
    PayloadLengthMismatch {
        /// The planned complete document byte length.
        expected: usize,
        /// The actual complete document byte length.
        actual: usize,
    },
    /// The constructed bytes did not satisfy the NIfTI document parser.
    #[error("constructed NIfTI document is invalid: {0}")]
    Document(#[source] NiftiDocumentError),
}
