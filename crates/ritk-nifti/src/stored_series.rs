//! Lossless stored-series conversion to a NIfTI single-file document.

use crate::document::{NiftiDocument, NiftiVersion, MAX_DOCUMENT_BYTES};
use crate::header::{HeaderDims, HeaderSpatial, HeaderVersion, NiftiDatatype, NiftiHeader};
use crate::spatial::sform_from_internal_lps_metadata;
use ritk_codecs::{ByteOrder as SampleByteOrder, SampleType};
use ritk_image_io::{
    ConversionAdapter, ConversionFeature, ConversionLocation, ConversionPrepareError,
    ConversionTarget, IntensityCalibration, SeriesAxis, StoredSeries,
};

mod error;
pub use error::{
    NiftiConversionPreparationError, NiftiSampleEncodingError, NiftiStoredSeriesError,
    NiftiStoredSeriesIssue, NiftiStoredSeriesRejection,
};

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
];

struct NiftiTarget {
    version: HeaderVersion,
}

struct NiftiWritePlan {
    header: Vec<u8>,
    voxel_offset: usize,
    document_bytes: usize,
}

impl ConversionTarget for NiftiTarget {
    const FORMAT: &'static str = "nifti";
    const FEATURES: &'static [ConversionFeature] = NIFTI_FEATURES;
}

impl ConversionAdapter for NiftiTarget {
    type Plan = NiftiWritePlan;
    type Rejection = NiftiStoredSeriesRejection;

    fn prepare(&self, series: &StoredSeries) -> Result<Self::Plan, Self::Rejection> {
        if matches!(series.axis(), SeriesAxis::List) && series.volumes().len() == 1 {
            return Err(reject(
                ConversionLocation::Series,
                NiftiStoredSeriesIssue::ListAxisRequiresMultipleVolumes,
            ));
        }
        let first = series
            .volumes()
            .first()
            .expect("invariant: StoredSeries rejects empty volume lists");
        let shape = first.shape();
        let sample_type = first.samples().sample_type();
        let datatype = datatype(sample_type).ok_or_else(|| {
            reject(
                ConversionLocation::Volume { volume_index: 0 },
                NiftiStoredSeriesIssue::UnsupportedSampleType { sample_type },
            )
        })?;
        let calibration = calibration_mapping(first.calibration(), 0)?;

        for (volume_index, volume) in series.volumes().iter().enumerate().skip(1) {
            let location = ConversionLocation::Volume { volume_index };
            if volume.shape() != shape {
                return Err(reject(
                    location,
                    NiftiStoredSeriesIssue::ShapeMismatch {
                        expected: shape,
                        actual: volume.shape(),
                    },
                ));
            }
            if volume.samples().sample_type() != sample_type {
                return Err(reject(
                    location,
                    NiftiStoredSeriesIssue::SampleTypeMismatch {
                        expected: sample_type,
                        actual: volume.samples().sample_type(),
                    },
                ));
            }
            if volume.metadata() != first.metadata() {
                return Err(reject(location, NiftiStoredSeriesIssue::GeometryMismatch));
            }
            let current = calibration_mapping(volume.calibration(), volume_index)?;
            if current != calibration {
                return Err(reject(
                    location,
                    NiftiStoredSeriesIssue::CalibrationMismatch,
                ));
            }
        }

        let spatial = nifti_spatial(first.metadata());
        let header = NiftiHeader::new_with_version(
            self.version,
            HeaderDims {
                nx: shape[2],
                ny: shape[1],
                nz: shape[0],
            },
            series.volumes().len(),
            datatype,
            spatial,
        )
        .map_err(|error| {
            reject(
                ConversionLocation::Series,
                NiftiStoredSeriesIssue::HeaderField {
                    detail: error.to_string().into_boxed_str(),
                },
            )
        })?;

        let (scl_slope, scl_inter) = if calibration == (1.0, 0.0) {
            (0.0, 0.0)
        } else {
            calibration
        };
        let mut header = header;
        header.scl_slope = scl_slope;
        header.scl_inter = scl_inter;
        header.ensure_scalar_fields().map_err(|error| {
            reject(
                ConversionLocation::Series,
                NiftiStoredSeriesIssue::HeaderField {
                    detail: error.to_string().into_boxed_str(),
                },
            )
        })?;
        let voxel_offset = header.vox_offset;
        let header = header.encode().map_err(|error| {
            reject(
                ConversionLocation::Series,
                NiftiStoredSeriesIssue::HeaderField {
                    detail: error.to_string().into_boxed_str(),
                },
            )
        })?;

        let voxels_per_volume = shape[0]
            .checked_mul(shape[1])
            .and_then(|plane| plane.checked_mul(shape[2]))
            .ok_or_else(|| {
                reject(
                    ConversionLocation::Series,
                    NiftiStoredSeriesIssue::PayloadSizeOverflow,
                )
            })?;
        let payload_bytes = voxels_per_volume
            .checked_mul(series.volumes().len())
            .and_then(|count| count.checked_mul(sample_type.byte_width()))
            .ok_or_else(|| {
                reject(
                    ConversionLocation::Series,
                    NiftiStoredSeriesIssue::PayloadSizeOverflow,
                )
            })?;
        let document_bytes = voxel_offset.checked_add(payload_bytes).ok_or_else(|| {
            reject(
                ConversionLocation::Series,
                NiftiStoredSeriesIssue::PayloadSizeOverflow,
            )
        })?;
        let limit = usize::try_from(MAX_DOCUMENT_BYTES)
            .expect("invariant: supported targets address at least one GiB");
        if document_bytes > limit {
            return Err(reject(
                ConversionLocation::Series,
                NiftiStoredSeriesIssue::DocumentTooLarge {
                    bytes: document_bytes,
                    limit,
                },
            ));
        }

        Ok(NiftiWritePlan {
            header,
            voxel_offset,
            document_bytes,
        })
    }
}

impl NiftiDocument {
    /// Builds a NIfTI-1 or NIfTI-2 document from exact stored samples.
    ///
    /// The conversion retains all ten RITK scalar sample types and their bit
    /// patterns. Single volumes and ordered lists with at least two volumes
    /// retain their axis shape. A singleton list is rejected because a rank-3
    /// NIfTI header would erase its list axis; unspecified and diffusion axes
    /// return typed capability losses. The volumes must use one shared
    /// Cartesian grid and representable calibration. NIfTI-1 stores spatial
    /// and calibration scalars as 32-bit floats, while NIfTI-2 stores them as
    /// 64-bit floats.
    ///
    /// # Errors
    ///
    /// Returns the capability report when the target cannot represent a
    /// declared semantic, a scoped rejection for inconsistent values, a
    /// sample-encoding error, a bounded-allocation error, or a document
    /// validation error.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_codecs::{ByteOrder, SampleBuffer, SampleType};
    /// use ritk_image::ImageMetadata;
    /// use ritk_image_io::{IntensityCalibration, SeriesAxis, StoredSeries, StoredVolume};
    /// use ritk_nifti::{NiftiDocument, NiftiVersion};
    /// use ritk_spatial::{CoordinateMap, Direction, Point, Spacing};
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let samples = SampleBuffer::decode(
    ///     SampleType::U16,
    ///     &[17, 0, 29, 0],
    ///     ByteOrder::LeastSignificantByteFirst,
    /// )?;
    /// let metadata = ImageMetadata::new(
    ///     Point::new([0.0; 3]),
    ///     Spacing::new([1.0; 3]),
    ///     Direction::identity(),
    /// );
    /// let volume = StoredVolume::new(
    ///     [1, 1, 2],
    ///     samples,
    ///     metadata,
    ///     CoordinateMap::Cartesian,
    ///     IntensityCalibration::Identity,
    /// )?;
    /// let series = StoredSeries::new(vec![volume], SeriesAxis::SingleVolume)?;
    /// let document = NiftiDocument::from_stored_series(&series, NiftiVersion::Two)?;
    /// assert_eq!(document.header().dimensions, [3, 2, 1, 1, 1, 1, 1, 1]);
    /// assert_eq!(document.sample_bytes(), &[17, 0, 29, 0]);
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_stored_series(
        series: &StoredSeries,
        version: NiftiVersion,
    ) -> Result<Self, NiftiStoredSeriesError> {
        let target = NiftiTarget {
            version: match version {
                NiftiVersion::One => HeaderVersion::One,
                NiftiVersion::Two => HeaderVersion::Two,
            },
        };
        let prepared =
            ritk_image_io::prepare_conversion(&target, "ritk-stored", series, std::iter::empty())
                .map_err(|error| match error {
                ConversionPrepareError::Capabilities(report) => {
                    NiftiStoredSeriesError::Capabilities(report)
                }
                ConversionPrepareError::Target { source, .. } => {
                    NiftiStoredSeriesError::Rejected(source)
                }
                source => NiftiStoredSeriesError::Preparation(
                    NiftiConversionPreparationError::new(source),
                ),
            })?;
        let plan = prepared.plan();
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(plan.document_bytes)
            .map_err(|source| NiftiStoredSeriesError::Allocation {
                bytes: plan.document_bytes,
                source,
            })?;
        bytes.extend_from_slice(&plan.header);
        bytes.resize(plan.voxel_offset, 0);
        for volume in prepared.series().volumes() {
            volume
                .samples()
                .write_to(&mut bytes, SampleByteOrder::LeastSignificantByteFirst)
                .map_err(|source| {
                    NiftiStoredSeriesError::SampleEncoding(NiftiSampleEncodingError::new(source))
                })?;
        }
        if bytes.len() != plan.document_bytes {
            return Err(NiftiStoredSeriesError::OutputSizeMismatch {
                expected: plan.document_bytes,
                actual: bytes.len(),
            });
        }
        Self::from_uncompressed_bytes(bytes).map_err(NiftiStoredSeriesError::Document)
    }
}

fn datatype(sample_type: SampleType) -> Option<NiftiDatatype> {
    match sample_type {
        SampleType::U8 => Some(NiftiDatatype::Uint8),
        SampleType::I8 => Some(NiftiDatatype::Int8),
        SampleType::U16 => Some(NiftiDatatype::Uint16),
        SampleType::I16 => Some(NiftiDatatype::Int16),
        SampleType::U32 => Some(NiftiDatatype::Uint32),
        SampleType::I32 => Some(NiftiDatatype::Int32),
        SampleType::U64 => Some(NiftiDatatype::Uint64),
        SampleType::I64 => Some(NiftiDatatype::Int64),
        SampleType::F32 => Some(NiftiDatatype::Float32),
        SampleType::F64 => Some(NiftiDatatype::Float64),
        _ => None,
    }
}

fn nifti_spatial(metadata: &ritk_image::ImageMetadata<3>) -> HeaderSpatial {
    let origin = metadata.origin().to_array();
    let spacing = metadata.spacing().to_array();
    let direction = metadata.direction();
    let direction_row_major = std::array::from_fn(|index| direction[(index / 3, index % 3)]);
    let sform = sform_from_internal_lps_metadata(origin, spacing, direction_row_major);
    let mut pixdim = [1.0; 8];
    pixdim[1] = spacing[2];
    pixdim[2] = spacing[1];
    pixdim[3] = spacing[0];
    HeaderSpatial {
        pixdim,
        srow_x: sform.x,
        srow_y: sform.y,
        srow_z: sform.z,
    }
}

fn calibration_mapping(
    calibration: &IntensityCalibration,
    volume_index: usize,
) -> Result<(f64, f64), NiftiStoredSeriesRejection> {
    let mapping = match calibration {
        IntensityCalibration::Identity => (1.0, 0.0),
        IntensityCalibration::Linear(linear) => (linear.slope(), linear.intercept()),
        IntensityCalibration::PerFrameLinear(frames) => {
            let first = *frames
                .first()
                .expect("invariant: stored depth is nonzero and frame calibration matches it");
            for (frame_index, frame) in frames.iter().enumerate().skip(1) {
                if *frame != first {
                    return Err(reject(
                        ConversionLocation::Frame {
                            volume_index,
                            frame_index,
                        },
                        NiftiStoredSeriesIssue::PerFrameCalibrationMismatch,
                    ));
                }
            }
            (first.slope(), first.intercept())
        }
        IntensityCalibration::ModalityLookup(_) => {
            return Err(reject(
                ConversionLocation::Volume { volume_index },
                NiftiStoredSeriesIssue::UnsupportedCalibration,
            ));
        }
    };

    if matches!(
        calibration,
        IntensityCalibration::Linear(_) | IntensityCalibration::PerFrameLinear(_)
    ) && mapping.0 == 0.0
    {
        return Err(reject(
            ConversionLocation::Volume { volume_index },
            NiftiStoredSeriesIssue::ZeroSlopeCalibration,
        ));
    }
    Ok(mapping)
}

fn reject(
    location: ConversionLocation,
    issue: NiftiStoredSeriesIssue,
) -> NiftiStoredSeriesRejection {
    NiftiStoredSeriesRejection { location, issue }
}

#[cfg(test)]
#[path = "stored_series_tests.rs"]
mod tests;
