//! Exact stored-sample conversion for Analyze 7.5 `.hdr`/`.img` pairs.
//!
//! [`crate::reader`] and [`crate::writer`] carry a volume as `f32` and apply the
//! header's intensity scale, which rounds wide integers and folds the source's
//! own sample representation into a compute scalar. This module is the stored
//! counterpart: it reads the payload as its exact fixed-width values beside the
//! same geometry, carries the header scale as an
//! [`IntensityCalibration`] instead of applying it, and refuses any series the
//! header cannot represent before a destination is created.
//!
//! Analyze 7.5 has no direction field, one intensity scale factor with no
//! additive term, and exactly one 3-D volume, so those limits are stated as
//! typed rejections rather than applied silently.

use anyhow::Context;
use ritk_codecs::{ByteOrder, SampleBuffer, SampleError, SampleType};
use ritk_image::ImageMetadata;
use ritk_image_io::{
    prepare_conversion, ConversionAdapter, ConversionFeature, ConversionLocation,
    ConversionPrepareError, ConversionRejection, ConversionTarget, FormatMetadataLoss,
    IntensityCalibration, LinearCalibration, PreparedConversion, SeriesAxis, StoredSeries,
    StoredSeriesError, StoredVolume, VolumeError,
};
use ritk_spatial::{CoordinateMap, Direction};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use thiserror::Error;

use crate::codec::HDR_SIZE;
use crate::header::{self, AnalyzeDatatype, AnalyzeHeaderFields};
use crate::reader::open_payload;

/// Source identifier reported when an Analyze import runs a conversion preflight.
pub const ANALYZE_STORED_SOURCE: &str = "analyze";

const ANALYZE_FEATURES: &[ConversionFeature] = &[
    ConversionFeature::SampleType(SampleType::U8),
    ConversionFeature::SampleType(SampleType::I16),
    ConversionFeature::SampleType(SampleType::I32),
    ConversionFeature::SampleType(SampleType::F32),
    ConversionFeature::SampleType(SampleType::F64),
    ConversionFeature::PhysicalGeometry,
    ConversionFeature::CartesianCoordinates,
    ConversionFeature::IdentityCalibration,
    ConversionFeature::LinearCalibration,
    ConversionFeature::SingleVolumeAxis,
];

/// Analyze 7.5's declared capability set, carrying the `descrip` bytes to emit.
#[derive(Debug)]
struct AnalyzeStoredSeriesTarget<'a> {
    description: &'a [u8],
}

/// The exact 348-byte header a validated series produces.
#[derive(Debug)]
struct AnalyzeStoredSeriesPlan {
    header: [u8; HDR_SIZE],
}

impl ConversionTarget for AnalyzeStoredSeriesTarget<'_> {
    const FORMAT: &'static str = "analyze";
    const FEATURES: &'static [ConversionFeature] = ANALYZE_FEATURES;
}

impl ConversionAdapter for AnalyzeStoredSeriesTarget<'_> {
    type Plan = AnalyzeStoredSeriesPlan;
    type Rejection = AnalyzeStoredSeriesRejection;

    fn prepare(&self, series: &StoredSeries) -> Result<Self::Plan, Self::Rejection> {
        // Analyze stores `dim[4] = 1`, so a series with any other acquisition
        // axis has no representation even when it holds one volume.
        let [volume] = series.volumes() else {
            return Err(AnalyzeStoredSeriesRejection::UnsupportedSeriesAxis);
        };
        if !matches!(series.axis(), SeriesAxis::SingleVolume) {
            return Err(AnalyzeStoredSeriesRejection::UnsupportedSeriesAxis);
        }

        let sample_type = volume.samples().sample_type();
        let datatype = AnalyzeDatatype::from_sample_type(sample_type).ok_or(
            AnalyzeStoredSeriesRejection::UnsupportedSampleType {
                volume_index: 0,
                sample_type,
            },
        )?;
        if !matches!(volume.coordinate_map(), CoordinateMap::Cartesian) {
            return Err(AnalyzeStoredSeriesRejection::UnsupportedCoordinateMap {
                volume_index: 0,
                reason: coordinate_map_name(volume.coordinate_map()),
            });
        }
        if *volume.metadata().direction() != Direction::identity() {
            return Err(AnalyzeStoredSeriesRejection::NonIdentityDirection { volume_index: 0 });
        }
        let scale = calibration_scale(volume.calibration(), 0)?;
        let header = header::encode(&AnalyzeHeaderFields {
            shape: volume.shape(),
            spacing: volume.metadata().spacing(),
            origin: volume.metadata().origin(),
            datatype,
            scale,
            description: self.description,
        })
        .map_err(AnalyzeStoredSeriesRejection::HeaderEncoding)?;
        Ok(AnalyzeStoredSeriesPlan { header })
    }
}

/// An Analyze 7.5 `.hdr`/`.img` pair held as exact stored samples.
#[derive(Debug)]
pub struct AnalyzeStoredSeries {
    series: StoredSeries,
    /// `descrip` provenance bytes re-emitted when the pair is written.
    pub(crate) description: Vec<u8>,
}

impl AnalyzeStoredSeries {
    /// Builds a pair representation from stored samples.
    ///
    /// The preflight runs here so a series the header cannot encode is rejected
    /// while no destination exists. `metadata_losses` lets the source adapter
    /// report fields [`StoredSeries`] cannot carry.
    ///
    /// # Errors
    ///
    /// Returns the scoped capability report or the target's typed rejection.
    pub fn from_stored_series(
        source_format: &'static str,
        series: StoredSeries,
        description: Vec<u8>,
        metadata_losses: impl IntoIterator<Item = FormatMetadataLoss>,
    ) -> Result<Self, AnalyzeStoredSeriesError> {
        let target = AnalyzeStoredSeriesTarget {
            description: &description,
        };
        // The plan is not retained: `write_analyze_stored` rebuilds the header
        // through the same target, so one producer owns the encoded bytes and
        // this call only gates construction.
        drop(
            prepare_conversion(&target, source_format, &series, metadata_losses)
                .map_err(AnalyzeStoredSeriesError::Preparation)?,
        );
        Ok(Self {
            series,
            description,
        })
    }

    /// Returns the stored samples with their validated physical geometry.
    #[must_use]
    pub fn series(&self) -> &StoredSeries {
        &self.series
    }

    /// Returns the `descrip` provenance bytes the pair records.
    #[must_use]
    pub fn description(&self) -> &[u8] {
        &self.description
    }

    /// Runs the shared conversion preflight for `target` against these samples.
    ///
    /// # Errors
    ///
    /// Returns the scoped capability report for the source and target formats,
    /// or the target's typed rejection.
    pub fn prepare_conversion<'a, T: ConversionAdapter>(
        &'a self,
        target: &'a T,
    ) -> Result<PreparedConversion<'a, T>, ConversionPrepareError<T::Rejection>> {
        prepare_conversion(target, ANALYZE_STORED_SOURCE, &self.series, [])
    }

    /// Re-runs the preflight and returns the header it produces.
    ///
    /// `AnalyzeStoredSeries`' fields are reachable from inside the crate, so a
    /// document can be mutated after construction. Validating here — and
    /// handing back a header only once every constraint holds — is what keeps a
    /// rejected write from creating either destination.
    fn validate_for_write(&self) -> Result<[u8; HDR_SIZE], AnalyzeStoredSeriesError> {
        let target = AnalyzeStoredSeriesTarget {
            description: &self.description,
        };
        prepare_conversion(&target, ANALYZE_STORED_SOURCE, &self.series, [])
            .map(|prepared| prepared.plan().header)
            .map_err(AnalyzeStoredSeriesError::Preparation)
    }
}

/// Reads an Analyze `.hdr`/`.img` pair as exact stored samples.
///
/// `path` may point to either file; the sibling is located by replacing the
/// extension. The payload is read in its declared representation, so no value
/// is rounded through `f32`, and the header's intensity scale travels as an
/// [`IntensityCalibration`] rather than being applied.
///
/// # Errors
///
/// Returns [`AnalyzeStoredReadError`] when the header or payload is malformed,
/// the payload cannot be read as its declared representation, or the resulting
/// geometry, coordinate map, or calibration is invalid.
pub fn read_analyze_stored<P: AsRef<Path>>(
    path: P,
) -> Result<AnalyzeStoredSeries, AnalyzeStoredReadError> {
    let path = path.as_ref();
    let header =
        header::parse(&path.with_extension("hdr")).map_err(AnalyzeStoredReadError::Header)?;
    let [nz, ny, nx] = header.shape;
    let voxel_count = nx
        .checked_mul(ny)
        .and_then(|plane| plane.checked_mul(nz))
        .context("Analyze voxel count overflows usize")
        .map_err(AnalyzeStoredReadError::Header)?;
    let mut img_file = open_payload(&path.with_extension("img"), &header, voxel_count)
        .map_err(AnalyzeStoredReadError::Payload)?;
    let samples = SampleBuffer::read_from(
        header.datatype.sample_type(),
        &mut img_file,
        voxel_count,
        ByteOrder::LeastSignificantByteFirst,
    )
    .map_err(AnalyzeStoredReadError::Samples)?;

    // A stored scale of one is the identity transform the header omits; any
    // other value is the multiplicative term of a zero-intercept linear map.
    let calibration = if header.scale == 1.0 {
        IntensityCalibration::Identity
    } else {
        IntensityCalibration::Linear(
            LinearCalibration::new(f64::from(header.scale), 0.0)
                .expect("invariant: a finite header scale is a valid slope"),
        )
    };
    let volume = StoredVolume::new(
        header.shape,
        samples,
        ImageMetadata::new(header.origin, header.spacing, Direction::identity()),
        CoordinateMap::Cartesian,
        calibration,
    )?;
    let series = StoredSeries::new(vec![volume], SeriesAxis::SingleVolume)?;
    Ok(AnalyzeStoredSeries {
        series,
        description: header.description,
    })
}

/// Writes a validated Analyze `.hdr`/`.img` pair.
///
/// The `.img` payload is flushed before the `.hdr` is published, so the header
/// remains the commit marker: a failure between the two writes leaves an inert
/// payload rather than a header describing data that was never written.
///
/// # Errors
///
/// Returns the scoped capability report or target rejection before either
/// destination exists, or an I/O or sample-encoding failure while writing.
pub fn write_analyze_stored<P: AsRef<Path>>(
    path: P,
    document: &AnalyzeStoredSeries,
) -> Result<(), AnalyzeStoredSeriesError> {
    let path = path.as_ref();
    let header = document.validate_for_write()?;
    let volume = document
        .series
        .volumes()
        .first()
        .expect("invariant: the preflight accepted exactly one volume");

    let img_file =
        File::create(path.with_extension("img")).map_err(AnalyzeStoredSeriesError::Io)?;
    let mut payload = BufWriter::with_capacity(8 * 1024, img_file);
    volume
        .samples()
        .write_to(&mut payload, ByteOrder::LeastSignificantByteFirst)
        .map_err(AnalyzeStoredSeriesError::SampleEncoding)?;
    payload.flush().map_err(AnalyzeStoredSeriesError::Io)?;

    // Publish the header only after the complete payload was written.
    std::fs::write(path.with_extension("hdr"), header).map_err(AnalyzeStoredSeriesError::Io)?;

    tracing::debug!(shape = ?volume.shape(), "write_analyze_stored: complete");

    Ok(())
}

fn coordinate_map_name(map: &CoordinateMap) -> &'static str {
    match map {
        CoordinateMap::Cartesian => "cartesian",
        CoordinateMap::CurvilinearArray(_) => "curvilinear array",
        CoordinateMap::PhasedArray3D(_) => "phased array",
        CoordinateMap::SliceSeries(_) => "slice series",
    }
}

/// Reduces a stored calibration to Analyze's single multiplicative scale.
fn calibration_scale(
    calibration: &IntensityCalibration,
    volume_index: usize,
) -> Result<f32, AnalyzeStoredSeriesRejection> {
    match calibration {
        IntensityCalibration::Identity => Ok(1.0),
        IntensityCalibration::Linear(linear) => {
            if linear.intercept() != 0.0 {
                return Err(AnalyzeStoredSeriesRejection::NonZeroInterceptCalibration {
                    volume_index,
                    intercept: linear.intercept(),
                });
            }
            if linear.slope() == 0.0 {
                return Err(AnalyzeStoredSeriesRejection::ZeroSlopeCalibration { volume_index });
            }
            Ok(linear.slope() as f32)
        }
        IntensityCalibration::PerFrameLinear(_) => {
            Err(AnalyzeStoredSeriesRejection::UnsupportedCalibration {
                volume_index,
                reason: "per-frame linear",
            })
        }
        IntensityCalibration::ModalityLookup(_) => {
            Err(AnalyzeStoredSeriesRejection::UnsupportedCalibration {
                volume_index,
                reason: "modality lookup table",
            })
        }
    }
}

/// A stored-series value that Analyze 7.5 cannot encode.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AnalyzeStoredSeriesRejection {
    /// The series axis has no single-volume Analyze representation.
    #[error("Analyze 7.5 stores exactly one 3-D volume; this series axis has no representation")]
    UnsupportedSeriesAxis,
    /// The codec mapping has no Analyze `datatype` code for the representation.
    #[error("volume {volume_index} stores {sample_type:?} samples, which has no Analyze 7.5 datatype code")]
    UnsupportedSampleType {
        /// The zero-based volume that differs.
        volume_index: usize,
        /// The stored scalar representation Analyze cannot encode.
        sample_type: SampleType,
    },
    /// Analyze stores a rectangular Cartesian grid only.
    #[error("volume {volume_index} uses {reason} coordinates, which Analyze 7.5 cannot encode")]
    UnsupportedCoordinateMap {
        /// The zero-based volume holding the map.
        volume_index: usize,
        /// Human-readable name of the rejected coordinate map.
        reason: &'static str,
    },
    /// Analyze has no direction field, so only an identity direction round-trips.
    #[error("volume {volume_index} has a non-identity direction, which Analyze 7.5 cannot encode")]
    NonIdentityDirection {
        /// The zero-based volume holding the direction.
        volume_index: usize,
    },
    /// Analyze's single scale factor has no additive term.
    #[error("volume {volume_index} has calibration intercept {intercept}, which Analyze 7.5 cannot encode")]
    NonZeroInterceptCalibration {
        /// The zero-based volume holding the calibration.
        volume_index: usize,
        /// The additive coefficient Analyze cannot represent.
        intercept: f64,
    },
    /// A zero scale is the header's no-scaling sentinel, so it cannot round-trip.
    #[error(
        "volume {volume_index} has a zero-slope calibration, which Analyze 7.5 reads as no scaling"
    )]
    ZeroSlopeCalibration {
        /// The zero-based volume holding the calibration.
        volume_index: usize,
    },
    /// The calibration category has no Analyze scalar scale equivalent.
    #[error("volume {volume_index} uses {reason} calibration, which Analyze 7.5 cannot encode")]
    UnsupportedCalibration {
        /// The zero-based volume holding the calibration.
        volume_index: usize,
        /// Human-readable name of the rejected calibration category.
        reason: &'static str,
    },
    /// A value needed for the header exceeds its field width.
    #[error("Analyze header cannot represent the stored series: {0}")]
    HeaderEncoding(#[source] anyhow::Error),
}

impl ConversionRejection for AnalyzeStoredSeriesRejection {
    fn location(&self) -> ConversionLocation {
        match self {
            Self::UnsupportedSampleType { volume_index, .. }
            | Self::UnsupportedCoordinateMap { volume_index, .. }
            | Self::NonIdentityDirection { volume_index }
            | Self::NonZeroInterceptCalibration { volume_index, .. }
            | Self::ZeroSlopeCalibration { volume_index }
            | Self::UnsupportedCalibration { volume_index, .. } => ConversionLocation::Volume {
                volume_index: *volume_index,
            },
            Self::UnsupportedSeriesAxis | Self::HeaderEncoding(_) => ConversionLocation::Series,
        }
    }
}

/// Failure to build or write an Analyze stored-sample pair.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AnalyzeStoredSeriesError {
    /// Preflight found a declared loss or a target-specific rejection.
    #[error("Analyze conversion preflight failed: {0}")]
    Preparation(#[source] ConversionPrepareError<AnalyzeStoredSeriesRejection>),
    /// A sample codec failed while writing exact stored values.
    #[error("cannot encode stored Analyze samples: {0}")]
    SampleEncoding(#[source] SampleError),
    /// A destination could not be created or written.
    #[error("cannot write Analyze pair: {0}")]
    Io(#[source] std::io::Error),
}

/// Failure to read an Analyze `.hdr`/`.img` pair as exact stored samples.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum AnalyzeStoredReadError {
    /// The header could not be parsed or validated.
    #[error("cannot read the Analyze header: {0}")]
    Header(#[source] anyhow::Error),
    /// The `.img` payload could not be opened or did not match the header.
    #[error("cannot read the Analyze payload: {0}")]
    Payload(#[source] anyhow::Error),
    /// The payload bytes are not a valid sequence of the declared samples.
    #[error("cannot decode stored Analyze samples: {0}")]
    Samples(#[source] SampleError),
    /// The decoded samples do not form a valid stored volume.
    #[error(transparent)]
    Volume(#[from] VolumeError),
    /// The volume does not form a valid stored series.
    #[error(transparent)]
    Series(#[from] StoredSeriesError),
}

#[cfg(test)]
mod tests;
