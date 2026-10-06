//! Reports target-format feature categories for stored-image conversion.

use ritk_codecs::SampleType;
use ritk_spatial::CoordinateMap;

use crate::{IntensityCalibration, SeriesAxis, StoredSeries};

/// A semantic capability category used by a stored-image format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConversionFeature {
    /// The target can encode this stored sample representation.
    SampleType(SampleType),
    /// The target can preserve physical origin, spacing, and direction.
    PhysicalGeometry,
    /// The target can preserve Cartesian coordinates.
    CartesianCoordinates,
    /// The target can preserve curvilinear array coordinates.
    CurvilinearArrayCoordinates,
    /// The target can preserve three-dimensional phased-array coordinates.
    PhasedArray3DCoordinates,
    /// The target can preserve a wobbler or freehand slice-series map.
    SliceSeriesCoordinates,
    /// The target can preserve identity calibration.
    IdentityCalibration,
    /// The target can preserve a linear calibration category.
    LinearCalibration,
    /// The target can preserve per-frame linear calibration.
    PerFrameLinearCalibration,
    /// The target can preserve a nonlinear modality lookup-table category.
    ModalityLookupCalibration,
    /// The target can preserve a single-volume axis.
    SingleVolumeAxis,
    /// The target can preserve an ordered list axis.
    ListAxis,
    /// The target can preserve an axis with unspecified meaning.
    UnspecifiedAxis,
    /// The target can preserve a diffusion axis.
    DiffusionAxis,
}

/// The scope of a semantic loss reported during conversion inspection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConversionLocation {
    /// A property belonging to the complete series.
    Series,
    /// A zero-based volume in the series.
    Volume {
        /// The zero-based volume index.
        volume_index: usize,
    },
    /// A zero-based frame within a zero-based volume.
    Frame {
        /// The zero-based volume index.
        volume_index: usize,
        /// The zero-based frame index.
        frame_index: usize,
    },
}

/// A format-specific metadata field the target adapter cannot retain.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FormatMetadataLoss {
    /// The target has no representation for the field.
    UnsupportedByTarget {
        /// The affected series, volume, or frame.
        location: ConversionLocation,
        /// The format-specific field name.
        field: Box<str>,
    },
    /// The adapter does not know the field's semantics.
    UnknownSemantics {
        /// The affected series, volume, or frame.
        location: ConversionLocation,
        /// The format-specific field name.
        field: Box<str>,
    },
}

/// A target format's declared feature categories.
pub trait ConversionTarget {
    /// Stable format identifier used in reports.
    const FORMAT: &'static str;
    /// Semantic categories the format can encode.
    const FEATURES: &'static [ConversionFeature];
}

/// A feature category or metadata field not reported as supported by a target.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConversionLoss {
    /// The target does not declare support for a volume or series category.
    UnsupportedFeature {
        /// The affected volume or series.
        location: ConversionLocation,
        /// The missing semantic category.
        feature: ConversionFeature,
    },
    /// A source-format metadata field is not retained by the adapter.
    FormatMetadata(FormatMetadataLoss),
}

/// A non-authoritative report of target feature categories and metadata losses.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ConversionCapabilityReport {
    /// Source format identifier supplied by the caller.
    pub source_format: &'static str,
    /// Target format identifier declared by its adapter.
    pub target_format: &'static str,
    /// Target feature categories declared by its adapter.
    pub target_features: &'static [ConversionFeature],
    /// Unsupported categories and adapter-reported metadata losses.
    pub losses: Box<[ConversionLoss]>,
}

/// Reports feature-category differences for a stored series and target format.
///
/// The report compares each volume's sample, geometry, coordinate-map, and
/// calibration categories, the series axis category, and scoped metadata losses
/// supplied by the format adapter. It does not compare values across volumes,
/// check adapter-specific value limits, or authorize output. A later preparation
/// step must establish those properties before writing.
///
/// # Examples
///
/// ```
/// # use ritk_codecs::{SampleBuffer, SampleType};
/// # use ritk_image::ImageMetadata;
/// # use ritk_image_io::{report_conversion_capabilities, ConversionFeature, ConversionTarget, IntensityCalibration, SeriesAxis, StoredSeries, StoredVolume};
/// # use ritk_spatial::CoordinateMap;
/// # struct Nrrd;
/// # impl ConversionTarget for Nrrd {
/// #     const FORMAT: &'static str = "nrrd";
/// #     const FEATURES: &'static [ConversionFeature] = &[ConversionFeature::SampleType(SampleType::U16)];
/// # }
/// # let volume = StoredVolume::new([1, 1, 1], SampleBuffer::from_samples(vec![7_u16]), ImageMetadata::default(), CoordinateMap::Cartesian, IntensityCalibration::Identity).expect("valid one-sample volume");
/// # let series = StoredSeries::new(vec![volume], SeriesAxis::SingleVolume).expect("single-volume series");
/// let report = report_conversion_capabilities::<Nrrd>("dicom", &series, []);
/// assert_eq!(report.source_format, "dicom");
/// assert_eq!(report.target_format, "nrrd");
/// assert_eq!(report.target_features, Nrrd::FEATURES);
/// ```
pub fn report_conversion_capabilities<T: ConversionTarget>(
    source_format: &'static str,
    series: &StoredSeries,
    metadata_losses: impl IntoIterator<Item = FormatMetadataLoss>,
) -> ConversionCapabilityReport {
    let mut losses = Vec::new();
    for (volume_index, volume) in series.volumes().iter().enumerate() {
        let location = ConversionLocation::Volume { volume_index };
        for feature in [
            ConversionFeature::SampleType(volume.samples().sample_type()),
            ConversionFeature::PhysicalGeometry,
            coordinate_feature(volume.coordinate_map()),
            calibration_feature(volume.calibration()),
        ] {
            if !T::FEATURES.contains(&feature) {
                losses.push(ConversionLoss::UnsupportedFeature { location, feature });
            }
        }
    }
    let axis = axis_feature(series.axis());
    if !T::FEATURES.contains(&axis) {
        losses.push(ConversionLoss::UnsupportedFeature {
            location: ConversionLocation::Series,
            feature: axis,
        });
    }
    losses.extend(
        metadata_losses
            .into_iter()
            .map(ConversionLoss::FormatMetadata),
    );
    ConversionCapabilityReport {
        source_format,
        target_format: T::FORMAT,
        target_features: T::FEATURES,
        losses: losses.into_boxed_slice(),
    }
}

fn coordinate_feature(map: &CoordinateMap) -> ConversionFeature {
    match map {
        CoordinateMap::Cartesian => ConversionFeature::CartesianCoordinates,
        CoordinateMap::CurvilinearArray(_) => ConversionFeature::CurvilinearArrayCoordinates,
        CoordinateMap::PhasedArray3D(_) => ConversionFeature::PhasedArray3DCoordinates,
        CoordinateMap::SliceSeries(_) => ConversionFeature::SliceSeriesCoordinates,
    }
}

fn calibration_feature(calibration: &IntensityCalibration) -> ConversionFeature {
    match calibration {
        IntensityCalibration::Identity => ConversionFeature::IdentityCalibration,
        IntensityCalibration::Linear(_) => ConversionFeature::LinearCalibration,
        IntensityCalibration::PerFrameLinear(_) => ConversionFeature::PerFrameLinearCalibration,
        IntensityCalibration::ModalityLookup(_) => ConversionFeature::ModalityLookupCalibration,
    }
}

fn axis_feature(axis: &SeriesAxis) -> ConversionFeature {
    match axis {
        SeriesAxis::SingleVolume => ConversionFeature::SingleVolumeAxis,
        SeriesAxis::List => ConversionFeature::ListAxis,
        SeriesAxis::Unspecified => ConversionFeature::UnspecifiedAxis,
        SeriesAxis::Diffusion(_) => ConversionFeature::DiffusionAxis,
    }
}
