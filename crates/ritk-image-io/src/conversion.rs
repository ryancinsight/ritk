//! Capability checks for conversions between typed stored-image formats.

use ritk_codecs::SampleType;
use ritk_spatial::CoordinateMap;

use crate::{IntensityCalibration, SeriesAxis, StoredSeries};

/// A semantic capability required to preserve a stored image.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConversionFeature {
    /// The target can encode this exact stored sample representation.
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
    /// The target can preserve one linear calibration.
    LinearCalibration,
    /// The target can preserve per-frame linear calibration.
    PerFrameLinearCalibration,
    /// The target can preserve a nonlinear modality lookup table.
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

/// A format-specific metadata field that conversion cannot retain.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum FormatMetadataLoss {
    /// The target format has no representation for this field.
    UnsupportedByTarget(Box<str>),
    /// The adapter does not know the field's semantics.
    UnknownSemantics(Box<str>),
}

/// A format's statically declared stored-image capabilities.
pub trait ConversionTarget {
    /// Stable format identifier used in conversion diagnostics.
    const FORMAT: &'static str;
    /// Every semantic capability this target can encode without loss.
    const FEATURES: &'static [ConversionFeature];
}

/// A typed reason why a requested conversion cannot preserve its semantics.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ConversionLoss {
    /// A volume or series capability is unsupported by the target.
    UnsupportedFeature {
        /// The affected volume, or `None` for series-level semantics.
        volume_index: Option<usize>,
        /// The semantic capability that cannot be retained.
        feature: ConversionFeature,
    },
    /// A format-specific field cannot be represented by the target.
    FormatMetadata(FormatMetadataLoss),
}

/// The capability comparison for a requested conversion.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct ConversionPlan {
    /// Source format identifier.
    pub source_format: &'static str,
    /// Target format identifier.
    pub target_format: &'static str,
    /// Capabilities declared by the target format.
    pub target_features: &'static [ConversionFeature],
    /// Unsupported semantics and fields the adapter cannot map.
    pub losses: Box<[ConversionLoss]>,
}

/// A source series whose semantics passed the target capability check.
#[derive(Debug)]
pub struct PreparedConversion<'a> {
    series: &'a StoredSeries,
    plan: ConversionPlan,
}

impl<'a> PreparedConversion<'a> {
    /// Returns the original stored series without copying or changing samples.
    #[must_use]
    pub const fn series(&self) -> &'a StoredSeries {
        self.series
    }

    /// Returns the successful capability comparison.
    #[must_use]
    pub const fn plan(&self) -> &ConversionPlan {
        &self.plan
    }
}

/// Checks a stored series against one format's declared capabilities.
///
/// The function is pure and has no destination handle. Adapters supply only
/// metadata fields they cannot map; an empty iterator means none are lost.
///
/// # Examples
///
/// ```
/// # use ritk_codecs::{SampleBuffer, SampleType};
/// # use ritk_image::ImageMetadata;
/// # use ritk_image_io::{preflight_conversion, ConversionFeature, ConversionTarget, IntensityCalibration, SeriesAxis, StoredSeries, StoredVolume};
/// # use ritk_spatial::CoordinateMap;
/// # struct Nrrd;
/// # impl ConversionTarget for Nrrd {
/// #     const FORMAT: &'static str = "nrrd";
/// #     const FEATURES: &'static [ConversionFeature] = &[ConversionFeature::SampleType(SampleType::U16), ConversionFeature::PhysicalGeometry, ConversionFeature::CartesianCoordinates, ConversionFeature::IdentityCalibration, ConversionFeature::SingleVolumeAxis];
/// # }
/// # let volume = StoredVolume::new([1, 1, 1], SampleBuffer::from_samples(vec![7_u16]), ImageMetadata::default(), CoordinateMap::Cartesian, IntensityCalibration::Identity).expect("valid one-sample volume");
/// # let series = StoredSeries::new(vec![volume], SeriesAxis::SingleVolume).expect("single-volume series");
/// let prepared = preflight_conversion::<Nrrd>("dicom", &series, std::iter::empty())
///     .expect("NRRD supports every source semantic");
/// assert_eq!(prepared.plan().target_format, "nrrd");
/// ```
///
/// # Errors
///
/// Returns the complete plan when a sample, spatial, calibration, acquisition,
/// or declared metadata semantic is unsupported.
pub fn preflight_conversion<'a, T: ConversionTarget>(
    source_format: &'static str,
    series: &'a StoredSeries,
    metadata_losses: impl IntoIterator<Item = FormatMetadataLoss>,
) -> Result<PreparedConversion<'a>, ConversionPlan> {
    let mut losses = Vec::new();
    for (volume_index, volume) in series.volumes().iter().enumerate() {
        for feature in [
            ConversionFeature::SampleType(volume.samples().sample_type()),
            ConversionFeature::PhysicalGeometry,
            coordinate_feature(volume.coordinate_map()),
            calibration_feature(volume.calibration()),
        ] {
            if !T::FEATURES.contains(&feature) {
                losses.push(ConversionLoss::UnsupportedFeature {
                    volume_index: Some(volume_index),
                    feature,
                });
            }
        }
    }
    let axis = axis_feature(series.axis());
    if !T::FEATURES.contains(&axis) {
        losses.push(ConversionLoss::UnsupportedFeature {
            volume_index: None,
            feature: axis,
        });
    }
    losses.extend(
        metadata_losses
            .into_iter()
            .map(ConversionLoss::FormatMetadata),
    );
    let plan = ConversionPlan {
        source_format,
        target_format: T::FORMAT,
        target_features: T::FEATURES,
        losses: losses.into_boxed_slice(),
    };
    if plan.losses.is_empty() {
        Ok(PreparedConversion { series, plan })
    } else {
        Err(plan)
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
    if calibration.is_identity() {
        return ConversionFeature::IdentityCalibration;
    }
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
