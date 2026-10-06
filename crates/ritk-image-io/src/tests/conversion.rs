use ritk_codecs::{SampleBuffer, SampleType};
use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
use ritk_image::ImageMetadata;
use ritk_spatial::{CoordinateMap, PhasedArray3D, Vector};

use crate::{
    prepare_conversion, report_conversion_capabilities, ConversionAdapter,
    ConversionCapabilityReport, ConversionFeature, ConversionLocation, ConversionLoss,
    ConversionPrepareError, ConversionRejection, ConversionTarget, FormatMetadataLoss,
    IntensityCalibration, LinearCalibration, SeriesAxis, StoredSeries, StoredVolume,
};
use thiserror::Error;

struct FullTarget;

impl ConversionTarget for FullTarget {
    const FORMAT: &'static str = "full";
    const FEATURES: &'static [ConversionFeature] = &[
        ConversionFeature::SampleType(SampleType::U16),
        ConversionFeature::PhysicalGeometry,
        ConversionFeature::CartesianCoordinates,
        ConversionFeature::IdentityCalibration,
        ConversionFeature::LinearCalibration,
        ConversionFeature::ListAxis,
    ];
}

struct NarrowTarget;

impl ConversionTarget for NarrowTarget {
    const FORMAT: &'static str = "narrow";
    const FEATURES: &'static [ConversionFeature] =
        &[ConversionFeature::SampleType(SampleType::U16)];
}

fn volume(samples: Vec<u16>, calibration: IntensityCalibration) -> StoredVolume {
    StoredVolume::new(
        [1, 1, samples.len()],
        SampleBuffer::from_samples(samples),
        ImageMetadata::default(),
        CoordinateMap::Cartesian,
        calibration,
    )
    .expect("test volume satisfies the stored-series contract")
}

fn unsupported(location: ConversionLocation, feature: ConversionFeature) -> ConversionLoss {
    ConversionLoss::UnsupportedFeature { location, feature }
}

fn volume_loss(volume_index: usize, feature: ConversionFeature) -> ConversionLoss {
    unsupported(ConversionLocation::Volume { volume_index }, feature)
}

#[test]
fn report_contains_format_ids_categories_and_scoped_metadata() {
    let linear = LinearCalibration::new(2.0, -1024.0).expect("finite transform");
    let series = StoredSeries::new(
        vec![
            volume(vec![17, 29], IntensityCalibration::Linear(linear)),
            volume(vec![31], IntensityCalibration::Identity),
        ],
        SeriesAxis::List,
    )
    .expect("nonempty list series");
    let series_loss = FormatMetadataLoss::UnsupportedByTarget {
        location: ConversionLocation::Series,
        field: Box::from("scanner_private_tag"),
    };
    let volume_loss = FormatMetadataLoss::UnsupportedByTarget {
        location: ConversionLocation::Volume { volume_index: 0 },
        field: Box::from("vendor_note"),
    };
    let frame_loss = FormatMetadataLoss::UnknownSemantics {
        location: ConversionLocation::Frame {
            volume_index: 1,
            frame_index: 0,
        },
        field: Box::from("private_acquisition_tag"),
    };

    let report = report_conversion_capabilities::<FullTarget>(
        "dicom",
        &series,
        [series_loss.clone(), volume_loss.clone(), frame_loss.clone()],
    );

    assert_eq!(
        report,
        ConversionCapabilityReport {
            source_format: "dicom",
            target_format: "full",
            target_features: FullTarget::FEATURES,
            losses: Box::from([
                ConversionLoss::FormatMetadata(series_loss),
                ConversionLoss::FormatMetadata(volume_loss),
                ConversionLoss::FormatMetadata(frame_loss),
            ]),
        }
    );
}

#[test]
fn unsupported_feature_report_identifies_each_volume_and_series_axis() {
    let first = StoredVolume::new(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![-15_i16]),
        ImageMetadata::default(),
        CoordinateMap::Cartesian,
        IntensityCalibration::Identity,
    )
    .expect("one-frame volume");
    let phased = PhasedArray3D::try_new(1.0, 0.0, 0.1, 0.1, 0.0, 0.0)
        .expect("finite three-dimensional steering geometry");
    let second = StoredVolume::new(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![23_u16]),
        ImageMetadata::default(),
        CoordinateMap::PhasedArray3D(phased),
        IntensityCalibration::Identity,
    )
    .expect("phased-array volume");
    let gradients = GradientScheme::from_seconds_per_square_millimeter(
        vec![
            (0.0, Vector::new([0.0; 3])),
            (1000.0, Vector::new([1.0, 0.0, 0.0])),
        ],
        GradientFrame::Lps,
    )
    .expect("valid two-volume gradient scheme");
    let series = StoredSeries::new(vec![first, second], SeriesAxis::Diffusion(gradients))
        .expect("two-volume diffusion series");

    let report = report_conversion_capabilities::<NarrowTarget>("dicom", &series, []);

    assert_eq!(
        report.losses.as_ref(),
        &[
            volume_loss(0, ConversionFeature::SampleType(SampleType::I16)),
            volume_loss(0, ConversionFeature::PhysicalGeometry),
            volume_loss(0, ConversionFeature::CartesianCoordinates),
            volume_loss(0, ConversionFeature::IdentityCalibration),
            volume_loss(1, ConversionFeature::PhysicalGeometry),
            volume_loss(1, ConversionFeature::PhasedArray3DCoordinates),
            volume_loss(1, ConversionFeature::IdentityCalibration),
            unsupported(ConversionLocation::Series, ConversionFeature::DiffusionAxis,),
        ]
    );
}

#[derive(Default)]
struct UniformTarget {
    preparation_calls: std::cell::Cell<usize>,
}

impl ConversionTarget for UniformTarget {
    const FORMAT: &'static str = "uniform";
    const FEATURES: &'static [ConversionFeature] = &[
        ConversionFeature::SampleType(SampleType::U16),
        ConversionFeature::PhysicalGeometry,
        ConversionFeature::CartesianCoordinates,
        ConversionFeature::IdentityCalibration,
        ConversionFeature::SingleVolumeAxis,
        ConversionFeature::ListAxis,
    ];
}

struct UniformPlan {
    shape: [usize; 3],
}

#[derive(Debug, Error, Eq, PartialEq)]
enum TargetRejection {
    #[error("volume shape {actual:?} differs from {expected:?}")]
    ShapeMismatch {
        location: ConversionLocation,
        expected: [usize; 3],
        actual: [usize; 3],
    },
    #[error("stored series contains no volume")]
    EmptySeries,
}

impl ConversionRejection for TargetRejection {
    fn location(&self) -> ConversionLocation {
        match self {
            Self::ShapeMismatch { location, .. } => *location,
            Self::EmptySeries => ConversionLocation::Series,
        }
    }
}

impl ConversionAdapter for UniformTarget {
    type Plan = UniformPlan;
    type Rejection = TargetRejection;

    fn prepare(&self, series: &StoredSeries) -> Result<Self::Plan, Self::Rejection> {
        self.preparation_calls.set(self.preparation_calls.get() + 1);
        let Some((first, rest)) = series.volumes().split_first() else {
            return Err(TargetRejection::EmptySeries);
        };
        let shape = first.shape();
        for (offset, volume) in rest.iter().enumerate() {
            if volume.shape() != shape {
                return Err(TargetRejection::ShapeMismatch {
                    location: ConversionLocation::Volume {
                        volume_index: offset + 1,
                    },
                    expected: shape,
                    actual: volume.shape(),
                });
            }
        }
        Ok(UniformPlan { shape })
    }
}

#[test]
fn prepared_conversion_carries_the_target_plan_and_exact_source() {
    let target = UniformTarget::default();
    let series = StoredSeries::new(
        vec![
            volume(vec![17, 29], IntensityCalibration::Identity),
            volume(vec![31, 47], IntensityCalibration::Identity),
        ],
        SeriesAxis::List,
    )
    .expect("two-volume list series");

    let prepared = prepare_conversion(&target, "dicom", &series, [])
        .expect("target accepts matching volume shapes");

    assert_eq!(target.preparation_calls.get(), 1);
    assert!(std::ptr::eq(prepared.target(), &target));
    assert!(std::ptr::eq(prepared.series(), &series));
    assert_eq!(prepared.plan().shape, [1, 1, 2]);
    assert_eq!(prepared.capabilities().source_format, "dicom");
    assert_eq!(prepared.capabilities().target_format, "uniform");
    assert!(prepared.capabilities().losses.is_empty());
}

#[test]
fn prepared_conversion_reports_cross_volume_mismatch_at_the_later_volume() {
    let target = UniformTarget::default();
    let series = StoredSeries::new(
        vec![
            volume(vec![17, 29], IntensityCalibration::Identity),
            volume(vec![31], IntensityCalibration::Identity),
        ],
        SeriesAxis::List,
    )
    .expect("stored series permits per-volume shapes");

    let Err(ConversionPrepareError::Target {
        target_format,
        location,
        source: TargetRejection::ShapeMismatch {
            expected, actual, ..
        },
    }) = prepare_conversion(&target, "dicom", &series, [])
    else {
        panic!("target must reject the non-uniform series at its mismatching volume");
    };

    assert_eq!(target.preparation_calls.get(), 1);
    assert_eq!(target_format, "uniform");
    assert_eq!(location, ConversionLocation::Volume { volume_index: 1 });
    assert_eq!(expected, [1, 1, 2]);
    assert_eq!(actual, [1, 1, 1]);
}

#[test]
fn reported_information_loss_prevents_target_preparation() {
    let target = UniformTarget::default();
    let series = StoredSeries::new(
        vec![volume(vec![17], IntensityCalibration::Identity)],
        SeriesAxis::SingleVolume,
    )
    .expect("single-volume series");
    let loss = FormatMetadataLoss::UnknownSemantics {
        location: ConversionLocation::Frame {
            volume_index: 0,
            frame_index: 0,
        },
        field: Box::from("private_geometry_tag"),
    };

    let Err(ConversionPrepareError::Capabilities(report)) =
        prepare_conversion(&target, "dicom", &series, [loss.clone()])
    else {
        panic!("reported metadata loss must prevent a prepared conversion");
    };

    assert_eq!(target.preparation_calls.get(), 0);
    assert_eq!(
        report.losses.as_ref(),
        &[ConversionLoss::FormatMetadata(loss)]
    );
}
