use ritk_codecs::{SampleBuffer, SampleType};
use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
use ritk_image::ImageMetadata;
use ritk_spatial::{CoordinateMap, PhasedArray3D, Vector};

use crate::conversion::{
    preflight_conversion, ConversionFeature, ConversionLoss, ConversionTarget,
};
use crate::{
    FormatMetadataLoss, IntensityCalibration, LinearCalibration, SeriesAxis, StoredSeries,
    StoredVolume,
};

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

fn unsupported(volume_index: Option<usize>, feature: ConversionFeature) -> ConversionLoss {
    ConversionLoss::UnsupportedFeature {
        volume_index,
        feature,
    }
}

#[test]
fn successful_plan_borrows_samples_and_records_capabilities() {
    let identity = LinearCalibration::new(1.0, 0.0).expect("finite identity transform");
    let linear = LinearCalibration::new(2.0, -1024.0).expect("finite transform");
    let series = StoredSeries::new(
        vec![
            volume(vec![17, 29], IntensityCalibration::Linear(identity)),
            volume(vec![31], IntensityCalibration::Linear(linear)),
        ],
        SeriesAxis::List,
    )
    .expect("nonempty list series");

    let prepared = preflight_conversion::<FullTarget>("nrrd", &series, std::iter::empty())
        .expect("every observed feature is supported");

    assert!(std::ptr::eq(prepared.series(), &series));
    assert_eq!(prepared.plan().source_format, "nrrd");
    assert_eq!(prepared.plan().target_format, "full");
    assert_eq!(prepared.plan().target_features, FullTarget::FEATURES);
}

#[test]
fn rejected_plan_identifies_each_unsupported_semantic() {
    let linear = LinearCalibration::new(2.0, -1024.0).expect("finite transform");
    let first = StoredVolume::new(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![-15_i16]),
        ImageMetadata::default(),
        CoordinateMap::Cartesian,
        IntensityCalibration::PerFrameLinear(Box::from([linear])),
    )
    .expect("one-frame volume");
    let phased = PhasedArray3D::try_new(1.0, 0.0, 0.1, 0.1, 0.0, 0.0)
        .expect("finite three-dimensional steering geometry");
    let lookup =
        crate::ModalityLookupTable::new(0, Box::from([3, 7]), crate::LutOutputBits::Sixteen)
            .expect("valid modality table");
    let second = StoredVolume::new(
        [1, 1, 1],
        SampleBuffer::from_samples(vec![23_u16]),
        ImageMetadata::default(),
        CoordinateMap::PhasedArray3D(phased),
        IntensityCalibration::ModalityLookup(lookup),
    )
    .expect("phased-array volume");
    let series = StoredSeries::new(vec![first, second], SeriesAxis::Unspecified)
        .expect("two-volume series with an unspecified axis");
    let metadata_loss = FormatMetadataLoss::UnknownSemantics(Box::from("private_acquisition_tag"));

    let plan = preflight_conversion::<NarrowTarget>("dicom", &series, [metadata_loss.clone()])
        .expect_err("unsupported semantics cannot produce a prepared conversion");

    assert_eq!(
        plan.losses.as_ref(),
        &[
            unsupported(Some(0), ConversionFeature::SampleType(SampleType::I16)),
            unsupported(Some(0), ConversionFeature::PhysicalGeometry),
            unsupported(Some(0), ConversionFeature::CartesianCoordinates),
            unsupported(Some(0), ConversionFeature::PerFrameLinearCalibration),
            unsupported(Some(1), ConversionFeature::PhysicalGeometry),
            unsupported(Some(1), ConversionFeature::PhasedArray3DCoordinates),
            unsupported(Some(1), ConversionFeature::ModalityLookupCalibration),
            unsupported(None, ConversionFeature::UnspecifiedAxis),
            ConversionLoss::FormatMetadata(metadata_loss),
        ]
    );
}

#[test]
fn unsupported_diffusion_axis_is_reported_without_volume_losses() {
    let gradients = GradientScheme::from_seconds_per_square_millimeter(
        vec![(0.0, Vector::new([0.0; 3]))],
        GradientFrame::Lps,
    )
    .expect("valid baseline gradient");
    let series = StoredSeries::new(
        vec![volume(vec![13], IntensityCalibration::Identity)],
        SeriesAxis::Diffusion(gradients),
    )
    .expect("one-volume diffusion series");

    let plan = preflight_conversion::<FullTarget>("dicom", &series, [])
        .expect_err("the target does not support a diffusion axis");

    assert_eq!(
        plan.losses.as_ref(),
        &[unsupported(None, ConversionFeature::DiffusionAxis)]
    );
}
