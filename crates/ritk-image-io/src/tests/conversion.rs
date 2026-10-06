use ritk_codecs::{SampleBuffer, SampleType};
use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
use ritk_image::ImageMetadata;
use ritk_spatial::{CoordinateMap, PhasedArray3D, Vector};

use crate::{
    report_conversion_capabilities, ConversionCapabilityReport, ConversionFeature,
    ConversionLocation, ConversionLoss, ConversionTarget, FormatMetadataLoss, IntensityCalibration,
    LinearCalibration, SeriesAxis, StoredSeries, StoredVolume,
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
