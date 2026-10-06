use crate::document::{NiftiDocument, NiftiVersion};
use crate::stored_series::NiftiStoredSeriesError;
use ritk_codecs::SampleType;
use ritk_diffusion_scheme::{GradientFrame, GradientScheme};
use ritk_image_io::{
    ConversionFeature, ConversionLocation, ConversionLoss, IntensityCalibration, SeriesAxis,
    StoredSeries,
};
use ritk_spatial::Vector;

use super::{metadata, volume};

fn diffusion_axis_is_reported_before_document_creation(volume_count: usize) {
    let volumes = (0..volume_count)
        .map(|_| {
            volume(
                [1, 1, 1],
                SampleType::U8,
                &[7],
                metadata([0.0; 3], [1.0; 3]),
                IntensityCalibration::Identity,
            )
        })
        .collect();
    let gradients = (0..volume_count)
        .map(|index| {
            if index == 0 {
                (0.0, Vector::new([0.0; 3]))
            } else {
                (1000.0, Vector::new([1.0, 0.0, 0.0]))
            }
        })
        .collect();
    let scheme = GradientScheme::from_seconds_per_square_millimeter(gradients, GradientFrame::Lps)
        .expect("gradient scheme matches the volume count");
    let series = StoredSeries::new(volumes, SeriesAxis::Diffusion(scheme))
        .expect("diffusion-axis fixture is valid");

    let Err(NiftiStoredSeriesError::Capabilities(report)) =
        NiftiDocument::from_stored_series(&series, NiftiVersion::Two)
    else {
        panic!("unsupported diffusion axis must be reported before document creation");
    };
    assert_eq!(
        report.losses.as_ref(),
        &[ConversionLoss::UnsupportedFeature {
            location: ConversionLocation::Series,
            feature: ConversionFeature::DiffusionAxis,
        }]
    );
}

#[test]
fn diffusion_axis_is_reported_for_single_and_multiple_volumes() {
    for volume_count in [1, 2] {
        diffusion_axis_is_reported_before_document_creation(volume_count);
    }
}
