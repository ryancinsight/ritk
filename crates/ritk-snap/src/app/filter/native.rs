//! Coeus-native Snap filter dispatch.
//!
//! The viewer owns host-side [`crate::LoadedVolume`] buffers, while native RITK
//! filters operate on `ritk_image::native::Image`. This module is the sole
//! explicit application boundary between those representations. It is entered
//! only for fully native filter variants; all other variants remain in the
//! legacy graph until their provider operation family is available.

use anyhow::{Context, Result};
use coeus_core::SequentialBackend;
use ritk_filter::{
    distance::euclidean::native::{distance_transform, signed_distance_transform},
    morphology::native::{
        binary_closing, binary_dilate, binary_erode, binary_fill_holes, binary_opening,
    },
    AbsImageFilter, AcosImageFilter, AsinImageFilter, AtanImageFilter, BinaryContourImageFilter,
    BoundedReciprocalImageFilter, ClaheFilter, ClampImageFilter, ConstantPadImageFilter,
    CosImageFilter, ExpImageFilter, FlipImageFilter, GradientAnisotropicDiffusionFilter,
    GradientDiffusionConfig, GrayscaleClosingFilter, GrayscaleDilation, GrayscaleErosion,
    GrayscaleFillholeFilter, GrayscaleGeodesicDilationFilter, GrayscaleGeodesicErosionFilter,
    GrayscaleMorphologicalGradientFilter, GrayscaleOpeningFilter, HistogramEqualizationFilter,
    InvertIntensityFilter, LabelContourImageFilter, LogImageFilter, MaskImageFilter,
    MeanImageFilter, MedianFilter, MirrorPadImageFilter, NormalizeImageFilter,
    PermuteAxesImageFilter, RegionOfInterestImageFilter, RescaleIntensityFilter,
    ShiftScaleImageFilter, SinImageFilter, SqrtImageFilter, SquareImageFilter, TanImageFilter,
    TileMeanShrinkFilter, VotingBinaryImageFilter, WrapPadImageFilter, ZeroCrossingImageFilter,
};
use ritk_image::native::Image;
use ritk_segmentation::{
    labeling::Connectivity as SegmentationConnectivity,
    native::{
        binary_threshold, confidence_connected, connected_components, connected_threshold,
        multi_otsu, neighborhood_connected, relabel_components,
    },
};
use ritk_spatial::{Direction, Point, Spacing};
use std::ops::Deref;

use crate::{FilterKind, LoadedVolume};

/// Host representation of a Coeus-native filter result.
///
/// The viewer owns the displayed geometry, so every native result carries the
/// image metadata required to replace it consistently with its pixel data.
#[derive(Debug)]
pub(super) struct NativeFilterOutput {
    pub(super) data: Vec<f32>,
    pub(super) shape: [usize; 3],
    pub(super) origin: [f64; 3],
    pub(super) spacing: [f64; 3],
    pub(super) direction: [f64; 9],
}

impl Deref for NativeFilterOutput {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl PartialEq<Vec<f32>> for NativeFilterOutput {
    fn eq(&self, other: &Vec<f32>) -> bool {
        self.data == *other
    }
}

/// Apply a filter when the full operation is available on the Coeus-native
/// image substrate.
///
/// `None` means this operation has not completed its native migration and must
/// remain on the legacy graph. `Some(Err(_))` reports an input or provider
/// contract failure without falling back to another implementation.
pub(super) fn apply_if_supported(
    volume: &LoadedVolume,
    filter: &FilterKind,
) -> Option<Result<NativeFilterOutput>> {
    if !matches!(
        filter,
        FilterKind::Abs
            | FilterKind::Square
            | FilterKind::Sqrt
            | FilterKind::Log
            | FilterKind::Exp
            | FilterKind::BinaryErode { .. }
            | FilterKind::BinaryDilate { .. }
            | FilterKind::BinaryClosing { .. }
            | FilterKind::BinaryOpening { .. }
            | FilterKind::BinaryFillhole { .. }
            | FilterKind::DistanceTransform { .. }
            | FilterKind::SignedDistanceTransform { .. }
            | FilterKind::ConnectedComponents { .. }
            | FilterKind::RelabelComponents { .. }
            | FilterKind::MultiOtsuThreshold { .. }
            | FilterKind::Median { .. }
            | FilterKind::HistEq { .. }
            | FilterKind::Clahe { .. }
            | FilterKind::GradientAnisotropicDiffusion { .. }
            | FilterKind::ConnectedThreshold { .. }
            | FilterKind::ConfidenceConnected { .. }
            | FilterKind::NeighborhoodConnected { .. }
            | FilterKind::BinaryThreshold { .. }
            | FilterKind::InvertIntensity { .. }
            | FilterKind::Clamp { .. }
            | FilterKind::ShiftScale { .. }
            | FilterKind::RescaleIntensity { .. }
            | FilterKind::NormalizeIntensity
            | FilterKind::FlipZ
            | FilterKind::FlipY
            | FilterKind::FlipX
            | FilterKind::RegionOfInterest { .. }
            | FilterKind::PermuteAxes { .. }
            | FilterKind::Shrink { .. }
            | FilterKind::ConstantPad { .. }
            | FilterKind::MirrorPad { .. }
            | FilterKind::WrapPad { .. }
            | FilterKind::MaskThreshold { .. }
            | FilterKind::Atan
            | FilterKind::Sin
            | FilterKind::Cos
            | FilterKind::Tan
            | FilterKind::Asin
            | FilterKind::Acos
            | FilterKind::BoundedReciprocal
            | FilterKind::Mean { .. }
            | FilterKind::GrayscaleErode { .. }
            | FilterKind::GrayscaleDilate { .. }
            | FilterKind::GrayscaleClosing { .. }
            | FilterKind::GrayscaleOpening { .. }
            | FilterKind::MorphologicalGradient { .. }
            | FilterKind::BinaryContour { .. }
            | FilterKind::LabelContour { .. }
            | FilterKind::VotingBinary { .. }
            | FilterKind::GrayscaleFillhole
            | FilterKind::GeodesicDilationSelf
            | FilterKind::GeodesicErosionSelf
            | FilterKind::ZeroCrossing { .. }
    ) {
        return None;
    }

    Some(apply_supported_filter(volume, filter))
}

fn apply_supported_filter(
    volume: &LoadedVolume,
    filter: &FilterKind,
) -> Result<NativeFilterOutput> {
    if volume.channels != 1 {
        anyhow::bail!(
            "native scalar filters require a scalar volume, received {} interleaved channels",
            volume.channels
        );
    }

    let backend = SequentialBackend;
    let image = Image::from_flat_on(
        (*volume.data).clone(),
        volume.shape,
        Point::new(volume.origin),
        Spacing::new(volume.spacing),
        Direction::from_rows([
            [
                volume.direction[0],
                volume.direction[1],
                volume.direction[2],
            ],
            [
                volume.direction[3],
                volume.direction[4],
                volume.direction[5],
            ],
            [
                volume.direction[6],
                volume.direction[7],
                volume.direction[8],
            ],
        ]),
        &backend,
    )
    .context("cannot construct Coeus-native image from loaded volume")?;

    let output = match filter {
        FilterKind::Abs => AbsImageFilter::new().apply_native(&image, &backend),
        FilterKind::Square => SquareImageFilter::new().apply_native(&image, &backend),
        FilterKind::Sqrt => SqrtImageFilter::new().apply_native(&image, &backend),
        FilterKind::Log => LogImageFilter::new().apply_native(&image, &backend),
        FilterKind::Exp => ExpImageFilter::new().apply_native(&image, &backend),
        FilterKind::InvertIntensity { maximum } => match maximum {
            Some(maximum) => {
                InvertIntensityFilter::with_maximum(*maximum).apply_native(&image, &backend)
            }
            None => InvertIntensityFilter::new().apply_native(&image, &backend),
        },
        FilterKind::Clamp { lower, upper } => {
            ClampImageFilter::new(*lower, *upper).apply_native(&image, &backend)
        }
        FilterKind::ShiftScale { shift, scale } => {
            ShiftScaleImageFilter::new(*shift, *scale).apply_native(&image, &backend)
        }
        FilterKind::RescaleIntensity { out_min, out_max } => {
            RescaleIntensityFilter::new(*out_min, *out_max).apply_native(&image, &backend)
        }
        FilterKind::NormalizeIntensity => {
            NormalizeImageFilter::new().apply_native(&image, &backend)
        }
        FilterKind::FlipZ => FlipImageFilter::flip_z().apply_native(&image, &backend),
        FilterKind::FlipY => FlipImageFilter::flip_y().apply_native(&image, &backend),
        FilterKind::FlipX => FlipImageFilter::flip_x().apply_native(&image, &backend),
        FilterKind::RegionOfInterest {
            start_z,
            start_y,
            start_x,
            size_z,
            size_y,
            size_x,
        } => RegionOfInterestImageFilter::new(
            [*start_z, *start_y, *start_x],
            [*size_z, *size_y, *size_x],
        )
        .apply_native(&image, &backend),
        FilterKind::PermuteAxes {
            order_0,
            order_1,
            order_2,
        } => PermuteAxesImageFilter::new([*order_0, *order_1, *order_2])
            .apply_native(&image, &backend),
        FilterKind::Shrink {
            factor_z,
            factor_y,
            factor_x,
        } => TileMeanShrinkFilter::new([*factor_z, *factor_y, *factor_x])
            .apply_native(&image, &backend),
        FilterKind::ConstantPad {
            pad_lower_z,
            pad_lower_y,
            pad_lower_x,
            pad_upper_z,
            pad_upper_y,
            pad_upper_x,
            constant,
        } => ConstantPadImageFilter::new(
            ritk_filter::Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
            ritk_filter::Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
            *constant,
        )
        .apply_native(&image, &backend),
        FilterKind::MirrorPad {
            pad_lower_z,
            pad_lower_y,
            pad_lower_x,
            pad_upper_z,
            pad_upper_y,
            pad_upper_x,
        } => MirrorPadImageFilter::new(
            ritk_filter::Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
            ritk_filter::Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
        )
        .apply_native(&image, &backend),
        FilterKind::WrapPad {
            pad_lower_z,
            pad_lower_y,
            pad_lower_x,
            pad_upper_z,
            pad_upper_y,
            pad_upper_x,
        } => WrapPadImageFilter::new(
            ritk_filter::Padding::new([*pad_lower_z, *pad_lower_y, *pad_lower_x]),
            ritk_filter::Padding::new([*pad_upper_z, *pad_upper_y, *pad_upper_x]),
        )
        .apply_native(&image, &backend),
        FilterKind::MaskThreshold { threshold } => {
            MaskImageFilter::apply_threshold_native(&image, *threshold, &backend)
        }
        FilterKind::Atan => AtanImageFilter::new().apply_native(&image, &backend),
        FilterKind::Sin => SinImageFilter::new().apply_native(&image, &backend),
        FilterKind::Cos => CosImageFilter::new().apply_native(&image, &backend),
        FilterKind::Tan => TanImageFilter::new().apply_native(&image, &backend),
        FilterKind::Asin => AsinImageFilter::new().apply_native(&image, &backend),
        FilterKind::Acos => AcosImageFilter::new().apply_native(&image, &backend),
        FilterKind::BoundedReciprocal => {
            BoundedReciprocalImageFilter::new().apply_native(&image, &backend)
        }
        FilterKind::Mean { radius } => MeanImageFilter::new(*radius).apply_native(&image, &backend),
        FilterKind::GrayscaleErode { radius } => {
            GrayscaleErosion::new(*radius).apply_native(&image, &backend)
        }
        FilterKind::GrayscaleDilate { radius } => {
            GrayscaleDilation::new(*radius).apply_native(&image, &backend)
        }
        FilterKind::GrayscaleClosing { radius } => {
            GrayscaleClosingFilter::new(*radius).apply_native(&image, &backend)
        }
        FilterKind::GrayscaleOpening { radius } => {
            GrayscaleOpeningFilter::new(*radius).apply_native(&image, &backend)
        }
        FilterKind::MorphologicalGradient { radius } => {
            GrayscaleMorphologicalGradientFilter::new(*radius).apply_native(&image, &backend)
        }
        FilterKind::BinaryContour {
            connectivity,
            foreground_value,
        } => BinaryContourImageFilter::new(*connectivity, *foreground_value)
            .apply_native(&image, &backend),
        FilterKind::LabelContour {
            connectivity,
            background_value,
        } => LabelContourImageFilter::new(*connectivity, *background_value)
            .apply_native(&image, &backend),
        FilterKind::VotingBinary {
            radius,
            birth_threshold,
            survival_threshold,
            foreground_value,
            background_value,
        } => VotingBinaryImageFilter::new(
            *radius,
            *birth_threshold,
            *survival_threshold,
            *foreground_value,
            *background_value,
        )
        .apply_native(&image, &backend),
        FilterKind::GrayscaleFillhole => {
            GrayscaleFillholeFilter::new().apply_native(&image, &backend)
        }
        FilterKind::GeodesicDilationSelf => {
            GrayscaleGeodesicDilationFilter::new().apply_native(&image, &image, &backend)
        }
        FilterKind::GeodesicErosionSelf => {
            GrayscaleGeodesicErosionFilter::new().apply_native(&image, &image, &backend)
        }
        FilterKind::ZeroCrossing {
            foreground_value,
            background_value,
        } => ZeroCrossingImageFilter::new()
            .with_foreground(*foreground_value)
            .with_background(*background_value)
            .apply_native(&image, &backend),
        FilterKind::BinaryErode {
            radius,
            foreground_value,
        } => binary_erode(&image, *radius, *foreground_value, &backend),
        FilterKind::BinaryDilate {
            radius,
            foreground_value,
        } => binary_dilate(&image, *radius, *foreground_value, &backend),
        FilterKind::BinaryClosing {
            radius,
            foreground_value,
        } => binary_closing(&image, *radius, *foreground_value, &backend),
        FilterKind::BinaryOpening {
            radius,
            foreground_value,
        } => binary_opening(&image, *radius, *foreground_value, &backend),
        FilterKind::BinaryFillhole { foreground_value } => {
            binary_fill_holes(&image, *foreground_value, &backend)
        }
        FilterKind::DistanceTransform { threshold } => {
            distance_transform(&image, *threshold, &backend)
        }
        FilterKind::SignedDistanceTransform { threshold } => {
            signed_distance_transform(&image, *threshold, &backend)
        }
        FilterKind::ConnectedComponents {
            connectivity,
            background_value,
        } => {
            let connectivity = match connectivity {
                ritk_filter::Connectivity::Face6 => SegmentationConnectivity::Six,
                ritk_filter::Connectivity::Vertex26 => SegmentationConnectivity::TwentySix,
            };
            connected_components(&image, connectivity, *background_value, &backend)
                .map(|(labels, _statistics)| labels)
        }
        FilterKind::RelabelComponents {
            minimum_object_size,
        } => relabel_components(&image, *minimum_object_size as usize, &backend)
            .map(|(labels, _statistics)| labels),
        FilterKind::MultiOtsuThreshold { num_classes } => {
            multi_otsu(&image, *num_classes as usize, 256, &backend)
        }
        FilterKind::Median { radius } => MedianFilter::new(*radius).apply_native(&image, &backend),
        FilterKind::HistEq { bins } => {
            HistogramEqualizationFilter::new(*bins).apply_native(&image, &backend)
        }
        FilterKind::Clahe {
            tile_grid_size,
            clip_limit,
        } => ClaheFilter::new(*tile_grid_size, *clip_limit, 256).apply_native(&image, &backend),
        FilterKind::GradientAnisotropicDiffusion {
            iterations,
            time_step,
            conductance,
        } => GradientAnisotropicDiffusionFilter::new(GradientDiffusionConfig {
            num_iterations: *iterations as usize,
            time_step: *time_step,
            conductance: *conductance,
        })
        .apply_native(&image, &backend),
        FilterKind::ConnectedThreshold {
            seed_z,
            seed_y,
            seed_x,
            lower,
            upper,
        } => connected_threshold(
            &image,
            ritk_spatial::VoxelIndex::from([*seed_z, *seed_y, *seed_x]),
            *lower,
            *upper,
            &backend,
        ),
        FilterKind::ConfidenceConnected {
            seed_z,
            seed_y,
            seed_x,
            initial_lower,
            initial_upper,
            multiplier,
            max_iterations,
        } => confidence_connected(
            &image,
            ritk_spatial::VoxelIndex::from([*seed_z, *seed_y, *seed_x]),
            *initial_lower,
            *initial_upper,
            *multiplier,
            *max_iterations as usize,
            &backend,
        ),
        FilterKind::NeighborhoodConnected {
            seed_z,
            seed_y,
            seed_x,
            lower,
            upper,
            radius_z,
            radius_y,
            radius_x,
        } => neighborhood_connected(
            &image,
            ritk_spatial::VoxelIndex::from([*seed_z, *seed_y, *seed_x]),
            *lower,
            *upper,
            [*radius_z, *radius_y, *radius_x],
            &backend,
        ),
        FilterKind::BinaryThreshold {
            lower,
            upper,
            foreground,
            background,
        } => binary_threshold(
            &image,
            *lower,
            *upper,
            (*foreground).into(),
            *background,
            &backend,
        ),
        _ => unreachable!("invariant: dispatch admits only fully native filter variants"),
    }
    .context("Coeus-native filter failed")?;

    let origin = output.origin();
    let spacing = output.spacing();
    let direction = output.direction();
    Ok(NativeFilterOutput {
        data: output.data_cow_on(&backend).into_owned(),
        shape: output.shape(),
        origin: [origin[0], origin[1], origin[2]],
        spacing: [spacing[0], spacing[1], spacing[2]],
        direction: [
            direction[(0, 0)],
            direction[(0, 1)],
            direction[(0, 2)],
            direction[(1, 0)],
            direction[(1, 1)],
            direction[(1, 2)],
            direction[(2, 0)],
            direction[(2, 1)],
            direction[(2, 2)],
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::apply_if_supported;
    use crate::app::tests::test_volume;
    use crate::app::SnapApp;
    use crate::FilterKind;
    use ritk_filter::{BinarizationThreshold, ForegroundValue};
    use std::sync::Arc;

    #[test]
    fn native_unary_filters_transform_loaded_volume_values() {
        let cases = [
            (FilterKind::Abs, vec![1.0, 0.0, 4.0, 1.0]),
            (FilterKind::Square, vec![1.0, 0.0, 16.0, 1.0]),
            (FilterKind::Sqrt, vec![f32::NAN, 0.0, 2.0, 1.0]),
            (
                FilterKind::Log,
                vec![f32::NAN, f32::NEG_INFINITY, 4.0_f32.ln(), 0.0],
            ),
            (
                FilterKind::Exp,
                vec![(-1.0_f32).exp(), 1.0, 4.0_f32.exp(), 1.0_f32.exp()],
            ),
        ];

        for (filter, expected) in cases {
            let mut volume = test_volume([1, 2, 2]);
            volume.data = Arc::new(vec![-1.0, 0.0, 4.0, 1.0]);
            volume.origin = [2.0, 3.0, 5.0];
            volume.spacing = [0.5, 1.5, 2.5];

            let result = apply_if_supported(&volume, &filter)
                .expect("invariant: unary filter has a native implementation")
                .expect("native unary filter succeeds for a scalar volume");

            assert_eq!(result.len(), expected.len());
            for (actual, expected) in result.iter().zip(expected) {
                if expected.is_nan() {
                    assert!(actual.is_nan(), "expected NaN, received {actual}");
                } else {
                    assert_eq!(*actual, expected);
                }
            }
        }
    }

    #[test]
    fn native_invert_intensity_honors_fixed_and_automatic_maxima() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![1.0, 4.0, 7.0]);
        let fixed = apply_if_supported(
            &volume,
            &FilterKind::InvertIntensity {
                maximum: Some(10.0),
            },
        )
        .expect("invariant: inversion has a native implementation")
        .expect("native inversion succeeds");
        assert_eq!(fixed, vec![9.0, 6.0, 3.0]);

        let automatic = apply_if_supported(&volume, &FilterKind::InvertIntensity { maximum: None })
            .expect("invariant: inversion has a native implementation")
            .expect("native inversion succeeds");
        assert_eq!(automatic, vec![6.0, 3.0, 0.0]);
    }

    #[test]
    fn native_clamp_limits_loaded_volume_values() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![-5.0, 50.0, 300.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::Clamp {
                lower: 0.0,
                upper: 100.0,
            },
        )
        .expect("invariant: clamp has a native implementation")
        .expect("native clamp succeeds");

        assert_eq!(output, vec![0.0, 50.0, 100.0]);
    }

    #[test]
    fn native_shift_scale_preserves_hu_conversion() {
        let mut volume = test_volume([1, 1, 2]);
        volume.data = Arc::new(vec![1024.0, 0.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::ShiftScale {
                shift: -1024.0,
                scale: 0.001,
            },
        )
        .expect("invariant: shift-scale has a native implementation")
        .expect("native shift-scale succeeds");
        assert_eq!(output, vec![0.0, -1.024]);
    }

    #[test]
    fn native_rescale_maps_loaded_volume_range() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 50.0, 100.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::RescaleIntensity {
                out_min: -1.0,
                out_max: 1.0,
            },
        )
        .expect("invariant: rescale has a native implementation")
        .expect("native rescale succeeds");
        assert_eq!(output, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn native_normalize_uses_sample_standard_deviation() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![1.0, 2.0, 3.0]);
        let output = apply_if_supported(&volume, &FilterKind::NormalizeIntensity)
            .expect("invariant: normalization has a native implementation")
            .expect("native normalization succeeds");
        assert_eq!(output, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn native_flip_x_reverses_loaded_volume_values() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![1.0, 2.0, 3.0]);
        let output = apply_if_supported(&volume, &FilterKind::FlipX)
            .expect("invariant: flip-x has a native implementation")
            .expect("native flip succeeds");
        assert_eq!(output, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn native_roi_updates_shape_and_physical_origin() {
        let mut volume = test_volume([2, 2, 2]);
        volume.data = Arc::new((1..=8).map(|value| value as f32).collect());
        volume.spacing = [2.0, 3.0, 4.0];
        let output = apply_if_supported(
            &volume,
            &FilterKind::RegionOfInterest {
                start_z: 1,
                start_y: 1,
                start_x: 1,
                size_z: 1,
                size_y: 1,
                size_x: 1,
            },
        )
        .expect("invariant: ROI has a native implementation")
        .expect("native ROI accepts an in-bounds scalar volume");

        assert_eq!(output.data, vec![8.0]);
        assert_eq!(output.shape, [1, 1, 1]);
        assert_eq!(output.origin, [2.0, 3.0, 4.0]);
        assert_eq!(output.spacing, [2.0, 3.0, 4.0]);
    }

    #[test]
    fn native_axis_permutation_updates_values_and_geometry() {
        let mut volume = test_volume([2, 1, 3]);
        volume.data = Arc::new((1..=6).map(|value| value as f32).collect());
        volume.origin = [5.0, 7.0, 11.0];
        volume.spacing = [1.0, 2.0, 3.0];
        let output = apply_if_supported(
            &volume,
            &FilterKind::PermuteAxes {
                order_0: 2,
                order_1: 1,
                order_2: 0,
            },
        )
        .expect("invariant: axis permutation has a native implementation")
        .expect("native axis permutation accepts a valid order");

        assert_eq!(output.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(output.shape, [3, 1, 2]);
        assert_eq!(output.origin, [5.0, 7.0, 11.0]);
        assert_eq!(output.spacing, [3.0, 2.0, 1.0]);
    }

    #[test]
    fn native_tile_mean_shrink_updates_values_and_geometry() {
        let mut volume = test_volume([1, 1, 4]);
        volume.data = Arc::new(vec![0.0, 2.0, 4.0, 6.0]);
        volume.origin = [5.0, 7.0, 11.0];
        volume.spacing = [1.0, 2.0, 3.0];
        let output = apply_if_supported(
            &volume,
            &FilterKind::Shrink {
                factor_z: 1,
                factor_y: 1,
                factor_x: 2,
            },
        )
        .expect("invariant: tile-mean shrink has a native implementation")
        .expect("native tile-mean shrink accepts valid factors");

        assert_eq!(output.data, vec![1.0, 5.0]);
        assert_eq!(output.shape, [1, 1, 2]);
        assert_eq!(output.origin, [5.0, 7.0, 11.0]);
        assert_eq!(output.spacing, [1.0, 2.0, 6.0]);
    }

    #[test]
    fn native_constant_padding_updates_shape_and_origin() {
        let mut volume = test_volume([1, 1, 1]);
        volume.data = Arc::new(vec![3.0]);
        volume.origin = [0.0, 0.0, 10.0];
        volume.spacing = [1.0, 1.0, 2.0];
        let output = apply_if_supported(
            &volume,
            &FilterKind::ConstantPad {
                pad_lower_z: 0,
                pad_lower_y: 0,
                pad_lower_x: 1,
                pad_upper_z: 0,
                pad_upper_y: 0,
                pad_upper_x: 0,
                constant: -1.0,
            },
        )
        .expect("invariant: constant padding has a native implementation")
        .expect("native constant padding succeeds");

        assert_eq!(output.data, vec![-1.0, 3.0]);
        assert_eq!(output.shape, [1, 1, 2]);
        assert_eq!(output.origin, [0.0, 0.0, 8.0]);
    }

    #[test]
    fn native_mirror_and_wrap_padding_preserve_policy_values() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![1.0, 2.0, 3.0]);
        let cases = [
            (
                FilterKind::MirrorPad {
                    pad_lower_z: 0,
                    pad_lower_y: 0,
                    pad_lower_x: 1,
                    pad_upper_z: 0,
                    pad_upper_y: 0,
                    pad_upper_x: 1,
                },
                vec![1.0, 1.0, 2.0, 3.0, 3.0],
            ),
            (
                FilterKind::WrapPad {
                    pad_lower_z: 0,
                    pad_lower_y: 0,
                    pad_lower_x: 1,
                    pad_upper_z: 0,
                    pad_upper_y: 0,
                    pad_upper_x: 1,
                },
                vec![3.0, 1.0, 2.0, 3.0, 1.0],
            ),
        ];
        for (filter, expected) in cases {
            let output = apply_if_supported(&volume, &filter)
                .expect("invariant: padding policy has a native implementation")
                .expect("native padding policy succeeds");
            assert_eq!(output.data, expected);
            assert_eq!(output.shape, [1, 1, 5]);
        }
    }

    #[test]
    fn native_mask_threshold_keeps_only_strictly_greater_values() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.5, 0.5001, 2.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::MaskThreshold {
                threshold: BinarizationThreshold::DEFAULT,
            },
        )
        .expect("invariant: threshold masking has a native implementation")
        .expect("native threshold masking succeeds");

        assert_eq!(output.data, vec![0.0, 0.5001, 2.0]);
        assert_eq!(output.shape, [1, 1, 3]);
    }

    #[test]
    fn native_trigonometric_family_preserves_known_values() {
        let cases = [
            (FilterKind::Atan, 1.0, core::f32::consts::FRAC_PI_4),
            (FilterKind::Sin, 0.0, 0.0),
            (FilterKind::Cos, 0.0, 1.0),
            (FilterKind::Tan, 0.0, 0.0),
            (FilterKind::Asin, 1.0, core::f32::consts::FRAC_PI_2),
            (FilterKind::Acos, 0.0, core::f32::consts::FRAC_PI_2),
            (FilterKind::BoundedReciprocal, -1.0, 0.5),
        ];
        for (filter, input, expected) in cases {
            let mut volume = test_volume([1, 1, 1]);
            volume.data = Arc::new(vec![input]);
            let output = apply_if_supported(&volume, &filter)
                .expect("invariant: trig filter has a native implementation")
                .expect("native trig filter succeeds");
            assert!(
                (output.data[0] - expected).abs() <= 2.0 * f32::EPSILON,
                "native trig output {} differs from expected {expected}",
                output.data[0]
            );
        }
    }

    #[test]
    fn native_mean_preserves_zero_flux_boundary_values() {
        let mut volume = test_volume([1, 1, 4]);
        volume.data = Arc::new(vec![0.0, 0.0, 10.0, 10.0]);
        let output = apply_if_supported(&volume, &FilterKind::Mean { radius: 1 })
            .expect("invariant: mean filter has a native implementation")
            .expect("native mean filter succeeds");

        assert_eq!(output.shape, [1, 1, 4]);
        assert!((output.data[1] - 10.0 / 3.0).abs() <= 1e-6);
        assert!((output.data[2] - 20.0 / 3.0).abs() <= 1e-6);
    }

    #[test]
    fn native_grayscale_morphology_uses_shared_extrema_kernels() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 2.0, 1.0]);
        let cases = [
            (
                FilterKind::GrayscaleErode { radius: 1 },
                vec![0.0, 0.0, 1.0],
            ),
            (
                FilterKind::GrayscaleDilate { radius: 1 },
                vec![2.0, 2.0, 2.0],
            ),
        ];
        for (filter, expected) in cases {
            let output = apply_if_supported(&volume, &filter)
                .expect("invariant: grayscale morphology has a native implementation")
                .expect("native grayscale morphology succeeds");
            assert_eq!(output.data, expected);
        }
    }

    #[test]
    fn native_grayscale_opening_and_closing_preserve_composition_order() {
        let cases = [
            (
                FilterKind::GrayscaleClosing { radius: 1 },
                vec![1.0, 0.0, 1.0],
                vec![1.0, 1.0, 1.0],
            ),
            (
                FilterKind::GrayscaleOpening { radius: 1 },
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ),
        ];
        for (filter, values, expected) in cases {
            let mut volume = test_volume([1, 1, 3]);
            volume.data = Arc::new(values);
            let output = apply_if_supported(&volume, &filter)
                .expect("invariant: grayscale composition has a native implementation")
                .expect("native grayscale composition succeeds");
            assert_eq!(output.data, expected);
        }
    }

    #[test]
    fn native_morphological_gradient_uses_extrema_difference() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 10.0, 0.0]);
        let output = apply_if_supported(&volume, &FilterKind::MorphologicalGradient { radius: 1 })
            .expect("invariant: morphological gradient has a native implementation")
            .expect("native morphological gradient succeeds");

        assert_eq!(output.data, vec![10.0, 10.0, 10.0]);
        assert!(output.data.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn native_binary_contour_removes_fully_enclosed_foreground() {
        let mut volume = test_volume([3, 3, 3]);
        volume.data = Arc::new(vec![1.0; 27]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::BinaryContour {
                connectivity: ritk_filter::Connectivity::Face6,
                foreground_value: ForegroundValue::ONE,
            },
        )
        .expect("invariant: binary contour has a native implementation")
        .expect("native binary contour succeeds");

        assert_eq!(output.data[13], 0.0);
        assert_eq!(output.data[0], 0.0);
    }

    #[test]
    fn native_label_contour_retains_only_different_label_boundaries() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![1.0, 1.0, 2.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::LabelContour {
                connectivity: ritk_filter::Connectivity::Face6,
                background_value: 0.0,
            },
        )
        .expect("invariant: label contour has a native implementation")
        .expect("native label contour succeeds");

        assert_eq!(output.data, vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn native_voting_binary_applies_single_step_birth_rule() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 1.0, 0.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::VotingBinary {
                radius: 1,
                birth_threshold: 1,
                survival_threshold: 1,
                foreground_value: ForegroundValue::ONE,
                background_value: 0.0,
            },
        )
        .expect("invariant: voting binary has a native implementation")
        .expect("native voting binary succeeds");

        assert_eq!(output.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn native_grayscale_fillhole_raises_enclosed_minimum() {
        let mut volume = test_volume([3, 3, 3]);
        let mut values = vec![1.0; 27];
        values[13] = 0.0;
        volume.data = Arc::new(values);
        let output = apply_if_supported(&volume, &FilterKind::GrayscaleFillhole)
            .expect("invariant: grayscale fill-hole has a native implementation")
            .expect("native grayscale fill-hole succeeds");

        assert_eq!(output.data[13], 1.0);
        assert_eq!(output.data[0], 1.0);
    }

    #[test]
    fn native_self_geodesic_filters_are_fixed_points() {
        let cases = [
            FilterKind::GeodesicDilationSelf,
            FilterKind::GeodesicErosionSelf,
        ];
        for filter in cases {
            let mut volume = test_volume([1, 1, 3]);
            volume.data = Arc::new(vec![0.0, 2.0, 1.0]);
            let output = apply_if_supported(&volume, &filter)
                .expect("invariant: self geodesic filter has a native implementation")
                .expect("native self geodesic reconstruction succeeds");
            assert_eq!(output.data, vec![0.0, 2.0, 1.0]);
        }
    }

    #[test]
    fn native_zero_crossing_uses_near_zero_tie_breaking() {
        let mut volume = test_volume([1, 1, 2]);
        volume.data = Arc::new(vec![-1.0, 2.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::ZeroCrossing {
                foreground_value: ForegroundValue::ONE,
                background_value: 0.0,
            },
        )
        .expect("invariant: zero crossing has a native implementation")
        .expect("native zero crossing succeeds");

        assert_eq!(output.data, vec![1.0, 0.0]);
    }

    #[test]
    fn native_unary_filters_reject_color_volumes() {
        let mut volume = test_volume([1, 1, 1]);
        volume.channels = 3;
        volume.data = Arc::new(vec![0.0, 0.0, 0.0]);

        let error = apply_if_supported(&volume, &FilterKind::Abs)
            .expect("invariant: abs filter has a native implementation")
            .expect_err("color volume violates scalar filter contract");

        assert!(error.to_string().contains("3 interleaved channels"));
    }

    #[test]
    fn native_binary_morphology_preserves_exact_contracts() {
        let identity = vec![0.0, 1.0, 0.0];
        let identity_variants = [
            FilterKind::BinaryErode {
                radius: 0,
                foreground_value: ForegroundValue::ONE,
            },
            FilterKind::BinaryDilate {
                radius: 0,
                foreground_value: ForegroundValue::ONE,
            },
            FilterKind::BinaryClosing {
                radius: 0,
                foreground_value: ForegroundValue::ONE,
            },
            FilterKind::BinaryOpening {
                radius: 0,
                foreground_value: ForegroundValue::ONE,
            },
        ];

        for filter in identity_variants {
            let mut volume = test_volume([1, 1, 3]);
            volume.data = Arc::new(identity.clone());
            let output = apply_if_supported(&volume, &filter)
                .expect("invariant: binary morphology has a native implementation")
                .expect("native binary morphology accepts a scalar volume");
            assert_eq!(output, identity);
        }

        let mut volume = test_volume([3, 3, 3]);
        let mut values = vec![1.0; 27];
        values[13] = 0.0;
        volume.data = Arc::new(values);
        let output = apply_if_supported(
            &volume,
            &FilterKind::BinaryFillhole {
                foreground_value: ForegroundValue::ONE,
            },
        )
        .expect("invariant: binary fill-hole has a native implementation")
        .expect("native binary fill-hole accepts a scalar volume");
        assert_eq!(output, vec![1.0; 27]);
    }

    #[test]
    fn native_distance_transform_uses_loaded_volume_spacing() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 1.0, 0.0]);
        volume.spacing = [0.5, 1.5, 2.5];
        let output = apply_if_supported(
            &volume,
            &FilterKind::DistanceTransform {
                threshold: BinarizationThreshold::DEFAULT,
            },
        )
        .expect("invariant: unsigned distance transform has a native implementation")
        .expect("native distance transform accepts a scalar volume");

        assert_eq!(output, vec![2.5, 0.0, 2.5]);
    }

    #[test]
    fn native_signed_distance_transform_uses_voxel_centre_convention() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 1.0, 0.0]);
        volume.spacing = [1.0, 1.0, 2.0];
        let output = apply_if_supported(
            &volume,
            &FilterKind::SignedDistanceTransform {
                threshold: BinarizationThreshold::DEFAULT,
            },
        )
        .expect("invariant: signed distance transform has a native implementation")
        .expect("native signed distance transform accepts a scalar volume");

        assert_eq!(output, vec![2.0, -2.0, 2.0]);
    }

    #[test]
    fn native_connected_components_preserves_labels_and_connectivity() {
        let mut volume = test_volume([2, 2, 2]);
        volume.data = Arc::new(vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::ConnectedComponents {
                connectivity: ritk_filter::Connectivity::Face6,
                background_value: 0.0,
            },
        )
        .expect("invariant: connected components has a native implementation")
        .expect("native connected components accepts a scalar volume");

        assert_eq!(output, vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn native_relabel_components_orders_sizes_and_discards_small_labels() {
        let mut volume = test_volume([1, 1, 5]);
        volume.data = Arc::new(vec![1.0, 2.0, 2.0, 2.0, 0.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::RelabelComponents {
                minimum_object_size: 2,
            },
        )
        .expect("invariant: relabel components has a native implementation")
        .expect("native relabel components accepts a scalar label volume");

        assert_eq!(output, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn native_multi_otsu_assigns_ordered_intensity_classes() {
        let mut volume = test_volume([1, 1, 6]);
        volume.data = Arc::new(vec![0.0, 0.0, 10.0, 10.0, 100.0, 100.0]);
        let output =
            apply_if_supported(&volume, &FilterKind::MultiOtsuThreshold { num_classes: 3 })
                .expect("invariant: multi-otsu has a native implementation")
                .expect("native multi-otsu accepts a scalar volume");

        assert_eq!(output, vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn native_median_removes_an_impulse() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 10.0, 0.0]);
        let output = apply_if_supported(&volume, &FilterKind::Median { radius: 1 })
            .expect("invariant: median has a native implementation")
            .expect("native median accepts a scalar volume");

        assert_eq!(output, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn native_histogram_equalization_maps_values_through_the_cdf() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 0.0, 1.0]);
        let output = apply_if_supported(&volume, &FilterKind::HistEq { bins: 2 })
            .expect("invariant: histogram equalization has a native implementation")
            .expect("native histogram equalization accepts a scalar volume");

        assert_eq!(output, vec![2.0 / 3.0, 2.0 / 3.0, 1.0]);
    }

    #[test]
    fn native_clahe_preserves_a_uniform_slice() {
        let mut volume = test_volume([1, 2, 2]);
        volume.data = Arc::new(vec![42.5; 4]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::Clahe {
                tile_grid_size: [1, 1],
                clip_limit: 40.0,
            },
        )
        .expect("invariant: clahe has a native implementation")
        .expect("native clahe accepts a scalar volume");

        assert_eq!(output, vec![42.5; 4]);
    }

    #[test]
    fn native_gradient_diffusion_preserves_a_constant_volume() {
        let mut volume = test_volume([1, 2, 2]);
        volume.data = Arc::new(vec![42.5; 4]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::GradientAnisotropicDiffusion {
                iterations: 2,
                time_step: 0.125,
                conductance: 1.0,
            },
        )
        .expect("invariant: gradient diffusion has a native implementation")
        .expect("native gradient diffusion accepts a scalar volume");

        assert_eq!(output, vec![42.5; 4]);
    }

    #[test]
    fn native_connected_threshold_keeps_only_the_seed_component() {
        let mut volume = test_volume([1, 1, 4]);
        volume.data = Arc::new(vec![1.0, 1.0, 0.0, 1.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::ConnectedThreshold {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                lower: 1.0,
                upper: 1.0,
            },
        )
        .expect("invariant: connected threshold has a native implementation")
        .expect("native connected threshold accepts a scalar volume");
        assert_eq!(output, vec![1.0, 1.0, 0.0, 0.0]);
    }

    #[test]
    fn native_confidence_connected_selects_a_constant_seed_component() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![2.0; 3]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::ConfidenceConnected {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                initial_lower: 1.0,
                initial_upper: 3.0,
                multiplier: 2.5,
                max_iterations: 2,
            },
        )
        .expect("invariant: confidence connected has a native implementation")
        .expect("native confidence connected succeeds");
        assert_eq!(output, vec![1.0; 3]);
    }

    #[test]
    fn native_neighborhood_connected_rejects_an_invalid_seed_neighborhood() {
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![1.0, 0.0, 1.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::NeighborhoodConnected {
                seed_z: 0,
                seed_y: 0,
                seed_x: 0,
                lower: 1.0,
                upper: 1.0,
                radius_z: 0,
                radius_y: 0,
                radius_x: 1,
            },
        )
        .expect("invariant: neighborhood connected has a native implementation")
        .expect("native neighborhood connected succeeds");
        assert_eq!(output, vec![0.0; 3]);
    }

    #[test]
    fn native_binary_threshold_includes_both_bounds() {
        let mut volume = test_volume([1, 1, 5]);
        volume.data = Arc::new(vec![-1.0, 0.0, 50.0, 100.0, 101.0]);
        let output = apply_if_supported(
            &volume,
            &FilterKind::BinaryThreshold {
                lower: 0.0,
                upper: 100.0,
                foreground: ForegroundValue::ONE,
                background: 0.0,
            },
        )
        .expect("invariant: binary threshold has a native implementation")
        .expect("native binary threshold accepts a scalar volume");

        assert_eq!(output, vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn unsupported_filter_stays_outside_native_unary_dispatch() {
        let volume = test_volume([1, 1, 1]);
        assert!(apply_if_supported(&volume, &FilterKind::Gaussian { sigma: 1.0 }).is_none());
    }

    #[test]
    fn snap_app_applies_native_unary_filter_and_invalidates_render_caches() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 2]);
        volume.data = Arc::new(vec![-2.0, 3.0]);
        volume.origin = [2.0, 3.0, 5.0];
        volume.spacing = [0.5, 1.5, 2.5];
        app.loaded = Some(volume);
        app.active_filter = FilterKind::Abs;
        app.texture_dirty = false;
        app.coronal_dirty = false;
        app.sagittal_dirty = false;
        app.mip_dirty = false;

        app.apply_filter_to_loaded_volume();

        let volume = app.loaded.expect("volume remains loaded");
        assert_eq!(volume.data.as_slice(), [2.0, 3.0]);
        assert_eq!(volume.origin, [2.0, 3.0, 5.0]);
        assert_eq!(volume.spacing, [0.5, 1.5, 2.5]);
        assert!(app.texture_dirty);
        assert!(app.coronal_dirty);
        assert!(app.sagittal_dirty);
        assert!(app.mip_dirty);
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_binary_morphology() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 1.0, 0.0]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::BinaryDilate {
            radius: 1,
            foreground_value: ForegroundValue::ONE,
        };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [1.0, 1.0, 1.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_distance_transform() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 1.0, 0.0]);
        volume.spacing = [1.0, 1.0, 2.0];
        app.loaded = Some(volume);
        app.active_filter = FilterKind::DistanceTransform {
            threshold: BinarizationThreshold::DEFAULT,
        };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [2.0, 0.0, 2.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_signed_distance_transform() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 1.0, 0.0]);
        volume.spacing = [1.0, 1.0, 2.0];
        app.loaded = Some(volume);
        app.active_filter = FilterKind::SignedDistanceTransform {
            threshold: BinarizationThreshold::DEFAULT,
        };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [2.0, -2.0, 2.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_connected_components() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![1.0, 0.0, 1.0]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::ConnectedComponents {
            connectivity: ritk_filter::Connectivity::Face6,
            background_value: 0.0,
        };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [1.0, 0.0, 2.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_relabel_components() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 5]);
        volume.data = Arc::new(vec![1.0, 2.0, 2.0, 2.0, 0.0]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::RelabelComponents {
            minimum_object_size: 2,
        };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [0.0, 1.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_multi_otsu() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 6]);
        volume.data = Arc::new(vec![0.0, 0.0, 10.0, 10.0, 100.0, 100.0]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::MultiOtsuThreshold { num_classes: 3 };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [0.0, 0.0, 1.0, 1.0, 2.0, 2.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_median() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 10.0, 0.0]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::Median { radius: 1 };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [0.0, 0.0, 0.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_histogram_equalization() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 0.0, 1.0]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::HistEq { bins: 2 };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [2.0 / 3.0, 2.0 / 3.0, 1.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_clahe() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 2, 2]);
        volume.data = Arc::new(vec![42.5; 4]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::Clahe {
            tile_grid_size: [1, 1],
            clip_limit: 40.0,
        };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [42.5; 4]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_gradient_diffusion() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 2, 2]);
        volume.data = Arc::new(vec![42.5; 4]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::GradientAnisotropicDiffusion {
            iterations: 2,
            time_step: 0.125,
            conductance: 1.0,
        };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [42.5; 4]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_binary_threshold() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 3]);
        volume.data = Arc::new(vec![0.0, 1.0, 2.0]);
        app.loaded = Some(volume);
        app.active_filter = FilterKind::BinaryThreshold {
            lower: 1.0,
            upper: 2.0,
            foreground: ForegroundValue::ONE,
            background: 0.0,
        };

        app.apply_filter_to_loaded_volume();

        assert_eq!(
            app.loaded.expect("volume remains loaded").data.as_slice(),
            [0.0, 1.0, 1.0]
        );
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_roi_with_updated_geometry() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([2, 2, 2]);
        volume.data = Arc::new((1..=8).map(|value| value as f32).collect());
        volume.spacing = [2.0, 3.0, 4.0];
        app.loaded = Some(volume);
        app.active_filter = FilterKind::RegionOfInterest {
            start_z: 1,
            start_y: 1,
            start_x: 1,
            size_z: 1,
            size_y: 1,
            size_x: 1,
        };

        app.apply_filter_to_loaded_volume();

        let volume = app.loaded.expect("volume remains loaded after an ROI crop");
        assert_eq!(volume.data.as_slice(), [8.0]);
        assert_eq!(volume.shape, [1, 1, 1]);
        assert_eq!(volume.origin, [2.0, 3.0, 4.0]);
        assert_eq!(volume.spacing, [2.0, 3.0, 4.0]);
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_axis_permutation_with_updated_geometry() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([2, 1, 3]);
        volume.data = Arc::new((1..=6).map(|value| value as f32).collect());
        volume.origin = [5.0, 7.0, 11.0];
        volume.spacing = [1.0, 2.0, 3.0];
        app.loaded = Some(volume);
        app.active_filter = FilterKind::PermuteAxes {
            order_0: 2,
            order_1: 1,
            order_2: 0,
        };

        app.apply_filter_to_loaded_volume();

        let volume = app
            .loaded
            .expect("volume remains loaded after axis permutation");
        assert_eq!(volume.data.as_slice(), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(volume.shape, [3, 1, 2]);
        assert_eq!(volume.origin, [5.0, 7.0, 11.0]);
        assert_eq!(volume.spacing, [3.0, 2.0, 1.0]);
        assert_eq!(app.status_message, "Filter applied.");
    }

    #[test]
    fn snap_app_applies_native_tile_mean_shrink_with_updated_geometry() {
        let mut app = SnapApp::default();
        let mut volume = test_volume([1, 1, 4]);
        volume.data = Arc::new(vec![0.0, 2.0, 4.0, 6.0]);
        volume.origin = [5.0, 7.0, 11.0];
        volume.spacing = [1.0, 2.0, 3.0];
        app.loaded = Some(volume);
        app.active_filter = FilterKind::Shrink {
            factor_z: 1,
            factor_y: 1,
            factor_x: 2,
        };

        app.apply_filter_to_loaded_volume();

        let volume = app
            .loaded
            .expect("volume remains loaded after tile-mean shrink");
        assert_eq!(volume.data.as_slice(), [1.0, 5.0]);
        assert_eq!(volume.shape, [1, 1, 2]);
        assert_eq!(volume.origin, [5.0, 7.0, 11.0]);
        assert_eq!(volume.spacing, [1.0, 2.0, 6.0]);
        assert_eq!(app.status_message, "Filter applied.");
    }
}
