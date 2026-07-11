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
    AbsImageFilter, ClampImageFilter, ExpImageFilter, FlipImageFilter, InvertIntensityFilter,
    LogImageFilter, NormalizeImageFilter, RescaleIntensityFilter, ShiftScaleImageFilter,
    SqrtImageFilter, SquareImageFilter,
};
use ritk_image::native::Image;
use ritk_segmentation::{
    labeling::Connectivity as SegmentationConnectivity,
    native::{binary_threshold, connected_components},
};
use ritk_spatial::{Direction, Point, Spacing};

use crate::{FilterKind, LoadedVolume};

/// Apply a filter when the full operation is available on the Coeus-native
/// image substrate.
///
/// `None` means this operation has not completed its native migration and must
/// remain on the legacy graph. `Some(Err(_))` reports an input or provider
/// contract failure without falling back to another implementation.
pub(super) fn apply_if_supported(
    volume: &LoadedVolume,
    filter: &FilterKind,
) -> Option<Result<Vec<f32>>> {
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
            | FilterKind::BinaryThreshold { .. }
            | FilterKind::InvertIntensity { .. }
            | FilterKind::Clamp { .. }
            | FilterKind::ShiftScale { .. }
            | FilterKind::RescaleIntensity { .. }
            | FilterKind::NormalizeIntensity
            | FilterKind::FlipZ
            | FilterKind::FlipY
            | FilterKind::FlipX
    ) {
        return None;
    }

    Some(apply_supported_filter(volume, filter))
}

fn apply_supported_filter(volume: &LoadedVolume, filter: &FilterKind) -> Result<Vec<f32>> {
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

    Ok(output.data_cow_on(&backend).into_owned())
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
}
