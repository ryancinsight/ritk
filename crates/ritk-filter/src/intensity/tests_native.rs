//! Differential + analytical coverage for the Coeus-native intensity paths.
//!
//! Each native wrapper must be value-identical to the Coeus filter it mirrors —
//! both call the identical substrate-agnostic host core (shared harness in
//! `native_support::assert_coeus_matches_coeus`). A handful of analytical
//! oracles pin the mathematical contract independent of the Burn reference.

use crate::native_support::{
    assert_coeus_matches_coeus, assert_coeus_matches_coeus_pair, make_native_image, native_vals,
};
use coeus_core::SequentialBackend;

use super::adaptive_equalization::AdaptiveHistogramEqualizationFilter;
use super::bed_separation::{BedSeparationConfig, BedSeparationFilter, ComponentPolicy};
use super::binary_threshold::BinaryThresholdImageFilter;
use super::clamp::ClampImageFilter;
use super::equalization::HistogramEqualizationFilter;
use super::mask::{MaskImageFilter, MaskNegatedImageFilter, MaskedAssignImageFilter};
use super::rescale::RescaleIntensityFilter;
use super::shift_scale::ShiftScaleImageFilter;
use super::sigmoid::SigmoidImageFilter;
use super::threshold::ThresholdImageFilter;
use super::windowing::IntensityWindowingFilter;
use super::zero_crossing::ZeroCrossingImageFilter;

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

// ── Rescale ───────────────────────────────────────────────────────────────────

mod rescale {
    use super::*;

    fn check(vals: Vec<f32>, dims: [usize; 3], out_min: f32, out_max: f32) {
        assert_coeus_matches_coeus(
            vals,
            dims,
            |img| {
                RescaleIntensityFilter::new(out_min, out_max)
                    .apply(img)
                    .expect("burn rescale")
            },
            |img, backend| RescaleIntensityFilter::new(out_min, out_max).apply_native(img, backend),
        );
    }

    #[test]
    fn matches_coeus_ramp() {
        check(ramp(60), [3, 4, 5], 0.0, 1.0);
    }

    #[test]
    fn matches_coeus_arbitrary_range() {
        check(
            vec![-3.0, 2.0, 7.5, -1.0, 4.0, 9.0, 0.0, 5.0],
            [2, 2, 2],
            -2.0,
            8.0,
        );
    }

    #[test]
    fn matches_coeus_constant_field() {
        check(vec![3.0f32; 27], [3, 3, 3], 0.0, 1.0);
    }

    #[test]
    fn oracle_endpoints_map_exactly() {
        // Affine bijection maps I_min -> out_min and I_max -> out_max exactly.
        let img = make_native_image(ramp(24), [2, 3, 4]);
        let out = RescaleIntensityFilter::new(0.0, 1.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native rescale");
        let v = native_vals(&out);
        let mn = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(mn, 0.0, "rescaled minimum endpoint must be exactly out_min");
        assert_eq!(mx, 1.0, "rescaled maximum endpoint must be exactly out_max");
    }
}

// ── Windowing ─────────────────────────────────────────────────────────────────

mod windowing {
    use super::*;

    fn check(vals: Vec<f32>, dims: [usize; 3], wmin: f32, wmax: f32, omin: f32, omax: f32) {
        assert_coeus_matches_coeus(
            vals,
            dims,
            |img| {
                IntensityWindowingFilter::new(wmin, wmax, omin, omax)
                    .apply(img)
                    .expect("burn windowing")
            },
            |img, backend| {
                IntensityWindowingFilter::new(wmin, wmax, omin, omax).apply_native(img, backend)
            },
        );
    }

    #[test]
    fn matches_coeus_interior_and_saturation() {
        check(ramp(60), [3, 4, 5], 20.0, 40.0, 0.0, 1.0);
    }

    #[test]
    fn matches_coeus_degenerate_window() {
        check(ramp(8), [2, 2, 2], 5.0, 5.0, 0.0, 1.0);
    }

    #[test]
    fn oracle_saturation_clamps_to_out_range() {
        let img = make_native_image(ramp(60), [3, 4, 5]);
        let out = IntensityWindowingFilter::new(20.0, 40.0, 0.0, 1.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native windowing");
        for &v in &native_vals(&out) {
            assert!((0.0..=1.0).contains(&v), "windowed value {v} out of [0,1]");
        }
    }
}

// ── Threshold ─────────────────────────────────────────────────────────────────

mod threshold {
    use super::*;

    fn check_below(vals: Vec<f32>, dims: [usize; 3], t: f32, out: f32) {
        assert_coeus_matches_coeus(
            vals,
            dims,
            |img| {
                ThresholdImageFilter::below(t, out)
                    .apply(img)
                    .expect("burn below")
            },
            |img, backend| ThresholdImageFilter::below(t, out).apply_native(img, backend),
        );
    }

    #[test]
    fn matches_coeus_below() {
        check_below(ramp(24), [2, 3, 4], 10.0, -1.0);
    }

    #[test]
    fn matches_coeus_above() {
        let vals = ramp(24);
        assert_coeus_matches_coeus(
            vals,
            [2, 3, 4],
            |img| {
                ThresholdImageFilter::above(15.0, -1.0)
                    .apply(img)
                    .expect("burn above")
            },
            |img, backend| ThresholdImageFilter::above(15.0, -1.0).apply_native(img, backend),
        );
    }

    #[test]
    fn matches_coeus_outside() {
        let vals = ramp(24);
        assert_coeus_matches_coeus(
            vals,
            [2, 3, 4],
            |img| {
                ThresholdImageFilter::outside(6.0, 18.0, 0.0)
                    .apply(img)
                    .expect("burn outside")
            },
            |img, backend| ThresholdImageFilter::outside(6.0, 18.0, 0.0).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_below_indicator() {
        // Values strictly below threshold -> outside_value; others unchanged.
        let img = make_native_image(ramp(24), [2, 3, 4]);
        let out = ThresholdImageFilter::below(10.0, -1.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native below");
        for (i, &v) in native_vals(&out).iter().enumerate() {
            let expected = if (i as f32) < 10.0 { -1.0 } else { i as f32 };
            assert_eq!(v, expected, "threshold-below indicator wrong at {i}");
        }
    }
}

// ── Sigmoid ───────────────────────────────────────────────────────────────────

mod sigmoid {
    use super::*;

    #[test]
    fn matches_coeus() {
        let vals = ramp(60);
        assert_coeus_matches_coeus(
            vals,
            [3, 4, 5],
            |img| {
                SigmoidImageFilter::new(30.0, 8.0, 0.0, 1.0)
                    .apply(img)
                    .expect("burn sigmoid")
            },
            |img, backend| SigmoidImageFilter::new(30.0, 8.0, 0.0, 1.0).apply_native(img, backend),
        );
    }

    #[test]
    fn matches_coeus_step_degenerate_beta() {
        let vals = ramp(8);
        assert_coeus_matches_coeus(
            vals,
            [2, 2, 2],
            |img| {
                SigmoidImageFilter::new(3.5, 0.0, 0.0, 1.0)
                    .apply(img)
                    .expect("burn sigmoid step")
            },
            |img, backend| SigmoidImageFilter::new(3.5, 0.0, 0.0, 1.0).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_midpoint_is_half() {
        // At v = alpha, sigmoid = (max + min) / 2 = 0.5 for [0,1] output.
        let img = make_native_image(vec![10.0f32; 8], [2, 2, 2]);
        let out = SigmoidImageFilter::new(10.0, 4.0, 0.0, 1.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native sigmoid");
        for &v in &native_vals(&out) {
            assert!(
                (v - 0.5).abs() < 1e-6,
                "sigmoid midpoint must be 0.5, got {v}"
            );
        }
    }
}

// ── Binary threshold ──────────────────────────────────────────────────────────

mod binary_threshold {
    use super::*;

    #[test]
    fn matches_coeus() {
        let vals = ramp(24);
        assert_coeus_matches_coeus(
            vals,
            [2, 3, 4],
            |img| {
                BinaryThresholdImageFilter::new(6.0, 18.0, 1.0, 0.0)
                    .apply(img)
                    .expect("burn binary threshold")
            },
            |img, backend| {
                BinaryThresholdImageFilter::new(6.0, 18.0, 1.0, 0.0).apply_native(img, backend)
            },
        );
    }

    #[test]
    fn oracle_indicator_and_binary_range() {
        let img = make_native_image(ramp(24), [2, 3, 4]);
        let out = BinaryThresholdImageFilter::new(6.0, 18.0, 1.0, 0.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native binary threshold");
        for (i, &v) in native_vals(&out).iter().enumerate() {
            let inside = (6.0..=18.0).contains(&(i as f32));
            let expected = if inside { 1.0 } else { 0.0 };
            assert_eq!(v, expected, "binary indicator wrong at {i}");
        }
    }
}

// ── Clamp ─────────────────────────────────────────────────────────────────────

mod clamp {
    use super::*;

    #[test]
    fn matches_coeus() {
        assert_coeus_matches_coeus(
            ramp(60),
            [3, 4, 5],
            |img| {
                ClampImageFilter::new(10.0, 40.0)
                    .apply(img)
                    .expect("burn clamp")
            },
            |img, backend| ClampImageFilter::new(10.0, 40.0).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_projects_into_interval() {
        let img = make_native_image(ramp(60), [3, 4, 5]);
        let out = ClampImageFilter::new(10.0, 40.0)
            .apply_native(&img, &SequentialBackend)
            .expect("native clamp");
        for (i, &v) in native_vals(&out).iter().enumerate() {
            assert_eq!(v, (i as f32).clamp(10.0, 40.0), "clamp wrong at {i}");
        }
    }
}

// ── Shift-scale ───────────────────────────────────────────────────────────────

mod shift_scale {
    use super::*;

    #[test]
    fn matches_coeus() {
        assert_coeus_matches_coeus(
            ramp(24),
            [2, 3, 4],
            |img| {
                ShiftScaleImageFilter::new(-5.0, 0.5)
                    .apply(img)
                    .expect("burn shift-scale")
            },
            |img, backend| ShiftScaleImageFilter::new(-5.0, 0.5).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_affine() {
        let img = make_native_image(ramp(24), [2, 3, 4]);
        let out = ShiftScaleImageFilter::new(-5.0, 0.5)
            .apply_native(&img, &SequentialBackend)
            .expect("native shift-scale");
        for (i, &v) in native_vals(&out).iter().enumerate() {
            assert_eq!(
                v,
                ((i as f64 - 5.0) * 0.5) as f32,
                "shift-scale wrong at {i}"
            );
        }
    }
}

// ── Mask (two-input) ──────────────────────────────────────────────────────────

mod mask {
    use super::*;

    fn mask_pattern(n: usize) -> Vec<f32> {
        (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect()
    }

    #[test]
    fn matches_coeus_mask() {
        assert_coeus_matches_coeus_pair(
            ramp(24),
            mask_pattern(24),
            [2, 3, 4],
            |img, m| {
                MaskImageFilter::new()
                    .with_outside_value(-1.0)
                    .apply(img, m)
                    .expect("burn mask")
            },
            |img, m, backend| {
                MaskImageFilter::new()
                    .with_outside_value(-1.0)
                    .apply_native(img, m, backend)
            },
        );
    }

    #[test]
    fn matches_coeus_mask_negated() {
        assert_coeus_matches_coeus_pair(
            ramp(24),
            mask_pattern(24),
            [2, 3, 4],
            |img, m| {
                MaskNegatedImageFilter::new()
                    .with_outside_value(-1.0)
                    .apply(img, m)
                    .expect("burn mask negated")
            },
            |img, m, backend| {
                MaskNegatedImageFilter::new()
                    .with_outside_value(-1.0)
                    .apply_native(img, m, backend)
            },
        );
    }

    #[test]
    fn matches_coeus_masked_assign() {
        assert_coeus_matches_coeus_pair(
            ramp(24),
            mask_pattern(24),
            [2, 3, 4],
            |img, m| {
                MaskedAssignImageFilter::new(99.0)
                    .apply(img, m)
                    .expect("burn masked assign")
            },
            |img, m, backend| MaskedAssignImageFilter::new(99.0).apply_native(img, m, backend),
        );
    }

    #[test]
    fn oracle_indicator() {
        let dims = [2, 3, 4];
        let img = make_native_image(ramp(24), dims);
        let m = make_native_image(mask_pattern(24), dims);
        let out = MaskImageFilter::new()
            .with_outside_value(-1.0)
            .apply_native(&img, &m, &SequentialBackend)
            .expect("native mask");
        for (i, &v) in native_vals(&out).iter().enumerate() {
            let expected = if i % 2 == 0 { i as f32 } else { -1.0 };
            assert_eq!(v, expected, "mask indicator wrong at {i}");
        }
    }
}

// ── Zero-crossing ─────────────────────────────────────────────────────────────

mod zero_crossing {
    use super::*;

    #[test]
    fn matches_coeus() {
        // A sign-changing field so crossings actually occur.
        let vals: Vec<f32> = (0..60).map(|i| i as f32 - 29.5).collect();
        assert_coeus_matches_coeus(
            vals,
            [3, 4, 5],
            |img| {
                ZeroCrossingImageFilter::new()
                    .apply(img)
                    .expect("burn zero-crossing")
            },
            |img, backend| ZeroCrossingImageFilter::new().apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_constant_has_no_crossings() {
        // A constant (non-zero) field can never sign-change: output all background.
        let img = make_native_image(vec![3.0f32; 27], [3, 3, 3]);
        let out = ZeroCrossingImageFilter::new()
            .apply_native(&img, &SequentialBackend)
            .expect("native zero-crossing");
        for &v in &native_vals(&out) {
            assert_eq!(v, 0.0, "constant field must yield no zero crossings");
        }
    }
}

// ── Histogram equalization ────────────────────────────────────────────────────

mod equalization {
    use super::*;

    #[test]
    fn matches_coeus() {
        let vals: Vec<f32> = (0..64).map(|i| (i * 3 % 17) as f32).collect();
        assert_coeus_matches_coeus(
            vals,
            [4, 4, 4],
            |img| {
                HistogramEqualizationFilter::new(32)
                    .apply(img)
                    .expect("burn equalize")
            },
            |img, backend| HistogramEqualizationFilter::new(32).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_uniform_stays_uniform() {
        // A single-value field has span 0 → identity (module invariant).
        let img = make_native_image(vec![7.0f32; 64], [4, 4, 4]);
        let out = HistogramEqualizationFilter::new(32)
            .apply_native(&img, &SequentialBackend)
            .expect("native equalize");
        for &v in &native_vals(&out) {
            assert_eq!(v, 7.0, "uniform field must be unchanged");
        }
    }
}

// ── Adaptive (Stark) equalization ─────────────────────────────────────────────

mod adaptive_equalization {
    use super::*;

    #[test]
    fn matches_coeus() {
        let vals: Vec<f32> = (0..60).map(|i| ((i * 7) % 13) as f32).collect();
        assert_coeus_matches_coeus(
            vals,
            [3, 4, 5],
            |img| {
                AdaptiveHistogramEqualizationFilter::new([1, 1, 1])
                    .apply(img)
                    .expect("burn adaptive")
            },
            |img, backend| {
                AdaptiveHistogramEqualizationFilter::new([1, 1, 1]).apply_native(img, backend)
            },
        );
    }

    #[test]
    fn oracle_constant_is_identity() {
        let img = make_native_image(vec![5.0f32; 27], [3, 3, 3]);
        let out = AdaptiveHistogramEqualizationFilter::new([1, 1, 1])
            .apply_native(&img, &SequentialBackend)
            .expect("native adaptive");
        for &v in &native_vals(&out) {
            assert_eq!(v, 5.0, "constant field must be identity");
        }
    }
}

// ── Bed separation ────────────────────────────────────────────────────────────

mod bed_separation {
    use super::*;

    fn ct_config() -> BedSeparationConfig {
        BedSeparationConfig {
            body_threshold: -350.0,
            closing_radius: 1,
            opening_radius: 1,
            outside_value: -2048.0,
            component_policy: ComponentPolicy::LargestOnly,
            ..Default::default()
        }
    }

    #[test]
    fn matches_coeus() {
        // Small CT-like volume: bright body block, dark background.
        let dims = [3usize, 4, 4];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n)
            .map(|i| if (5..=30).contains(&i) { 50.0 } else { -1000.0 })
            .collect();
        let cfg = ct_config();
        assert_coeus_matches_coeus(
            vals,
            dims,
            move |img| BedSeparationFilter::new(cfg).apply(img).expect("burn bed"),
            move |img, backend| BedSeparationFilter::new(cfg).apply_native(img, backend),
        );
    }

    #[test]
    fn oracle_background_masked_out() {
        let dims = [3usize, 4, 4];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n)
            .map(|i| if (5..=30).contains(&i) { 50.0 } else { -1000.0 })
            .collect();
        let mut cfg = ct_config();
        cfg.closing_radius = 0;
        cfg.opening_radius = 0;
        let img = make_native_image(vals, dims);
        let out = BedSeparationFilter::new(cfg)
            .apply_native(&img, &SequentialBackend)
            .expect("native bed");
        let v = native_vals(&out);
        // Corner voxel 0 (< threshold, isolated) must be the outside value.
        assert_eq!(
            v[0], -2048.0,
            "background voxel must be masked to outside_value"
        );
    }
}
