//! Differential + analytical coverage for the Coeus-native intensity paths.
//!
//! Each native wrapper must be value-identical to the Burn filter it mirrors —
//! both call the identical substrate-agnostic host core (shared harness in
//! `native_support::assert_native_matches_burn`). A handful of analytical
//! oracles pin the mathematical contract independent of the Burn reference.

use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
use coeus_core::SequentialBackend;

use super::bed_separation::{BedSeparationConfig, BedSeparationFilter, ComponentPolicy};
use super::binary_threshold::BinaryThresholdImageFilter;
use super::rescale::RescaleIntensityFilter;
use super::sigmoid::SigmoidImageFilter;
use super::threshold::ThresholdImageFilter;
use super::windowing::IntensityWindowingFilter;

fn ramp(n: usize) -> Vec<f32> {
    (0..n).map(|i| i as f32).collect()
}

// ── Rescale ───────────────────────────────────────────────────────────────────

mod rescale {
    use super::*;

    fn check(vals: Vec<f32>, dims: [usize; 3], out_min: f32, out_max: f32) {
        assert_native_matches_burn(
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
    fn matches_burn_ramp() {
        check(ramp(60), [3, 4, 5], 0.0, 1.0);
    }

    #[test]
    fn matches_burn_arbitrary_range() {
        check(
            vec![-3.0, 2.0, 7.5, -1.0, 4.0, 9.0, 0.0, 5.0],
            [2, 2, 2],
            -2.0,
            8.0,
        );
    }

    #[test]
    fn matches_burn_constant_field() {
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
        assert_native_matches_burn(
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
    fn matches_burn_interior_and_saturation() {
        check(ramp(60), [3, 4, 5], 20.0, 40.0, 0.0, 1.0);
    }

    #[test]
    fn matches_burn_degenerate_window() {
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
        assert_native_matches_burn(
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
    fn matches_burn_below() {
        check_below(ramp(24), [2, 3, 4], 10.0, -1.0);
    }

    #[test]
    fn matches_burn_above() {
        let vals = ramp(24);
        assert_native_matches_burn(
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
    fn matches_burn_outside() {
        let vals = ramp(24);
        assert_native_matches_burn(
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
    fn matches_burn() {
        let vals = ramp(60);
        assert_native_matches_burn(
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
    fn matches_burn_step_degenerate_beta() {
        let vals = ramp(8);
        assert_native_matches_burn(
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
    fn matches_burn() {
        let vals = ramp(24);
        assert_native_matches_burn(
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
    fn matches_burn() {
        // Small CT-like volume: bright body block, dark background.
        let dims = [3usize, 4, 4];
        let n = dims[0] * dims[1] * dims[2];
        let vals: Vec<f32> = (0..n)
            .map(|i| if (5..=30).contains(&i) { 50.0 } else { -1000.0 })
            .collect();
        let cfg = ct_config();
        assert_native_matches_burn(
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
