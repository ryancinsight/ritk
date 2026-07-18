//! Coeus-native normalization paths: analytical oracles plus differential parity
//! against the sequential Coeus path.
//!
//! ## Tolerances
//! - `HistogramMatcher`, `NyulUdupaNormalizer`, `WhiteStripeNormalizer`: the
//!   native and sequential adapters delegate to the **same** host core
//!   (`transform_values` / `compute_white_stripe`), so their outputs are bitwise
//!   identical — asserted with `PARITY = 1e-6` (allowing only for tensor
//!   round-trip representation, which is exact for `f32`).
//! - `ZScoreNormalizer`, `MinMaxNormalizer`: the sequential path uses tensor
//!   scalar ops and the native path a host `f32` loop, but both execute the same
//!   `f32` operations in the same order on statistics derived from the identical
//!   `f64` two-pass reduction, so the results agree to `PARITY`.
//! - Analytical oracles (`z-score of a ramp → mean 0, var 1`, `min-max known
//!   values`) use tolerances derived from the `f32` epsilon of the closed form.

use super::*;
use coeus_core::{MoiraiBackend, SequentialBackend};
use ritk_image::native::Image as NativeImage;
use ritk_image::test_support;
use ritk_image::Image as SequentialImage;
use ritk_spatial::{Direction, Point, Spacing};

/// Shared-core paths are bitwise identical; `f32` tensor round-trip is exact.
const PARITY: f32 = 1e-6;

fn native<const D: usize>(data: Vec<f32>, dims: [usize; D]) -> NativeImage<f32, MoiraiBackend, D> {
    NativeImage::from_flat(
        data,
        dims,
        Point::new([0.0_f64; D]),
        Spacing::new([1.0_f64; D]),
        Direction::identity(),
    )
    .expect("native image construction")
}

fn sequential<const D: usize>(
    data: Vec<f32>,
    dims: [usize; D],
) -> SequentialImage<f32, SequentialBackend, D> {
    test_support::make_image(data, dims)
}

fn native_values<const D: usize>(image: &NativeImage<f32, MoiraiBackend, D>) -> Vec<f32> {
    ritk_tensor_ops::native::extract_image_slice(image)
        .expect("native slice")
        .0
        .to_vec()
}

fn sequential_values<const D: usize>(
    image: &SequentialImage<f32, SequentialBackend, D>,
) -> Vec<f32> {
    ritk_tensor_ops::extract_vec_infallible(image).0
}

fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() <= tol, "{label}[{i}]: {x} vs {y}");
    }
}

// ── Z-score ──────────────────────────────────────────────────────────────────

#[test]
fn zscore_ramp_has_zero_mean_unit_variance() {
    // Ramp [1..5]: mean 3, population var 2. After (x-3)/√2 the output has
    // mean 0 and population variance 1 (up to the ε in the denominator).
    let img = native(vec![1.0, 2.0, 3.0, 4.0, 5.0], [5]);
    let out = ZScoreNormalizer::new().normalize_native(&img).unwrap();
    let stats = crate::image_statistics::native::compute_statistics(&out).unwrap();
    assert!(stats.mean.abs() < 1e-5, "mean ≈ 0, got {}", stats.mean);
    assert!((stats.std - 1.0).abs() < 1e-3, "std ≈ 1, got {}", stats.std);
}

#[test]
fn zscore_matches_sequential() {
    let data = vec![2.0, -1.0, 4.0, 7.0, 0.5, 3.3];
    let nb = ZScoreNormalizer::new()
        .normalize_native(&native(data.clone(), [6]))
        .unwrap();
    let sequential_result = ZScoreNormalizer::new().normalize(&sequential(data, [6]));
    assert_close(
        &native_values(&nb),
        &sequential_values(&sequential_result),
        PARITY,
        "zscore",
    );
}

#[test]
fn zscore_masked_matches_sequential() {
    let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
    let mask = vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0];
    let nb = ZScoreNormalizer::new()
        .normalize_masked_native(&native(data.clone(), [6]), &native(mask.clone(), [6]))
        .unwrap();
    let sequential_result =
        ZScoreNormalizer::new().normalize_masked(&sequential(data, [6]), &sequential(mask, [6]));
    assert_close(
        &native_values(&nb),
        &sequential_values(&sequential_result),
        PARITY,
        "zscore_masked",
    );
}

// ── Min-max ──────────────────────────────────────────────────────────────────

#[test]
fn minmax_known_unit_range() {
    let out = MinMaxNormalizer::new()
        .normalize_native(&native(vec![0.0, 5.0, 10.0], [3]))
        .unwrap();
    let v = native_values(&out);
    assert!(v[0].abs() < 1e-5, "N(0) ≈ 0, got {}", v[0]);
    assert!((v[1] - 0.5).abs() < 1e-4, "N(5) ≈ 0.5, got {}", v[1]);
    assert!((v[2] - 1.0).abs() < 1e-4, "N(10) ≈ 1, got {}", v[2]);
}

#[test]
fn minmax_custom_range_matches_sequential() {
    let data = vec![-3.0, 1.0, 4.0, 9.0, 2.0];
    let nb = MinMaxNormalizer::with_range(-1.0, 2.0)
        .normalize_native(&native(data.clone(), [5]))
        .unwrap();
    let sequential_result =
        MinMaxNormalizer::with_range(-1.0, 2.0).normalize(&sequential(data, [5]));
    assert_close(
        &native_values(&nb),
        &sequential_values(&sequential_result),
        PARITY,
        "minmax",
    );
}

// ── Histogram matching ───────────────────────────────────────────────────────

#[test]
fn histogram_matching_matches_sequential() {
    let src: Vec<f32> = (0..64).map(|i| i as f32).collect();
    let reference: Vec<f32> = (0..64).map(|i| (i as f32) * 2.0 + 5.0).collect();
    let matcher = HistogramMatcher::new(32);
    let nb = matcher
        .match_histograms_native(&native(src.clone(), [64]), &native(reference.clone(), [64]))
        .unwrap();
    let sequential_result =
        matcher.match_histograms(&sequential(src, [64]), &sequential(reference, [64]));
    assert_close(
        &native_values(&nb),
        &sequential_values(&sequential_result),
        PARITY,
        "histogram",
    );
}

#[test]
fn histogram_matching_constant_source_unchanged() {
    let src = vec![7.0f32; 16];
    let reference: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let out = HistogramMatcher::new(16)
        .match_histograms_native(&native(src.clone(), [16]), &native(reference, [16]))
        .unwrap();
    assert_close(&native_values(&out), &src, PARITY, "constant-source");
}

// ── Nyúl-Udupa ───────────────────────────────────────────────────────────────

#[test]
fn nyul_udupa_matches_sequential() {
    let train_a: Vec<f32> = (0..50).map(|i| i as f32).collect();
    let train_b: Vec<f32> = (0..50).map(|i| (i as f32) * 1.5 + 3.0).collect();
    let target: Vec<f32> = (0..50).map(|i| (i as f32) * 0.8 - 2.0).collect();

    let mut n = NyulUdupaNormalizer::new();
    n.learn_standard_native(&[
        &native(train_a.clone(), [50]),
        &native(train_b.clone(), [50]),
    ])
    .unwrap();
    let nb = n.apply_native(&native(target.clone(), [50])).unwrap();

    let mut sequential_normalizer = NyulUdupaNormalizer::new();
    sequential_normalizer.learn_standard(&[&sequential(train_a, [50]), &sequential(train_b, [50])]);
    let sequential_result = sequential_normalizer
        .apply(&sequential(target, [50]))
        .unwrap();

    assert_close(
        &native_values(&nb),
        &sequential_values(&sequential_result),
        PARITY,
        "nyul",
    );
}

#[test]
fn nyul_udupa_apply_before_learn_errors() {
    let n = NyulUdupaNormalizer::new();
    assert!(n.apply_native(&native(vec![1.0, 2.0, 3.0], [3])).is_err());
}

// ── White stripe ─────────────────────────────────────────────────────────────

#[test]
fn white_stripe_matches_sequential() {
    // Bimodal-ish intensity distribution with a clear upper (WM) mode.
    let mut data = Vec::with_capacity(400);
    for i in 0..400 {
        let x = i as f32;
        data.push(if i < 250 {
            20.0 + (x % 15.0)
        } else {
            90.0 + (x % 8.0)
        });
    }
    let cfg = WhiteStripeConfig::default();
    let nb =
        WhiteStripeNormalizer::normalize_native(&native(data.clone(), [400, 1, 1]), None, &cfg)
            .unwrap();
    let sequential_result =
        WhiteStripeNormalizer::normalize(&sequential(data, [400, 1, 1]), None, &cfg);

    assert!(
        (nb.mu - sequential_result.mu).abs() < 1e-9,
        "mu native={} sequential={}",
        nb.mu,
        sequential_result.mu
    );
    assert!(
        (nb.sigma - sequential_result.sigma).abs() < 1e-9,
        "sigma native={} sequential={}",
        nb.sigma,
        sequential_result.sigma
    );
    assert!(
        (nb.wm_peak - sequential_result.wm_peak).abs() < 1e-9,
        "wm_peak native={} sequential={}",
        nb.wm_peak,
        sequential_result.wm_peak
    );
    assert_eq!(nb.stripe_size, sequential_result.stripe_size, "stripe_size");
    assert_close(
        &native_values(&nb.normalized),
        &sequential_values(&sequential_result.normalized),
        PARITY,
        "white_stripe",
    );
}
