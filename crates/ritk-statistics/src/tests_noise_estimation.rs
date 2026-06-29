use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support;
use ritk_image::Image;

type TestBackend = NdArray<f32>;

fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
    test_support::make_image_1d(data)
}

// ── Positive tests ────────────────────────────────────────────────────────

#[test]
fn test_mad_gaussian_noise_estimate_within_tolerance() {
    // Deterministic pseudo-Gaussian noise via Box-Muller transform.
    // Use a simple LCG for reproducibility:
    //   x_{n+1} = (a * x_n + c) mod m
    // Parameters from Numerical Recipes.
    let n = 10_000usize;
    let true_sigma: f32 = 5.0;
    let true_mean: f32 = 100.0;

    let mut rng_state: u64 = 42;
    let a: u64 = 6_364_136_223_846_793_005;
    let c: u64 = 1_442_695_040_888_963_407;

    let mut uniform = || -> f64 {
        rng_state = rng_state.wrapping_mul(a).wrapping_add(c);
        // Map to (0, 1) — avoid exact 0.
        ((rng_state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };

    // Box-Muller: generate pairs of standard normals.
    let mut samples: Vec<f32> = Vec::with_capacity(n);
    while samples.len() < n {
        let u1 = uniform();
        let u2 = uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        let z0 = r * theta.cos();
        let z1 = r * theta.sin();
        samples.push((true_mean as f64 + true_sigma as f64 * z0) as f32);
        if samples.len() < n {
            samples.push((true_mean as f64 + true_sigma as f64 * z1) as f32);
        }
    }
    samples.truncate(n);

    let image = make_image_1d(samples);
    let estimated = estimate_noise_mad(&image);

    // MAD estimator should be within 20% of the true σ for 10k samples.
    let relative_error = ((estimated - true_sigma) / true_sigma).abs();
    assert!(
        relative_error < 0.20,
        "estimated σ = {}, true σ = {}, relative error = {} (> 20%)",
        estimated,
        true_sigma,
        relative_error
    );
}

#[test]
fn test_mad_constant_image_returns_zero() {
    // Constant image: all deviations are 0 → MAD = 0 → σ̂ = 0.
    let data = vec![7.0f32; 100];
    let image = make_image_1d(data);
    let estimated = estimate_noise_mad(&image);
    assert!(
        estimated.abs() < 1e-10,
        "constant image must yield σ̂ = 0.0, got {}",
        estimated
    );
}

#[test]
fn test_mad_masked_agrees_with_unmasked_for_all_ones_mask() {
    // When mask is all ones, the masked variant must produce the same
    // result as the unmasked variant.
    let data: Vec<f32> = (0..200).map(|i| (i as f32) * 0.3 - 30.0).collect();
    let n = data.len();
    let mask_data = vec![1.0f32; n];

    let image = make_image_1d(data);
    let mask = make_image_1d(mask_data);

    let sigma_unmasked = estimate_noise_mad(&image);
    let sigma_masked = estimate_noise_mad_masked(&image, &mask);

    assert!(
        (sigma_unmasked - sigma_masked).abs() < 1e-6,
        "all-ones mask: unmasked σ̂ = {}, masked σ̂ = {}",
        sigma_unmasked,
        sigma_masked
    );
}

// ── Boundary / edge cases ─────────────────────────────────────────────────

#[test]
fn test_mad_single_voxel_returns_zero() {
    let image = make_image_1d(vec![42.0]);
    let estimated = estimate_noise_mad(&image);
    assert!(
        estimated.abs() < 1e-10,
        "single voxel → σ̂ = 0.0, got {}",
        estimated
    );
}

#[test]
fn test_mad_two_voxels_known_value() {
    // Values [0, 10]: median = 5, |0-5| = 5, |10-5| = 5.
    // median(abs_devs) = 5. σ̂ = 1.4826 * 5 = 7.413.
    let image = make_image_1d(vec![0.0, 10.0]);
    let estimated = estimate_noise_mad(&image);
    let expected = 1.4826 * 5.0;
    assert!(
        (estimated - expected).abs() < 1e-3,
        "expected {}, got {}",
        expected,
        estimated
    );
}

#[test]
fn test_mad_from_slice_preserves_caller_order() {
    let data = vec![9.0, 1.0, 7.0, 3.0, 5.0];
    let before = data.clone();

    let estimated = estimate_noise_mad_from_slice(&data);

    assert_eq!(
        data, before,
        "borrowed-slice MAD API must not mutate caller-owned data"
    );
    assert!(
        (estimated - 2.9652).abs() < 1e-4,
        "expected MAD sigma 2.9652 for symmetric odd sample, got {}",
        estimated
    );
}

#[test]
fn test_mad_masked_empty_foreground_returns_zero() {
    // No foreground voxels → 0.0 (graceful degradation, not panic).
    let image = make_image_1d(vec![1.0, 2.0, 3.0]);
    let mask = make_image_1d(vec![0.0, 0.0, 0.0]);
    let estimated = estimate_noise_mad_masked(&image, &mask);
    assert!(
        estimated.abs() < 1e-10,
        "empty mask must yield σ̂ = 0.0, got {}",
        estimated
    );
}

#[test]
fn test_mad_masked_subset_selection() {
    // Mask selects only constant voxels → σ̂ = 0.
    // image = [1, 5, 5, 5, 100], mask selects indices 1..=3 (value 5.0 only).
    let image = make_image_1d(vec![1.0, 5.0, 5.0, 5.0, 100.0]);
    let mask = make_image_1d(vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    let estimated = estimate_noise_mad_masked(&image, &mask);
    assert!(
        estimated.abs() < 1e-10,
        "masked constant subset → σ̂ = 0.0, got {}",
        estimated
    );
}

// ── Negative tests ────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "identical element count")]
fn test_mad_masked_shape_mismatch_panics() {
    let image = make_image_1d(vec![1.0, 2.0, 3.0]);
    let mask = make_image_1d(vec![1.0, 1.0]);
    let _ = estimate_noise_mad_masked(&image, &mask);
}

// ── Helper unit tests ─────────────────────────────────────────────────────

#[test]
fn test_median_sorted_odd() {
    // [1, 3, 5] → median = 3
    assert!((median_sorted(&[1.0, 3.0, 5.0]) - 3.0).abs() < 1e-10);
}

#[test]
fn test_median_sorted_even() {
    // [1, 3, 5, 7] → median = (3 + 5) / 2 = 4
    assert!((median_sorted(&[1.0, 3.0, 5.0, 7.0]) - 4.0).abs() < 1e-10);
}

#[test]
fn test_median_sorted_single() {
    assert!((median_sorted(&[42.0]) - 42.0).abs() < 1e-10);
}
