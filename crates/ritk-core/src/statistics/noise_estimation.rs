//! MAD-based noise estimation for medical images.
//!
//! # Mathematical Specification
//!
//! The Median Absolute Deviation (MAD) provides a robust estimator of the
//! standard deviation of additive Gaussian noise:
//!
//!   σ̂ = 1.4826 · median(|Xᵢ − median(X)|)
//!
//! ## Derivation of the 1.4826 constant
//!
//! For a normal distribution N(μ, σ²), the MAD satisfies:
//!
//!   MAD = median(|X − median(X)|) = σ · Φ⁻¹(3/4)
//!
//! where Φ⁻¹ is the quantile function (inverse CDF) of the standard normal.
//! Φ⁻¹(3/4) ≈ 0.6744897501960817. Therefore, to recover σ:
//!
//!   σ = MAD / Φ⁻¹(3/4) = MAD · (1 / 0.6744897501960817) ≈ MAD · 1.4826
//!
//! This makes the MAD a consistent estimator of σ under Gaussian noise.
//!
//! ## Complexity
//!
//! O(n log n) dominated by sorting the voxel values and the absolute deviations.
//!
//! ## References
//!
//! - Hampel, F. R. (1974). The influence curve and its role in robust estimation.
//!   *J. Amer. Statist. Assoc.*, 69(346), 383–393.
//! - Rousseeuw, P. J. & Croux, C. (1993). Alternatives to the Median Absolute
//!   Deviation. *J. Amer. Statist. Assoc.*, 88(424), 1273–1283.

use crate::image::Image;
use burn::tensor::backend::Backend;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Consistency constant for Gaussian noise: 1 / Φ⁻¹(3/4) ≈ 1.4826.
///
/// Φ⁻¹(3/4) = 0.6744897501960817, so 1/0.6744897501960817 = 1.4826022185056018.
const MAD_CONSISTENCY_CONSTANT: f32 = 1.4826;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Compute the median of a sorted, non-empty slice.
///
/// For even-length slices, returns the average of the two middle elements.
/// For odd-length slices, returns the middle element.
///
/// # Precondition
/// `sorted` must be sorted in non-decreasing order and non-empty.
#[inline]
fn median_sorted(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    debug_assert!(n > 0, "median_sorted requires non-empty input");
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Sort a mutable slice of f32 values, treating NaN as greater than all finite values.
#[inline]
fn sort_f32(values: &mut [f32]) {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
}

/// Compute σ̂ = 1.4826 · median(|Xᵢ − median(X)|) from a mutable slice of values.
///
/// Returns 0.0 for empty or single-element inputs and for constant-valued inputs.
fn mad_sigma(values: &mut Vec<f32>) -> f32 {
    let n = values.len();
    if n <= 1 {
        return 0.0;
    }

    sort_f32(values);
    let med = median_sorted(values);

    // Compute absolute deviations from the median.
    let mut abs_devs: Vec<f32> = values.iter().map(|&x| (x - med).abs()).collect();
    sort_f32(&mut abs_devs);

    let mad = median_sorted(&abs_devs);

    MAD_CONSISTENCY_CONSTANT * mad
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Estimate the standard deviation of additive Gaussian noise in `image` using
/// the Median Absolute Deviation (MAD).
///
/// # Formula
///
/// ```text
/// σ̂ = 1.4826 · median(|Xᵢ − median(X)|)
/// ```
///
/// # Arguments
/// * `image` – Input image whose noise level is to be estimated.
///
/// # Returns
/// Estimated noise standard deviation (σ̂). Returns 0.0 for constant images.
///
/// # Complexity
/// O(n log n) where n is the total number of voxels.
pub fn estimate_noise_mad<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    let data = image.data().clone().into_data();
    let slice = data.as_slice::<f32>().expect("f32 image tensor data");
    let mut values: Vec<f32> = slice.to_vec();
    mad_sigma(&mut values)
}

/// Estimate the standard deviation of additive Gaussian noise in `image` using
/// the MAD, restricted to voxels where `mask` > 0.5.
///
/// # Formula
///
/// ```text
/// σ̂ = 1.4826 · median(|Xᵢ − median(X)|)
/// ```
///
/// computed only over the set {i : mask(i) > 0.5}.
///
/// # Arguments
/// * `image` – Input image whose noise level is to be estimated.
/// * `mask`  – Binary mask selecting voxels to include (values > 0.5 are foreground).
///
/// # Returns
/// Estimated noise standard deviation (σ̂). Returns 0.0 if no foreground voxels
/// exist or if all foreground voxels have identical intensity.
///
/// # Panics
/// Panics if `image` and `mask` have different element counts.
///
/// # Complexity
/// O(n log n) where n is the number of foreground voxels.
pub fn estimate_noise_mad_masked<B: Backend, const D: usize>(
    image: &Image<B, D>,
    mask: &Image<B, D>,
) -> f32 {
    let img_data = image.data().clone().into_data();
    let img_slice = img_data.as_slice::<f32>().expect("f32 image tensor data");

    let mask_data = mask.data().clone().into_data();
    let mask_slice = mask_data.as_slice::<f32>().expect("f32 mask tensor data");

    assert_eq!(
        img_slice.len(),
        mask_slice.len(),
        "image and mask must have identical element count"
    );

    let mut values: Vec<f32> = img_slice
        .iter()
        .zip(mask_slice.iter())
        .filter(|(_, &m)| m > 0.5)
        .map(|(&v, _)| v)
        .collect();

    mad_sigma(&mut values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::{Direction, Point, Spacing};
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_image_1d(data: Vec<f32>) -> Image<TestBackend, 1> {
        let n = data.len();
        let device = Default::default();
        let tensor =
            Tensor::<TestBackend, 1>::from_data(TensorData::new(data, Shape::new([n])), &device);
        Image::new(
            tensor,
            Point::new([0.0]),
            Spacing::new([1.0]),
            Direction::identity(),
        )
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
}
