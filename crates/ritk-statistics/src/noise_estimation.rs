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
//! The implementation reuses one mutable work buffer for both sorts.
//!
//! ## References
//!
//! - Hampel, F. R. (1974). The influence curve and its role in robust estimation.
//!   *J. Amer. Statist. Assoc.*, 69(346), 383–393.
//! - Rousseeuw, P. J. & Croux, C. (1993). Alternatives to the Median Absolute
//!   Deviation. *J. Amer. Statist. Assoc.*, 88(424), 1273–1283.

use ritk_image::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

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

/// Compute σ̂ = 1.4826 · median(|Xᵢ − median(X)|) from a mutable slice of values.
///
/// Returns 0.0 for empty or single-element inputs and for constant-valued inputs.
fn mad_sigma(values: &mut [f32]) -> f32 {
    let n = values.len();
    if n <= 1 {
        return 0.0;
    }

    crate::sort_floats(values);
    let med = median_sorted(values);

    for value in values.iter_mut() {
        *value = (*value - med).abs();
    }
    crate::sort_floats(values);

    let mad = median_sorted(values);

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
    let (vals, _) = extract_vec_infallible(image);
    let mut values = vals;
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
    let (img_vals, _) = extract_vec_infallible(image);
    let img_slice: &[f32] = &img_vals;
    let (mask_vals, _) = extract_vec_infallible(mask);
    let mask_slice: &[f32] = &mask_vals;

    assert_eq!(
        img_slice.len(),
        mask_slice.len(),
        "image and mask must have identical element count"
    );

    let mut values: Vec<f32> = img_slice
        .iter()
        .zip(mask_slice.iter())
        .filter(|(_, &m)| m > crate::FOREGROUND_THRESHOLD)
        .map(|(&v, _)| v)
        .collect();

    mad_sigma(&mut values)
}

/// Estimate Gaussian noise sigma-hat using MAD directly from a flat `&[f32]` slice.
///
/// Equivalent to [`estimate_noise_mad`] but accepts pre-extracted slice data,
/// enabling zero-copy extraction from the NdArray backend (e.g., via
/// `ArcArray::as_slice_memory_order`).
///
/// # Formula
/// sigma-hat = 1.4826 * median(|Xi - median(X)|)
///
/// # Returns
/// Estimated noise standard deviation. Returns 0.0 for empty, single-element,
/// or constant inputs.
pub fn estimate_noise_mad_from_slice(slice: &[f32]) -> f32 {
    let mut values: Vec<f32> = slice.to_vec();
    mad_sigma(&mut values)
}

/// Estimate Gaussian noise sigma-hat using MAD from pre-extracted image and mask slices.
///
/// Equivalent to [`estimate_noise_mad_masked`] but accepts pre-extracted slice data,
/// enabling zero-copy extraction from the NdArray backend.
///
/// Only voxels where `mask_slice[i] > 0.5` are included in the estimate.
///
/// # Formula
/// sigma-hat = 1.4826 * median(|Xi - median(X)|)
/// computed over the foreground set {i : mask_slice\[i\] > 0.5}.
///
/// # Returns
/// Estimated noise standard deviation over foreground voxels. Returns 0.0 if the
/// foreground set is empty or constant.
///
/// # Panics
/// Panics if `img_slice.len() != mask_slice.len()`.
pub fn estimate_noise_mad_masked_from_slices(img_slice: &[f32], mask_slice: &[f32]) -> f32 {
    assert_eq!(
        img_slice.len(),
        mask_slice.len(),
        "image and mask must have identical element count"
    );
    let mut values: Vec<f32> = img_slice
        .iter()
        .zip(mask_slice.iter())
        .filter(|(_, &m)| m > crate::FOREGROUND_THRESHOLD)
        .map(|(&v, _)| v)
        .collect();
    mad_sigma(&mut values)
}

#[cfg(test)]
#[path = "tests_noise_estimation.rs"]
mod tests;
