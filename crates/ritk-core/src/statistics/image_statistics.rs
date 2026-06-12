//! Image intensity statistics.
//!
//! Computes descriptive statistics over image intensities, either for the
//! full image or restricted to a foreground mask.
//!
//! # Mathematical Specification
//! Given a vector of intensity values V = {v₁, …, vₙ}:
//! - min     = min(V)
//! - max     = max(V)
//! - mean    = (1/n) · Σ vᵢ
//! - std     = √( (1/n) · Σ (vᵢ − mean)² )   (population std)
//! - p25     = V_sorted[⌊n/4⌋]
//! - p50     = V_sorted[⌊n/2⌋]
//! - p75     = V_sorted[⌊3n/4⌋]

use crate::filter::ops::extract_vec_infallible;
use ritk_image::Image;
use burn::tensor::backend::Backend;

/// Descriptive statistics over image intensities.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageStatistics {
    /// Minimum intensity value.
    pub min: f32,
    /// Maximum intensity value.
    pub max: f32,
    /// Arithmetic mean intensity.
    pub mean: f32,
    /// Population standard deviation.
    pub std: f32,
    /// Percentiles: \[p25, p50, p75\].
    pub percentiles: [f32; 3],
}

/// Compute statistics over **all** voxels in `image`.
///
/// Extraction path: `tensor.clone().into_data()` → `as_slice::<f32>()` → CPU arithmetic.
pub fn compute_statistics<B: Backend, const D: usize>(image: &Image<B, D>) -> ImageStatistics {
    let (vals, _) = extract_vec_infallible(image);
    let slice: &[f32] = &vals;
    compute_statistics_from_slice(slice)
}

/// Compute statistics from an immutable slice.
///
/// This is the zero-domain-logic public helper for callers that already have
/// borrowed f32 tensor storage. The sorted copy required for percentile
/// computation is allocated once inside `compute_from_values`.
pub fn compute_statistics_from_slice(slice: &[f32]) -> ImageStatistics {
    // Delegate directly; compute_from_values allocates a sorted copy internally.
    compute_from_values(slice)
}

/// Compute statistics restricted to voxels where `mask` > 0.5 (foreground).
///
/// `mask` must have the same shape as `image` and contain 0.0 (background) or
/// 1.0 (foreground). Panics if the shapes differ or no foreground voxels exist.
pub fn masked_statistics<B: Backend, const D: usize>(
    image: &Image<B, D>,
    mask: &Image<B, D>,
) -> ImageStatistics {
    let (img_vals, _) = extract_vec_infallible(image);
    let image_slice: &[f32] = &img_vals;
    let (mask_vals, _) = extract_vec_infallible(mask);
    let mask_slice: &[f32] = &mask_vals;

    assert_eq!(
        image_slice.len(),
        mask_slice.len(),
        "image and mask must have identical element count"
    );

    let values: Vec<f32> = image_slice
        .iter()
        .zip(mask_slice.iter())
        .filter(|(_, &m)| m > 0.5)
        .map(|(&v, _)| v)
        .collect();

    assert!(!values.is_empty(), "mask contains no foreground voxels");
    compute_from_values(&values)
}

/// Core statistics computation.
///
/// # Invariants
/// - `values` is non-empty (caller enforced).
/// - Sorts `values` in-place; NaN propagates arithmetic and is ordered last by
///   the `partial_cmp` fallback.
///
/// # Precision
/// Mean and variance accumulate in f64 to avoid catastrophic f32 cancellation
/// for large arrays (n > ~10^7).  Sequential f32 summation of n ≈ 10^8 values
/// with mean ≈ −789 produces a running sum of ~−85 billion; at that scale the
/// f32 ULP (≈8192) exceeds individual element magnitudes, so additions are
/// rounded to zero and the sum saturates.  Two-pass f64 accumulation is the
/// algorithm's numerical contract requirement, not a convenience cast.
pub fn compute_from_values(values: &[f32]) -> ImageStatistics {
    let mut sorted_values = values.to_vec();
    let values = sorted_values.as_mut_slice();
    let n = values.len();
    debug_assert!(n > 0, "compute_from_values requires non-empty input");

    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = values[0];
    let max = values[n - 1];

    // Accumulate in f64: sequential f32 sum saturates for n > ~10^7 elements
    // when the running total exceeds the f32 ULP of individual values.
    let sum_wide: f64 = values.iter().map(|&v| v as f64).sum::<f64>();
    let mean_wide: f64 = sum_wide / n as f64;
    let mean: f32 = mean_wide as f32;

    // Two-pass f64 variance: squared deviations sum to ~10^13 for CT-scale
    // data, exceeding f32 representable precision per element at n > ~10^7.
    let var_wide: f64 = values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean_wide;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    let std = var_wide.sqrt() as f32;

    // Floor-division percentile indices as specified in the module contract.
    let p25 = values[n / 4];
    let p50 = values[n / 2];
    let p75 = values[(3 * n) / 4];

    ImageStatistics {
        min,
        max,
        mean,
        std,
        percentiles: [p25, p50, p75],
    }
}

#[cfg(test)]
#[path = "tests_image_statistics.rs"]
mod tests;
