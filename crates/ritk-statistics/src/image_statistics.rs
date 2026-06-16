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
//! - std     = √( (1/(n − ddof)) · Σ (vᵢ − mean)² )
//!   `ddof` is the numpy-style delta degrees of freedom: 0 → population
//!   std (the default), 1 → sample std (Bessel-corrected, matching
//!   ITK/SimpleITK `StatisticsImageFilter`/`LabelStatisticsImageFilter`).
//! - p25     = V_sorted[⌊n/4⌋]
//! - p50     = V_sorted[⌊n/2⌋]
//! - p75     = V_sorted[⌊3n/4⌋]

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// Descriptive statistics over image intensities.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageStatistics {
    /// Minimum intensity value.
    pub min: f32,
    /// Maximum intensity value.
    pub max: f32,
    /// Arithmetic mean intensity.
    pub mean: f32,
    /// Standard deviation with the requested `ddof` (0 = population, 1 = sample).
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
    compute_statistics_from_slice(slice, 0)
}

/// Compute statistics from an immutable slice.
///
/// This is the zero-domain-logic public helper for callers that already have
/// borrowed f32 tensor storage. The sorted copy required for percentile
/// computation is allocated once inside `compute_from_values`.
pub fn compute_statistics_from_slice(slice: &[f32], ddof: usize) -> ImageStatistics {
    // Delegate directly; compute_from_values allocates a sorted copy internally.
    compute_from_values(slice, ddof)
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
        .filter(|(_, &m)| m > crate::FOREGROUND_THRESHOLD)
        .map(|(&v, _)| v)
        .collect();

    assert!(!values.is_empty(), "mask contains no foreground voxels");
    compute_from_values(&values, 0)
}

/// Core statistics computation.
///
/// # Invariants
/// - `values` is non-empty (caller enforced).
/// - Copies `values` and partially reorders the copy in place; NaN compares
///   `Equal` under the `partial_cmp` fallback and is ignored by min/max.
///
/// # Algorithm
/// The three percentiles are the order statistics at floor-division ranks
/// `n/4`, `n/2`, `3n/4`. Computing them with a full sort is `O(n log n)` and
/// dominates the cost; instead each rank is isolated with `select_nth_unstable`
/// (quickselect, `O(n)` average). The selections run on progressively smaller
/// suffixes — after rank `k` is placed, every element before `k` is `≤` it, so
/// the next (larger) rank is sought only in `values[k+1..]` — giving `≈2.25n`
/// comparisons total versus `n log n` for the sort. Min, max, and the f64 sum
/// are gathered in a single fused pass before any reordering.
///
/// # Precision
/// Mean and variance accumulate in f64 to avoid catastrophic f32 cancellation
/// for large arrays (n > ~10^7).  Sequential f32 summation of n ≈ 10^8 values
/// with mean ≈ −789 produces a running sum of ~−85 billion; at that scale the
/// f32 ULP (≈8192) exceeds individual element magnitudes, so additions are
/// rounded to zero and the sum saturates.  Two-pass f64 accumulation is the
/// algorithm's numerical contract requirement, not a convenience cast.
pub fn compute_from_values(values: &[f32], ddof: usize) -> ImageStatistics {
    let mut buffer = values.to_vec();
    let values = buffer.as_mut_slice();
    let n = values.len();
    debug_assert!(n > 0, "compute_from_values requires non-empty input");

    // Fused pass: min, max, and the f64 sum. NaN compares `Equal` (the
    // `partial_cmp` fallback) so it never displaces a real extremum, matching
    // the prior sorted-array semantics; it still propagates into the f64 sum.
    let mut min = values[0];
    let mut max = values[0];
    let mut sum_wide = 0.0_f64;
    for &v in values.iter() {
        sum_wide += v as f64;
        if matches!(v.partial_cmp(&min), Some(std::cmp::Ordering::Less)) {
            min = v;
        }
        if matches!(v.partial_cmp(&max), Some(std::cmp::Ordering::Greater)) {
            max = v;
        }
    }

    let mean_wide: f64 = sum_wide / n as f64;
    let mean: f32 = mean_wide as f32;

    // Two-pass f64 variance: squared deviations sum to ~10^13 for CT-scale
    // data, exceeding f32 representable precision per element at n > ~10^7.
    let sum_sq_dev: f64 = values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean_wide;
            d * d
        })
        .sum::<f64>();
    // numpy-style ddof: divisor = n − ddof. n ≤ ddof (e.g. sample std of a single
    // voxel) yields a degenerate 0.0 rather than a NaN/∞.
    let denom = n.saturating_sub(ddof);
    let std = if denom == 0 {
        0.0
    } else {
        (sum_sq_dev / denom as f64).sqrt() as f32
    };

    // Floor-division percentile ranks (module contract). Quickselect each rank
    // on the suffix left of the previous one — O(n) average, exact order
    // statistic, no full sort.
    let cmp = |a: &f32, b: &f32| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal);
    let ranks = [n / 4, n / 2, (3 * n) / 4];
    let mut percentiles = [0.0_f32; 3];
    let mut lo = 0usize;
    for (slot, &rank) in percentiles.iter_mut().zip(ranks.iter()) {
        if rank >= lo {
            // Elements in `values[..lo]` are already ≤ everything in
            // `values[lo..]`, so the (rank − lo)-th smallest of the suffix is
            // the rank-th smallest overall.
            values[lo..].select_nth_unstable_by(rank - lo, cmp);
            lo = rank + 1;
        }
        *slot = values[rank];
    }

    ImageStatistics {
        min,
        max,
        mean,
        std,
        percentiles,
    }
}

#[cfg(test)]
#[path = "tests_image_statistics.rs"]
mod tests;
