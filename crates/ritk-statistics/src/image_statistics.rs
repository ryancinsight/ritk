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

use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

pub mod native;

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
pub fn compute_statistics<B: Backend, const D: usize>(image: &Image<f32, B, D>) -> ImageStatistics
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (vals, _) = extract_vec_infallible(image);
    compute_from_owned(vals, 0)
}

/// Compute statistics from an immutable slice.
///
/// This is the zero-domain-logic public helper for callers that already have
/// borrowed f32 tensor storage. The sorted copy required for percentile
/// computation is allocated once inside [`compute_from_values`].
pub fn compute_statistics_from_slice(slice: &[f32], ddof: usize) -> ImageStatistics {
    compute_from_values(slice, ddof)
}

/// Compute statistics restricted to voxels where `mask` > 0.5 (foreground).
///
/// `mask` must have the same shape as `image` and contain 0.0 (background) or
/// 1.0 (foreground). Panics if the shapes differ or no foreground voxels exist.
pub fn masked_statistics<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    mask: &Image<f32, B, D>,
) -> ImageStatistics
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
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
    compute_from_owned(values, 0)
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
    compute_from_owned(values.to_vec(), ddof)
}

/// Compute statistics while consuming an owned buffer that may be reordered.
///
/// Masked-statistics paths already allocate this foreground buffer, so this
/// helper avoids cloning it before the in-place percentile selection.
pub(crate) fn compute_from_owned(mut buffer: Vec<f32>, ddof: usize) -> ImageStatistics {
    let values = buffer.as_mut_slice();
    let n = values.len();
    debug_assert!(n > 0, "compute_from_values requires non-empty input");

    // Fused pass: min, max, and the f64 sum in parallel.
    let (min, max, sum_wide) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        n,
        || (values[0], values[0], 0.0_f64),
        |(min_acc, max_acc, sum_acc), i| {
            let v = values[i];
            let new_min = if matches!(v.partial_cmp(&min_acc), Some(std::cmp::Ordering::Less)) {
                v
            } else {
                min_acc
            };
            let new_max = if matches!(v.partial_cmp(&max_acc), Some(std::cmp::Ordering::Greater)) {
                v
            } else {
                max_acc
            };
            (new_min, new_max, sum_acc + v as f64)
        },
        |(amin, amax, asum), (bmin, bmax, bsum)| {
            let rmin = if matches!(bmin.partial_cmp(&amin), Some(std::cmp::Ordering::Less)) {
                bmin
            } else {
                amin
            };
            let rmax = if matches!(bmax.partial_cmp(&amax), Some(std::cmp::Ordering::Greater)) {
                bmax
            } else {
                amax
            };
            (rmin, rmax, asum + bsum)
        },
    );

    let mean_wide: f64 = sum_wide / n as f64;
    let mean: f32 = mean_wide as f32;

    // Two-pass f64 variance in parallel.
    let sum_sq_dev: f64 = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        n,
        || 0.0_f64,
        |acc, i| {
            let d = values[i] as f64 - mean_wide;
            acc + d * d
        },
        |a, b| a + b,
    );
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
