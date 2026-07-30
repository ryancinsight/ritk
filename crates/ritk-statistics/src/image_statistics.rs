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

use coeus_core::{Backend, CpuAddressableStorage};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

use crate::StatisticsError;

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
///
/// # Errors
///
/// Returns [`StatisticsError::EmptyInput`] for an empty image or
/// [`StatisticsError::NonFiniteSample`] for NaN or infinite intensities.
pub fn compute_statistics<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
) -> Result<ImageStatistics, StatisticsError>
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
///
/// # Errors
///
/// Returns [`StatisticsError::EmptyInput`] for an empty slice,
/// [`StatisticsError::NonFiniteSample`] for NaN or infinite samples, and
/// [`StatisticsError::DegreesOfFreedomOutOfRange`] when `ddof >= slice.len()`.
pub fn compute_statistics_from_slice(
    slice: &[f32],
    ddof: usize,
) -> Result<ImageStatistics, StatisticsError> {
    compute_from_values(slice, ddof)
}

/// Compute statistics restricted to voxels where `mask` > 0.5 (foreground).
///
/// `mask` must have the same element count as `image`. Values greater than 0.5
/// select foreground voxels.
///
/// # Errors
///
/// Returns a typed error for empty or non-finite input, a length mismatch, or
/// a mask that selects no foreground samples.
pub fn masked_statistics<B: Backend, const D: usize>(
    image: &Image<f32, B, D>,
    mask: &Image<f32, B, D>,
) -> Result<ImageStatistics, StatisticsError>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (img_vals, _) = extract_vec_infallible(image);
    let (mask_vals, _) = extract_vec_infallible(mask);
    masked_statistics_from_slices(&img_vals, &mask_vals, 0)
}

/// Compute foreground statistics from borrowed image and mask buffers.
///
/// The foreground allocation is also the percentile workspace, so the masked
/// path performs no second full-size clone.
///
/// # Errors
///
/// Returns a typed error for empty or non-finite input, a length mismatch, an
/// empty foreground, or `ddof` greater than or equal to the foreground count.
pub fn masked_statistics_from_slices(
    image: &[f32],
    mask: &[f32],
    ddof: usize,
) -> Result<ImageStatistics, StatisticsError> {
    if image.len() != mask.len() {
        return Err(StatisticsError::ImageMaskLengthMismatch {
            image_count: image.len(),
            mask_count: mask.len(),
        });
    }
    if image.is_empty() {
        return Err(StatisticsError::EmptyInput);
    }

    let mut foreground = Vec::new();
    for (index, (&value, &mask_value)) in image.iter().zip(mask).enumerate() {
        if !value.is_finite() {
            return Err(StatisticsError::NonFiniteSample { index, value });
        }
        if !mask_value.is_finite() {
            return Err(StatisticsError::NonFiniteMaskSample {
                index,
                value: mask_value,
            });
        }
        if mask_value > crate::FOREGROUND_THRESHOLD {
            foreground.push(value);
        }
    }

    if foreground.is_empty() {
        return Err(StatisticsError::EmptyForeground);
    }
    validate_ddof(foreground.len(), ddof)?;
    Ok(compute_from_validated_owned(foreground, ddof))
}

/// Core statistics computation.
///
/// Copies validated finite `values` and partially reorders the copy in place.
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
///
/// # Errors
///
/// Returns a typed error for empty or non-finite input, or when `ddof` is
/// greater than or equal to the sample count.
pub fn compute_from_values(
    values: &[f32],
    ddof: usize,
) -> Result<ImageStatistics, StatisticsError> {
    validate_values(values, ddof)?;
    Ok(compute_from_validated_owned(values.to_vec(), ddof))
}

/// Compute statistics while consuming an owned buffer that may be reordered.
///
/// Masked-statistics paths already allocate this foreground buffer, so this
/// helper avoids cloning it before the in-place percentile selection.
pub(crate) fn compute_from_owned(
    buffer: Vec<f32>,
    ddof: usize,
) -> Result<ImageStatistics, StatisticsError> {
    validate_values(&buffer, ddof)?;
    Ok(compute_from_validated_owned(buffer, ddof))
}

fn validate_values(values: &[f32], ddof: usize) -> Result<(), StatisticsError> {
    if values.is_empty() {
        return Err(StatisticsError::EmptyInput);
    }
    validate_ddof(values.len(), ddof)?;
    if let Some((index, &value)) = values
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(StatisticsError::NonFiniteSample { index, value });
    }
    Ok(())
}

fn validate_ddof(sample_count: usize, ddof: usize) -> Result<(), StatisticsError> {
    if ddof >= sample_count {
        return Err(StatisticsError::DegreesOfFreedomOutOfRange { sample_count, ddof });
    }
    Ok(())
}

fn compute_from_validated_owned(mut buffer: Vec<f32>, ddof: usize) -> ImageStatistics {
    let values = buffer.as_mut_slice();
    let n = values.len();
    debug_assert!(n > 0, "validated statistics input is non-empty");
    debug_assert!(ddof < n, "validated ddof is less than sample count");

    // Fused pass: min, max, and the f64 sum in parallel.
    let (min, max, sum_wide) = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        n,
        || (values[0], values[0], 0.0_f64),
        |(min_acc, max_acc, sum_acc), i| {
            let v = values[i];
            let new_min = if v < min_acc { v } else { min_acc };
            let new_max = if v > max_acc { v } else { max_acc };
            (new_min, new_max, sum_acc + v as f64)
        },
        |(amin, amax, asum), (bmin, bmax, bsum)| {
            let rmin = if bmin < amin { bmin } else { amin };
            let rmax = if bmax > amax { bmax } else { amax };
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
    let denom = n - ddof;
    let std = (sum_sq_dev / denom as f64).sqrt() as f32;

    // Floor-division percentile ranks (module contract). Quickselect each rank
    // on the suffix left of the previous one — O(n) average, exact order
    // statistic, no full sort.
    let ranks = [n / 4, n / 2, (n / 4) * 3 + ((n % 4) * 3) / 4];
    let mut percentiles = [0.0_f32; 3];
    let mut lo = 0usize;
    for (slot, &rank) in percentiles.iter_mut().zip(ranks.iter()) {
        if rank >= lo {
            // Elements in `values[..lo]` are already ≤ everything in
            // `values[lo..]`, so the (rank − lo)-th smallest of the suffix is
            // the rank-th smallest overall.
            values[lo..].select_nth_unstable_by(rank - lo, f32::total_cmp);
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
