//! Per-label intensity statistics over a labeled segmentation map.
//!
//! # Mathematical Specification
//!
//! Given a label map L and intensity image I with identical shape [Z, Y, X]:
//!
//! For each non-zero label k in {1, 2, ..., K}:
//! - V_k = { I(x) : L(x) = k }
//! - count_k  = |V_k|
//! - min_k    = min(V_k)
//! - max_k    = max(V_k)
//! - mean_k   = (1/count_k) * sum_{v in V_k} v
//! - std_k    = sqrt( (1/count_k) * sum_{v in V_k} (v - mean_k)^2 )  (population std)
//!
//! Background (label 0) is excluded from results.
//!
//! # Complexity
//! Single O(N) parallel fold/reduce pass over (label, intensity) pairs.
//! HashMap per thread accumulates (min, max, sum, sum_sq, count).
//!
//! # Reference
//! ITK LabelStatisticsImageFilter -- per-label min, max, mean, sigma, count.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
use std::collections::HashMap;

/// Intensity statistics for a single label region.
#[derive(Debug, Clone, PartialEq)]
pub struct LabelIntensityStatistics {
    /// Integer label index (>= 1); background (0) is excluded.
    pub label: u32,
    /// Number of voxels carrying this label.
    pub count: usize,
    /// Minimum intensity value within this label region.
    pub min: f32,
    /// Maximum intensity value within this label region.
    pub max: f32,
    /// Arithmetic mean intensity.
    pub mean: f32,
    /// Standard deviation of intensity with the requested `ddof`
    /// (0 = population, 1 = sample / ITK `LabelStatisticsImageFilter`).
    pub std: f32,
}

/// Compute per-label intensity statistics.
///
/// For each non-zero label in `label_image`, accumulates `min`, `max`, `mean`,
/// and `std` over all voxels of `intensity_image` where `label_image` carries
/// that label.  Background voxels (label = 0) are excluded from results.
///
/// Returns a `Vec<LabelIntensityStatistics>` sorted by label index.
///
/// # Panics
/// Panics if `label_image` and `intensity_image` have different shapes.
pub fn compute_label_intensity_statistics<B: Backend>(
    label_image: &Image<f32, B, 3>,
    intensity_image: &Image<f32, B, 3>,
) -> Vec<LabelIntensityStatistics> {
    assert_eq!(
        label_image.shape(),
        intensity_image.shape(),
        "label_image and intensity_image must have identical shapes"
    );

    let (label_vals, _) = extract_vec_infallible(label_image);
    let label_slice: &[f32] = &label_vals;
    let (intensity_vals, _) = extract_vec_infallible(intensity_image);
    let intensity_slice: &[f32] = &intensity_vals;

    compute_label_intensity_statistics_from_slices(label_slice, intensity_slice, 0)
}

/// Compute per-label intensity statistics from pre-extracted flat slices.
///
/// Zero-copy variant: accepts slices directly.  Label values cast to `u32` via
/// truncation; background (label 0) is excluded.
///
/// Returns a `Vec<LabelIntensityStatistics>` sorted by label index.
///
/// # Panics
/// Panics if `label_slice` and `intensity_slice` have different lengths.
pub fn compute_label_intensity_statistics_from_slices(
    label_slice: &[f32],
    intensity_slice: &[f32],
    ddof: usize,
) -> Vec<LabelIntensityStatistics> {
    assert_eq!(
        label_slice.len(),
        intensity_slice.len(),
        "label_slice and intensity_slice must have equal length"
    );

    // Accumulator: (min, max, sum_f64, sum_sq_f64, count)
    type Acc = (f32, f32, f64, f64, usize);

    let combined: HashMap<u32, Acc> = moirai::fold_reduce_with::<moirai::Adaptive, _, _, _, _>(
        label_slice.len(),
        HashMap::<u32, Acc>::new,
        |mut acc, i| {
            let label = label_slice[i] as u32;
            let intensity = intensity_slice[i];
            if label == 0 {
                return acc;
            }
            let entry = acc.entry(label).or_insert((
                f32::INFINITY,
                f32::NEG_INFINITY,
                0.0_f64,
                0.0_f64,
                0_usize,
            ));
            let v = intensity as f64;
            entry.0 = entry.0.min(intensity);
            entry.1 = entry.1.max(intensity);
            entry.2 += v;
            entry.3 += v * v;
            entry.4 += 1;
            acc
        },
        |mut a, b| {
            for (k, (bmin, bmax, bsum, bsumsq, bcnt)) in b {
                let e = a
                    .entry(k)
                    .or_insert((f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0, 0));
                e.0 = e.0.min(bmin);
                e.1 = e.1.max(bmax);
                e.2 += bsum;
                e.3 += bsumsq;
                e.4 += bcnt;
            }
            a
        },
    );

    let mut result: Vec<LabelIntensityStatistics> = combined
        .into_iter()
        .map(|(label, (min, max, sum, sum_sq, count))| {
            let n = count as f64;
            let mean_wide = sum / n;
            let mean = mean_wide as f32;
            // Σ(v − mean)² = Σv² − n·mean²; .max(0.0) absorbs f64 cancellation.
            let sum_sq_dev = (sum_sq - n * mean_wide * mean_wide).max(0.0);
            // numpy-style ddof: divisor = count − ddof (0 = population, 1 = sample).
            let denom = count.saturating_sub(ddof);
            let std = if denom == 0 {
                0.0
            } else {
                (sum_sq_dev / denom as f64).sqrt() as f32
            };
            LabelIntensityStatistics {
                label,
                count,
                min,
                max,
                mean,
                std,
            }
        })
        .collect();

    result.sort_by_key(|s| s.label);
    result
}

#[cfg(test)]
#[path = "tests_label_statistics.rs"]
mod tests;
