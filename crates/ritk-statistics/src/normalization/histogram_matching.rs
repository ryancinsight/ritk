//! Histogram matching (histogram specification), matching ITK's
//! `HistogramMatchingImageFilter`.
//!
//! Transforms the intensity histogram of a source image so that it approximates
//! the intensity histogram of a reference image.
//!
//! # Algorithm (Nyúl, Udupa & Zhang 2000 / ITK)
//!
//! 1. Optionally threshold at the mean: when `threshold_at_mean` is set, the
//!    histogram landmarks are computed only over intensities `≥ mean`, so the
//!    (usually dominant) sub-mean background does not skew the mapping. The
//!    threshold is the image min otherwise.
//! 2. Build `num_bins`-bin histograms of source and reference over
//!    `[threshold, max]` and their cumulative distributions.
//! 3. Take `num_match_points` quantile landmarks at `j/(K+1)`, `j = 1..K`, of
//!    each cumulative distribution (the bin-centre intensity where the CDF first
//!    reaches the quantile).
//! 4. Form the monotone landmark pairs
//!    `[min, threshold, sQ₁…sQ_K, max] → [min, threshold, rQ₁…rQ_K, max]` and map
//!    every source intensity through the resulting piecewise-linear transform.
//!
//! # Invariants
//! - Output carries the same spatial metadata (origin, spacing, direction) as
//!   the source.
//! - `src_min → ref_min`, `src_max → ref_max` (landmark endpoints).
//! - A constant source (`src_min == src_max`) is returned unchanged.

use ritk_image::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec_infallible, rebuild};

/// Histogram matcher following ITK's `HistogramMatchingImageFilter`: quantile
/// landmarks with an optional mean threshold, mapped piecewise-linearly.
pub struct HistogramMatcher {
    /// Number of histogram levels used to estimate the quantile landmarks.
    pub num_bins: usize,
    /// Number of interior quantile match points `K` (ITK `NumberOfMatchPoints`).
    pub num_match_points: usize,
    /// Exclude sub-mean intensities from the landmark estimation
    /// (ITK `ThresholdAtMeanIntensity`).
    pub threshold_at_mean: bool,
}

impl HistogramMatcher {
    /// Create a matcher with `num_bins` histogram levels and ITK's default
    /// 7 match points with mean thresholding enabled.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn new(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self {
            num_bins,
            num_match_points: 7,
            threshold_at_mean: true,
        }
    }

    /// Set the number of interior quantile match points.
    pub fn with_match_points(mut self, num_match_points: usize) -> Self {
        self.num_match_points = num_match_points;
        self
    }

    /// Set whether sub-mean intensities are excluded from landmark estimation.
    pub fn with_threshold_at_mean(mut self, threshold_at_mean: bool) -> Self {
        self.threshold_at_mean = threshold_at_mean;
        self
    }

    /// Match the intensity histogram of `source` to that of `reference`.
    ///
    /// Returns a new `Image` with the same shape and spatial metadata as
    /// `source`. A constant source is returned unchanged.
    pub fn match_histograms<B: Backend, const D: usize>(
        &self,
        source: &Image<B, D>,
        reference: &Image<B, D>,
    ) -> Image<B, D> {
        let (mut src_vec, dims) = extract_vec_infallible(source);
        let (ref_vec, _) = extract_vec_infallible(reference);

        let (src_min, src_max, src_mean) = min_max_mean(&src_vec);
        let (ref_min, ref_max, ref_mean) = min_max_mean(&ref_vec);

        // Constant source: no transform can be estimated — return unchanged.
        if (src_max - src_min).abs() < f32::EPSILON {
            return Image::new(
                source.data().clone(),
                *source.origin(),
                *source.spacing(),
                *source.direction(),
            );
        }

        let src_thresh = if self.threshold_at_mean {
            src_mean
        } else {
            src_min
        };
        let ref_thresh = if self.threshold_at_mean {
            ref_mean
        } else {
            ref_min
        };

        // Interior quantile landmarks of each distribution above its threshold.
        let k = self.num_match_points;
        let src_q = quantile_landmarks(&src_vec, src_thresh, src_max, self.num_bins, k);
        let ref_q = quantile_landmarks(&ref_vec, ref_thresh, ref_max, self.num_bins, k);

        // Monotone landmark pairs: [min, thresh, Q₁…Q_K, max].
        let mut src_land = Vec::with_capacity(k + 3);
        let mut ref_land = Vec::with_capacity(k + 3);
        src_land.push(src_min);
        ref_land.push(ref_min);
        src_land.push(src_thresh);
        ref_land.push(ref_thresh);
        src_land.extend_from_slice(&src_q);
        ref_land.extend_from_slice(&ref_q);
        src_land.push(src_max);
        ref_land.push(ref_max);
        enforce_monotone(&mut src_land);
        enforce_monotone(&mut ref_land);

        for value in &mut src_vec {
            *value = piecewise_linear(*value, &src_land, &ref_land);
        }

        rebuild(src_vec, dims, source)
    }
}

impl Default for HistogramMatcher {
    fn default() -> Self {
        Self::new(256)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Minimum, maximum, and arithmetic mean of a non-empty slice.
fn min_max_mean(data: &[f32]) -> (f32, f32, f32) {
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sum = 0.0_f64;
    for &v in data {
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
        sum += v as f64;
    }
    (min, max, (sum / data.len() as f64) as f32)
}

/// Quantile landmark intensities at `j/(K+1)`, `j = 1..K`, of the cumulative
/// histogram of `data` restricted to `[lo, hi]` with `bins` levels.
///
/// Each landmark is the centre of the first bin whose cumulative frequency
/// reaches the quantile, mirroring ITK's `Histogram::Quantile` + bin measurement.
fn quantile_landmarks(data: &[f32], lo: f32, hi: f32, bins: usize, k: usize) -> Vec<f32> {
    if k == 0 {
        return Vec::new();
    }
    if hi <= lo {
        return vec![lo; k];
    }
    let bin_w = (hi - lo) / bins as f32;
    let mut hist = vec![0u64; bins];
    let mut total = 0u64;
    for &v in data {
        if v < lo {
            continue;
        }
        let b = (((v - lo) / bin_w) as usize).min(bins - 1);
        hist[b] += 1;
        total += 1;
    }
    if total == 0 {
        return vec![lo; k];
    }

    let mut landmarks = Vec::with_capacity(k);
    let mut acc = 0u64;
    let mut j = 1usize;
    for (bin, &count) in hist.iter().enumerate() {
        let next_acc = acc + count;
        while j <= k {
            let target = j as f64 / (k as f64 + 1.0) * total as f64;
            if (next_acc as f64) < target {
                break;
            }
            // ITK's `Histogram::Quantile` interpolates LINEARLY within the bin
            // from its lower edge (not the bin centre): the fraction is how far
            // the target sits between the cumulative count before this bin and
            // after it.
            let in_bin = count as f64;
            let frac = if in_bin > 0.0 {
                ((target - acc as f64) / in_bin).clamp(0.0, 1.0)
            } else {
                0.5
            };
            landmarks.push(lo + (bin as f64 + frac) as f32 * bin_w);
            j += 1;
        }
        acc = next_acc;
    }
    debug_assert_eq!(landmarks.len(), k);
    landmarks
}

/// Make a landmark sequence non-decreasing (numerical guard for interpolation).
fn enforce_monotone(v: &mut [f32]) {
    for i in 1..v.len() {
        if v[i] < v[i - 1] {
            v[i] = v[i - 1];
        }
    }
}

/// Piecewise-linear map of `v` through monotone landmark pairs `(xs → ys)`.
/// Constant (identity-on-the-output) outside the landmark span.
fn piecewise_linear(v: f32, xs: &[f32], ys: &[f32]) -> f32 {
    if v <= xs[0] {
        return ys[0];
    }
    let last = xs.len() - 1;
    if v >= xs[last] {
        return ys[last];
    }
    // xs is sorted non-decreasing; find the segment containing v.
    let hi = xs.partition_point(|&x| x <= v).min(last).max(1);
    let lo = hi - 1;
    let span = xs[hi] - xs[lo];
    if span.abs() < f32::EPSILON {
        return ys[lo];
    }
    let frac = (v - xs[lo]) / span;
    ys[lo].mul_add(1.0 - frac, ys[hi] * frac)
}

#[cfg(test)]
#[path = "tests_histogram_matching.rs"]
mod tests_histogram_matching;
