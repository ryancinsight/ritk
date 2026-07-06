//! Li's minimum cross-entropy thresholding (Li & Tam 1998).
//!
//! # Mathematical Specification
//!
//! Li's method iteratively minimizes the cross-entropy between the original
//! image and its thresholded version. The iteration scheme converges to the
//! threshold that minimizes the Kullback–Leibler divergence of the two-class
//! model from the original intensity distribution.
//!
//! ## Algorithm
//!
//! 1. Compute a normalized histogram h\[i\] over N bins.
//! 2. Initialize: t₀ = μ (global mean intensity in bin-index space).
//! 3. Iterate:
//!    μ_b(t) = Σ_{i=0}^{⌊t⌋}   i·h\[i\] / Σ_{i=0}^{⌊t⌋}   h\[i\]
//!    μ_f(t) = Σ_{i=⌊t⌋+1}^{N-1} i·h\[i\] / Σ_{i=⌊t⌋+1}^{N-1} h\[i\]
//!    t_{n+1} = (μ_b + μ_f) / 2
//! 4. Converge when |t_{n+1} − t_n| < tolerance (1e-6) or max_iterations reached.
//! 5. Convert the converged bin index to intensity units:
//!    t*_intensity = centre of the converged bin under ITK histogram geometry
//!    (see `auto_threshold`); the iteration runs in measurement space.
//!
//! # Complexity
//!
//! Histogram construction: O(n) voxels.
//! Each iteration:          O(N) bins.
//! Total:                   O(n + k·N), k = number of iterations until convergence.
//!
//! # References
//!
//! - Li, C.H. & Tam, P.K.S. (1998). "An iterative algorithm for minimum
//!   cross entropy thresholding." *Pattern Recognition Letters*, 19(8), 771–776.

use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

// ── Public API ─────────────────────────────────────────────────────────────────

/// Li's minimum cross-entropy thresholding.
///
/// Iteratively refines a threshold by computing the midpoint of the
/// foreground and background conditional means until convergence.
#[derive(Debug, Clone)]
pub struct LiThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
    /// Maximum number of iterations before forced termination. Default 1000.
    pub max_iterations: usize,
}

impl LiThreshold {
    /// Create a `LiThreshold` with 256 histogram bins and 1000 max iterations.
    pub fn new() -> Self {
        Self {
            num_bins: 256,
            max_iterations: 1000,
        }
    }

    /// Compute the optimal Li threshold for `image`.
    ///
    /// Returns the intensity value t* that minimizes the cross-entropy
    /// between the image and its binary thresholded version.
    /// For a constant image, returns the image's uniform intensity.
    ///
    /// Delegates to [`AutoThreshold::compute`].
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Li threshold to produce a binary mask.
    ///
    /// - Pixels with intensity ≥ t* → 1.0 (foreground).
    /// - Pixels with intensity <  t* → 0.0 (background).
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    ///
    /// Delegates to [`AutoThreshold::apply`].
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        <Self as AutoThreshold>::apply(self, image)
    }
}

impl Default for LiThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ── AutoThreshold implementation ───────────────────────────────────────────────

impl AutoThreshold for LiThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Li's minimum cross-entropy iterative refinement, faithful to
    /// `itk::LiThresholdCalculator`.
    ///
    /// # Algorithm
    /// The iteration runs in **measurement (intensity) space** over the bin
    /// centres `c[i] = x_min + (i + 0.5)·bin_width`, not bin-index space — the
    /// `log` in Li's update is non-linear, so index-space and measurement-space
    /// iterations converge to different thresholds.
    ///
    /// 1. `mean = Σ c[i]·f[i] / Σ f[i]`; initialise `new = mean`, `old = NaN`.
    /// 2. `bin_min = min(x_min, 0)` (shift applied before the logs so they are
    ///    defined for non-positive intensities).
    /// 3. Loop while `|new − old| > 0.5` (ITK's fixed tolerance):
    ///    - `ht` = bin index containing `old`.
    ///    - `μ_b` = mean centre of bins `[0, ht]`, `μ_f` = mean centre of bins
    ///      `(ht, N)`; both shifted by `−bin_min`.
    ///    - `temp = (μ_b − μ_f) / (ln μ_b − ln μ_f)`, **rounded to the nearest
    ///      integer** (ITK truncates `temp ± 0.5` toward zero), then
    ///      `new = temp + bin_min`.
    /// 4. Return the centre of the bin containing the converged `new`
    ///    (ITK `GetMeasurement(ht)`).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let total: f64 = hist.iter().map(|&c| c as f64).sum();
        if total == 0.0 {
            return x_min;
        }

        let bin_width = itk_bin_width(x_min, x_max, n_bins);
        let x_lo = x_min as f64;
        let center = |i: usize| x_lo + (i as f64 + 0.5) * bin_width;
        let index_of = |v: f64| -> usize {
            (((v - x_lo) / bin_width).floor().max(0.0) as usize).min(n_bins - 1)
        };

        // Inclusive prefix sums of frequency and centre-weighted frequency for
        // O(1) background/foreground means at any split bin.
        let mut prefix_f = vec![0.0_f64; n_bins];
        let mut prefix_cf = vec![0.0_f64; n_bins];
        let mut acc_f = 0.0_f64;
        let mut acc_cf = 0.0_f64;
        for i in 0..n_bins {
            acc_f += hist[i] as f64;
            acc_cf += center(i) * hist[i] as f64;
            prefix_f[i] = acc_f;
            prefix_cf[i] = acc_cf;
        }
        let total_cf = acc_cf;

        // ITK shifts means by bin_min = min(x_min, 0) so the logarithms are
        // defined when intensities are non-positive.
        let bin_min = x_lo.min(0.0);
        let eps = f64::EPSILON;

        let mut new_thresh = total_cf / total; // global mean intensity
        let mut old_thresh = f64::NAN;

        for _ in 0..self.max_iterations {
            if (new_thresh - old_thresh).abs() <= 0.5 {
                break;
            }
            old_thresh = new_thresh;

            let ht = index_of(old_thresh);
            let n_back = prefix_f[ht];
            let s_back = prefix_cf[ht];
            let n_fore = total - n_back;
            let s_fore = total_cf - s_back;
            if n_back == 0.0 || n_fore == 0.0 {
                break;
            }

            let mean_back = s_back / n_back - bin_min;
            let mean_obj = s_fore / n_fore - bin_min;
            if mean_back <= 0.0 || mean_obj <= 0.0 {
                break;
            }

            let temp = if (mean_back - mean_obj).abs() < eps {
                mean_back
            } else {
                (mean_back - mean_obj) / (mean_back.ln() - mean_obj.ln())
            };

            // ITK rounds toward the nearest integer, truncating ±0.5 toward zero.
            let temp = if temp < -eps {
                (temp - 0.5).trunc()
            } else {
                (temp + 0.5).trunc()
            };
            new_thresh = temp + bin_min;
        }

        // ITK returns the centre of the bin containing the converged threshold.
        bin_center(x_min, bin_width, index_of(new_thresh))
    }
}

// ── Convenience functions ──────────────────────────────────────────────────────

/// Convenience function: compute the Li threshold with default parameters (256 bins, 1000 iterations).
pub fn li_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    LiThreshold::new().compute(image)
}

/// Compute the Li threshold for a contiguous f32 intensity slice.
///
/// Delegates to the shared `threshold_from_slice` pipeline so it is
/// bit-identical to [`LiThreshold::compute`].
pub fn compute_li_threshold_from_slice(
    slice: &[f32],
    num_bins: usize,
    max_iterations: usize,
) -> f32 {
    assert!(num_bins >= 2, "num_bins must be >= 2");
    threshold_from_slice(
        &LiThreshold {
            num_bins,
            max_iterations,
        },
        slice,
    )
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_li.rs"]
mod tests;
