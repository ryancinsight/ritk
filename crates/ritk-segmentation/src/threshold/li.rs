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
//!    t*_intensity = x_min + t* / (N − 1) · range
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

use burn::tensor::backend::Backend;
use ritk_image::Image;

use super::auto_threshold::AutoThreshold;

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

    /// Li's minimum cross-entropy iterative refinement.
    ///
    /// # Algorithm
    /// 1. Normalise `hist` to probabilities `h[i] = count[i] / n_total`.
    /// 2. Initialise t₀ = global mean bin index.
    /// 3. Iterate up to `self.max_iterations`:
    ///    - Compute background mean μ_b (bins [0, ⌊t⌋]).
    ///    - Compute foreground mean μ_f (bins [⌊t⌋+1, N−1]).
    ///    - t_{n+1} = (μ_b + μ_f) / 2.
    /// 4. Converge when |t_{n+1} − t_n| < 1e-6.
    /// 5. t*_intensity = x_min + t* / (N−1) · (x_max − x_min).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let n: u64 = hist.iter().map(|&c| c as u64).sum();
        if n == 0 {
            return x_min;
        }

        // Normalise to probabilities.
        let h: Vec<f64> = hist.iter().map(|&c| c as f64 / n as f64).collect();

        // Initialise at global mean bin index.
        let global_mean: f64 = (0..n_bins).map(|i| i as f64 * h[i]).sum();
        let mut t = global_mean;
        let tolerance = 1e-6_f64;

        for _ in 0..self.max_iterations {
            let t_floor = (t.floor() as usize).min(n_bins - 1);

            // Background: bins [0, t_floor].
            let mut w_b = 0.0_f64;
            let mut sum_b = 0.0_f64;
            for (i, &hi) in h.iter().enumerate().take(t_floor + 1) {
                w_b += hi;
                sum_b += i as f64 * hi;
            }

            // Foreground: bins [t_floor+1, N-1].
            let mut w_f = 0.0_f64;
            let mut sum_f = 0.0_f64;
            for (i, &hi) in h.iter().enumerate().take(n_bins).skip(t_floor + 1) {
                w_f += hi;
                sum_f += i as f64 * hi;
            }

            if w_b < 1e-12 || w_f < 1e-12 {
                break;
            }

            // Li's minimum cross-entropy update is the *logarithmic mean* of the
            // two class means, not their arithmetic mean — the latter is the
            // ISODATA/intermeans method and converges to a different threshold.
            // Means are taken in 1-based bin space so the logarithm is defined
            // when a class mean sits at bin 0 (matches ITK's LiThresholdCalculator).
            let mu_b = sum_b / w_b + 1.0;
            let mu_f = sum_f / w_f + 1.0;
            let t_new = if (mu_b - mu_f).abs() < 1e-12 {
                t
            } else {
                (mu_b - mu_f) / (mu_b.ln() - mu_f.ln()) - 1.0
            };

            if (t_new - t).abs() < tolerance {
                t = t_new;
                break;
            }
            t = t_new;
        }

        // Convert bin index to intensity units.
        x_min + t as f32 / (n_bins - 1) as f32 * (x_max - x_min)
    }
}

// ── Convenience functions ──────────────────────────────────────────────────────

/// Convenience function: compute the Li threshold with default parameters (256 bins, 1000 iterations).
pub fn li_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    LiThreshold::new().compute(image)
}

/// Compute the Li threshold for a contiguous f32 intensity slice.
pub fn compute_li_threshold_from_slice(
    slice: &[f32],
    num_bins: usize,
    max_iterations: usize,
) -> f32 {
    assert!(num_bins >= 2, "num_bins must be >= 2");

    let n = slice.len();
    if n == 0 {
        return 0.0;
    }

    // ── Intensity range ────────────────────────────────────────────────────────
    let x_min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
    let x_max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Degenerate case: constant image has no separable classes.
    if (x_max - x_min).abs() < f32::EPSILON {
        return x_min;
    }

    let range = x_max - x_min;
    let num_bins_f = (num_bins - 1) as f64;

    // ── Build normalised histogram ─────────────────────────────────────────────
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) / range * num_bins_f as f32).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    // ── Global mean bin index (initialization) ─────────────────────────────────
    let global_mean: f64 = (0..num_bins).map(|i| i as f64 * h[i]).sum();

    // ── Iterative refinement ───────────────────────────────────────────────────
    let mut t = global_mean;
    let tolerance = 1e-6_f64;

    for _ in 0..max_iterations {
        let t_floor = (t.floor() as usize).min(num_bins - 1);

        // ── Background: bins [0, t_floor] ──────────────────────────────────────
        let mut w_b = 0.0_f64;
        let mut sum_b = 0.0_f64;
        for (i, &hi) in h.iter().enumerate().take(t_floor + 1) {
            w_b += hi;
            sum_b += i as f64 * hi;
        }

        // ── Foreground: bins [t_floor+1, N-1] ─────────────────────────────────
        let mut w_f = 0.0_f64;
        let mut sum_f = 0.0_f64;
        for (i, &hi) in h.iter().enumerate().take(num_bins).skip(t_floor + 1) {
            w_f += hi;
            sum_f += i as f64 * hi;
        }

        // If either class is empty, the threshold is at the boundary.
        if w_b < 1e-12 || w_f < 1e-12 {
            break;
        }

        // Li's minimum cross-entropy update: the logarithmic mean of the class
        // means (1-based bin space so log is defined at bin 0), not the arithmetic
        // mean — (mu_b + mu_f)/2 is the ISODATA method and converges elsewhere.
        let mu_b = sum_b / w_b + 1.0;
        let mu_f = sum_f / w_f + 1.0;
        let t_new = if (mu_b - mu_f).abs() < 1e-12 {
            t
        } else {
            (mu_b - mu_f) / (mu_b.ln() - mu_f.ln()) - 1.0
        };

        if (t_new - t).abs() < tolerance {
            t = t_new;
            break;
        }
        t = t_new;
    }

    // ── Convert bin index to intensity units ───────────────────────────────────
    x_min + (t as f32) / num_bins_f as f32 * range
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_li.rs"]
mod tests;
