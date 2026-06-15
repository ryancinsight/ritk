//! Kapur's maximum entropy thresholding (Kapur, Sahoo & Wong 1985).
//!
//! # Mathematical Specification
//!
//! Kapur's method selects the intensity threshold t* that maximises the
//! sum of foreground and background entropies of the intensity histogram:
//!
//!   H(t) = H_b(t) + H_f(t)
//!
//! where:
//! - P_b(t) = Σ_{i=0}^{t} p(i)                          (background probability mass)
//! - P_f(t) = Σ_{i=t+1}^{N-1} p(i)                      (foreground probability mass)
//! - H_b(t) = -Σ_{i=0}^{t} (p(i)/P_b) · ln(p(i)/P_b)   (background entropy)
//! - H_f(t) = -Σ_{i=t+1}^{N-1} (p(i)/P_f) · ln(p(i)/P_f) (foreground entropy)
//! - p(i)   = count\[i\] / n_total                          (normalised histogram)
//!
//! The optimal threshold in original intensity units is:
//!
//!   t*_intensity = x_min + t* · (x_max − x_min) / (N − 1)
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N²) bins (entropy sums per candidate).
//! Total:                  O(n + N²).
//!
//! # References
//! - J. N. Kapur, P. K. Sahoo, A. K. C. Wong, "A New Method for Gray-Level
//!   Picture Thresholding Using the Entropy of the Histogram," *Computer
//!   Vision, Graphics, and Image Processing*, 29(3):273–285, 1985.

use burn::tensor::backend::Backend;
use ritk_image::Image;

use super::auto_threshold::AutoThreshold;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Maximum-entropy threshold segmentation (Kapur et al. 1985).
///
/// Selects a threshold t* that maximises the combined foreground and background
/// entropy of the intensity histogram, then applies it to produce a binary mask.
#[derive(Debug, Clone)]
pub struct KapurThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl KapurThreshold {
    /// Create a `KapurThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `KapurThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the optimal Kapur threshold for `image`.
    ///
    /// Returns the intensity value t* that maximises total entropy H(t).
    /// For a constant image, returns the image's uniform intensity (degenerate case).
    ///
    /// Delegates to [`AutoThreshold::compute`].
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Kapur threshold to produce a binary mask.
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

impl Default for KapurThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ── AutoThreshold implementation ───────────────────────────────────────────────

impl AutoThreshold for KapurThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Kapur's maximum combined entropy criterion.
    ///
    /// # Algorithm
    /// 1. Normalise `hist` to probabilities `h[i] = count[i] / n_total`.
    /// 2. Build cumulative probability prefix sums.
    /// 3. For each candidate t ∈ [0, N−2]:
    ///    H(t) = H_b(t) + H_f(t)
    ///    (background and foreground Shannon entropies of their respective
    ///    conditional distributions).
    /// 4. t* = argmax H(t).
    /// 5. t*_intensity = x_min + t* / (N−1) · (x_max − x_min).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let n: u64 = hist.iter().map(|&c| c as u64).sum();
        if n == 0 {
            return x_min;
        }

        // Normalise to probabilities.
        let h: Vec<f64> = hist.iter().map(|&c| c as f64 / n as f64).collect();

        // Prefix sums for cumulative probability.
        let mut cum_prob = vec![0.0_f64; n_bins];
        cum_prob[0] = h[0];
        for i in 1..n_bins {
            cum_prob[i] = cum_prob[i - 1] + h[i];
        }

        let mut best_entropy = f64::NEG_INFINITY;
        let mut best_t = 0_usize;

        for (t, &cp_t) in cum_prob.iter().enumerate().take(n_bins - 1) {
            let p_b = cp_t;
            let p_f = 1.0 - p_b;

            if p_b < super::PROB_ZERO_GUARD || p_f < super::PROB_ZERO_GUARD {
                continue;
            }

            // Background entropy: H_b = -Σ_{i=0}^{t} (p(i)/P_b)·ln(p(i)/P_b).
            let mut h_b = 0.0_f64;
            for &hi in h.iter().take(t + 1) {
                if hi > super::PROB_ZERO_GUARD {
                    let q = hi / p_b;
                    h_b -= q * q.ln();
                }
            }

            // Foreground entropy: H_f = -Σ_{i=t+1}^{N-1} (p(i)/P_f)·ln(p(i)/P_f).
            let mut h_f = 0.0_f64;
            for &hi in h.iter().take(n_bins).skip(t + 1) {
                if hi > super::PROB_ZERO_GUARD {
                    let q = hi / p_f;
                    h_f -= q * q.ln();
                }
            }

            let total_entropy = h_b + h_f;
            if total_entropy > best_entropy {
                best_entropy = total_entropy;
                best_t = t;
            }
        }

        // Convert best bin index to intensity units.
        x_min + best_t as f32 / (n_bins - 1) as f32 * (x_max - x_min)
    }
}

// ── Convenience functions ──────────────────────────────────────────────────────

/// Convenience function: compute the Kapur threshold with 256 bins.
pub fn kapur_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    KapurThreshold::new().compute(image)
}

/// Compute the Kapur threshold for a contiguous f32 intensity slice.
pub fn compute_kapur_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
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
    let num_bins_f = (num_bins - 1) as f32;

    // ── Build normalised histogram ─────────────────────────────────────────────
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) / range * num_bins_f).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    // ── Prefix sums for cumulative probability ─────────────────────────────────
    let mut cum_prob = vec![0.0_f64; num_bins];
    cum_prob[0] = h[0];
    for i in 1..num_bins {
        cum_prob[i] = cum_prob[i - 1] + h[i];
    }

    // ── Search for optimal threshold ───────────────────────────────────────────
    let mut best_entropy = f64::NEG_INFINITY;
    let mut best_t = 0_usize;

    for (t, &cp_t) in cum_prob.iter().enumerate().take(num_bins - 1) {
        let p_b = cp_t;
        let p_f = 1.0 - p_b;

        // Skip degenerate splits where one class is empty.
        if p_b < super::PROB_ZERO_GUARD || p_f < super::PROB_ZERO_GUARD {
            continue;
        }

        // Background entropy: H_b = -Σ_{i=0}^{t} (p(i)/P_b) · ln(p(i)/P_b)
        let mut h_b = 0.0_f64;
        for &hi in h.iter().take(t + 1) {
            if hi > super::PROB_ZERO_GUARD {
                let q = hi / p_b;
                h_b -= q * q.ln();
            }
        }

        // Foreground entropy: H_f = -Σ_{i=t+1}^{N-1} (p(i)/P_f) · ln(p(i)/P_f)
        let mut h_f = 0.0_f64;
        for &hi in h.iter().take(num_bins).skip(t + 1) {
            if hi > super::PROB_ZERO_GUARD {
                let q = hi / p_f;
                h_f -= q * q.ln();
            }
        }

        let total_entropy = h_b + h_f;
        if total_entropy > best_entropy {
            best_entropy = total_entropy;
            best_t = t;
        }
    }

    // ── Convert best bin index to intensity units ──────────────────────────────
    x_min + best_t as f32 / num_bins_f * range
}

#[cfg(test)]
#[path = "tests_kapur.rs"]
mod tests;
