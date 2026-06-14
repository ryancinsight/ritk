//! Yen's maximum correlation thresholding method.
//!
//! # Mathematical Specification
//!
//! Yen's method (Yen, Chang, & Chang 1995) selects the threshold t* that
//! maximises a correlation criterion derived from the second-order statistics
//! of the thresholded image:
//!
//!   C(t) = −log( A(t)² + B(t)² )
//!
//! where:
//! - A(t) = Σ_{i=0}^{t}   p(i)²
//! - B(t) = Σ_{i=t+1}^{N−1} p(i)²
//! - p(i) = h\[i\] / n_total   (normalised histogram probability)
//!
//! The optimal threshold is:
//!
//!   t* = argmax_t C(t)
//!
//! with the intensity-domain mapping:
//!
//!   t*_intensity = x_min + t* · (x_max − x_min) / (N − 1)
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N) bins using prefix sums of squared probabilities.
//! Total:                  O(n + N).
//!
//! # References
//! - Yen, J.-C., Chang, F.-J., & Chang, S. (1995). "A new criterion for
//!   automatic multilevel thresholding." *IEEE Trans. Image Process.*, 4(3),
//!   370–378.

use burn::tensor::backend::Backend;
use ritk_image::Image;

use super::auto_threshold::AutoThreshold;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Yen's maximum correlation threshold segmentation.
///
/// Selects a threshold t* that maximises the correlation criterion
/// C(t) = −log(A(t)² + B(t)²), then applies it to produce a binary mask.
#[derive(Debug, Clone)]
pub struct YenThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl YenThreshold {
    /// Create a `YenThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `YenThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the optimal Yen threshold for `image`.
    ///
    /// Returns the intensity value t* that maximises the correlation criterion.
    /// For a constant image, returns the image's uniform intensity (degenerate case).
    ///
    /// Delegates to [`AutoThreshold::compute`].
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Yen threshold to produce a binary mask.
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

impl Default for YenThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ── AutoThreshold implementation ───────────────────────────────────────────────

impl AutoThreshold for YenThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Yen's maximum correlation criterion (ISO/ITK `YenThresholdCalculator`).
    ///
    /// # Algorithm
    /// 1. Normalise `hist` to probabilities `p[i] = count[i] / n_total`.
    /// 2. Cumulative probability `P1(t) = Σ_{i≤t} p(i)`, `P2(t) = 1 − P1(t)`, and
    ///    cumulative squared probability `P1sq(t) = Σ_{i≤t} p(i)²`,
    ///    `P2sq(t) = total_sq − P1sq(t)`.
    /// 3. For each t: `C(t) = −log(P1sq·P2sq) + 2·log(P1·P2)` (each `log` term is
    ///    taken as 0 when its argument is ≤ 0, matching ITK).
    /// 4. t* = argmax C(t).
    /// 5. t*_intensity = x_min + (t* + 0.5) / (N−1) · (x_max − x_min).
    ///
    /// Note: the criterion must keep the squared-probability masses `P1sq`/`P2sq`
    /// separate — their *sum* `P1sq + P2sq = total_sq` is constant in t, so the
    /// old `−log(P1sq + P2sq)` was degenerate (every bin scored identically,
    /// collapsing the threshold to the first bin).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let n: u64 = hist.iter().map(|&c| c as u64).sum();
        if n == 0 {
            return x_min;
        }

        // Normalise to probabilities and accumulate cumulative prob and prob².
        let p: Vec<f64> = hist.iter().map(|&c| c as f64 / n as f64).collect();
        let mut p1 = vec![0.0_f64; n_bins]; // Σ_{i≤t} p(i)
        let mut p1_sq = vec![0.0_f64; n_bins]; // Σ_{i≤t} p(i)²
        p1[0] = p[0];
        p1_sq[0] = p[0] * p[0];
        for i in 1..n_bins {
            p1[i] = p1[i - 1] + p[i];
            p1_sq[i] = p1_sq[i - 1] + p[i] * p[i];
        }
        let total_sq: f64 = p1_sq[n_bins - 1];

        // Search for t* = argmax C(t).
        let mut best_criterion = f64::NEG_INFINITY;
        let mut best_t = 0_usize;
        for t in 0..n_bins {
            let p1t = p1[t];
            let p2t = 1.0 - p1t;
            let p1sq = p1_sq[t];
            let p2sq = total_sq - p1_sq[t];
            let term_sq = if p1sq * p2sq > 0.0 {
                (p1sq * p2sq).ln()
            } else {
                0.0
            };
            let term_lin = if p1t * p2t > 0.0 {
                (p1t * p2t).ln()
            } else {
                0.0
            };
            let criterion = -term_sq + 2.0 * term_lin;
            if criterion > best_criterion {
                best_criterion = criterion;
                best_t = t;
            }
        }

        // Convert best bin index to intensity units (bin-centre half-bin shift).
        x_min + (best_t as f32 + 0.5) / (n_bins - 1) as f32 * (x_max - x_min)
    }
}

// ── Convenience functions ──────────────────────────────────────────────────────

/// Convenience function: compute the Yen threshold with 256 bins.
pub fn yen_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    YenThreshold::new().compute(image)
}

/// Compute the Yen threshold for a contiguous f32 intensity slice.
pub fn compute_yen_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
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
    let p: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    // ── Cumulative probability and squared probability ─────────────────────────
    // Yen's criterion C(t) = −log(P1sq·P2sq) + 2·log(P1·P2) needs the cumulative
    // probability mass P1 and squared mass P1sq kept separate; their *sums*
    // P1sq + P2sq = total_sq are constant in t, so collapsing to −log(P1sq+P2sq)
    // scores every bin identically (the prior degenerate bug). Matches ITK.
    let mut p1 = vec![0.0_f64; num_bins]; // Σ_{i≤t} p(i)
    let mut p1_sq = vec![0.0_f64; num_bins]; // Σ_{i≤t} p(i)²
    p1[0] = p[0];
    p1_sq[0] = p[0] * p[0];
    for i in 1..num_bins {
        p1[i] = p1[i - 1] + p[i];
        p1_sq[i] = p1_sq[i - 1] + p[i] * p[i];
    }
    let total_sq: f64 = p1_sq[num_bins - 1];

    // ── Search for optimal threshold ───────────────────────────────────────────
    let mut best_criterion = f64::NEG_INFINITY;
    let mut best_t = 0_usize;
    for t in 0..num_bins {
        let p1t = p1[t];
        let p2t = 1.0 - p1t;
        let p1sq = p1_sq[t];
        let p2sq = total_sq - p1_sq[t];
        let term_sq = if p1sq * p2sq > 0.0 {
            (p1sq * p2sq).ln()
        } else {
            0.0
        };
        let term_lin = if p1t * p2t > 0.0 {
            (p1t * p2t).ln()
        } else {
            0.0
        };
        let criterion = -term_sq + 2.0 * term_lin;
        if criterion > best_criterion {
            best_criterion = criterion;
            best_t = t;
        }
    }

    // ── Convert best bin index to intensity units ──────────────────────────────
    x_min + (best_t as f32 + 0.5) / num_bins_f * range
}

#[cfg(test)]
#[path = "tests_yen.rs"]
mod tests;
