//! Otsu's single-threshold segmentation method.
//!
//! # Mathematical Specification
//!
//! Otsu's method selects the intensity threshold t* that maximises the
//! between-class variance of two intensity classes:
//!
//!   σ²_B(t) = P₁(t) · P₂(t) · (μ₁(t) − μ₂(t))²
//!
//! where:
//! - P₁(t) = Σ_{i=0}^{t−1} h\[i\]              (weight of class 1, bins 0..t−1)
//! - P₂(t) = 1 − P₁(t)                         (weight of class 2, bins t..N−1)
//! - μ₁(t) = Σ_{i=0}^{t−1} i·h\[i\] / P₁(t)   (mean bin index of class 1)
//! - μ₂(t) = Σ_{i=t}^{N−1} i·h\[i\] / P₂(t)   (mean bin index of class 2)
//! - h\[i\] = count\[i\] / n_total              (normalised histogram)
//!
//! The optimal threshold in original intensity units is:
//!
//!   t*_intensity = x_min + t* · (x_max − x_min) / (N − 1)
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N) bins using prefix sums.
//! Total:                  O(n + N).

use burn::tensor::backend::Backend;
use ritk_image::Image;

use super::auto_threshold::AutoThreshold;

// ── Public API ─────────────────────────────────────────────────────────────────

/// Single-threshold Otsu segmentation.
///
/// Selects a threshold t* that maximises the between-class variance of the
/// intensity histogram, then applies it to produce a binary mask.
pub struct OtsuThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl OtsuThreshold {
    /// Create an `OtsuThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create an `OtsuThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the optimal Otsu threshold for `image`.
    ///
    /// Returns the intensity value t* that maximises between-class variance.
    /// For a constant image, returns the image's uniform intensity (degenerate case).
    ///
    /// Delegates to [`AutoThreshold::compute`].
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Otsu threshold to produce a binary mask.
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

impl Default for OtsuThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// ── AutoThreshold implementation ───────────────────────────────────────────────

impl AutoThreshold for OtsuThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Otsu's between-class variance criterion.
    ///
    /// # Algorithm
    /// 1. Normalise `hist` to probabilities `h[i] = count[i] / n_total`.
    /// 2. Compute the total weighted mean `μ = Σ i·h[i]`.
    /// 3. O(N) prefix-sum scan over t ∈ [1, N−1]:
    ///    σ²_B(t) = P₁(t)·P₂(t)·(μ₁(t)−μ₂(t))²
    /// 4. t* = argmax σ²_B.
    /// 5. t*_intensity = x_min + t* / (N−1) · (x_max − x_min).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let n: u64 = hist.iter().map(|&c| c as u64).sum();
        if n == 0 {
            return x_min;
        }

        // Normalise to probabilities.
        let h: Vec<f64> = hist.iter().map(|&c| c as f64 / n as f64).collect();

        // Total weighted mean over bin indices.
        let total_mu: f64 = (0..n_bins).map(|i| i as f64 * h[i]).sum();

        // O(N) prefix-sum scan.
        let mut best_sigma2 = 0.0_f64;
        let mut best_t = 0_usize;
        let mut w1 = 0.0_f64; // Σ h[0..t−1]
        let mut mu1_partial = 0.0_f64; // Σ i·h[i] for i ∈ [0, t−1]

        for t in 1..n_bins {
            w1 += h[t - 1];
            mu1_partial += (t - 1) as f64 * h[t - 1];

            let w2 = 1.0 - w1;
            if w1 < 1e-12 || w2 < 1e-12 {
                continue;
            }

            let mu1 = mu1_partial / w1;
            let mu2 = (total_mu - mu1_partial) / w2;
            let sigma2 = w1 * w2 * (mu1 - mu2) * (mu1 - mu2);

            if sigma2 > best_sigma2 {
                best_sigma2 = sigma2;
                best_t = t;
            }
        }

        // Convert best bin index to intensity units.
        x_min + best_t as f32 / (n_bins - 1) as f32 * (x_max - x_min)
    }
}

// ── Convenience functions ──────────────────────────────────────────────────────

/// Convenience function: compute the Otsu threshold with 256 bins.
pub fn otsu_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    OtsuThreshold::new().compute(image)
}

/// Compute the Otsu threshold directly from a flat `&[f32]` slice.
///
/// Equivalent to [`otsu_threshold`] but accepts pre-extracted slice data,
/// enabling zero-copy extraction when the caller has already obtained a slice
/// from the backend primitive (e.g., NdArray `ArcArray::as_slice_memory_order`).
///
/// # Arguments
/// * `slice`    - Flat pixel intensities in any order.
/// * `num_bins` - Number of equally-spaced histogram bins; must be >= 2.
///
/// # Returns
/// The threshold intensity value t* that maximises between-class variance.
/// For an empty or constant input, returns 0.0 or the uniform intensity respectively.
pub fn compute_otsu_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    let n = slice.len();
    if n == 0 {
        return 0.0;
    }

    // -- Intensity range -------------------------------------------------------
    let (x_min, x_max) = slice
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(mn, mx), &v| {
            (mn.min(v), mx.max(v))
        });

    // Degenerate case: constant image has no separable classes.
    if (x_max - x_min).abs() < f32::EPSILON {
        return x_min;
    }

    let range = x_max - x_min;
    let num_bins_f = (num_bins - 1) as f32;

    // -- Build normalised histogram --------------------------------------------
    let mut counts = vec![0u64; num_bins];
    for &v in slice {
        let bin = ((v - x_min) / range * num_bins_f).floor() as usize;
        let bin = bin.min(num_bins - 1);
        counts[bin] += 1;
    }
    let h: Vec<f64> = counts.iter().map(|&c| c as f64 / n as f64).collect();

    // -- Total weighted mean over bin indices ----------------------------------
    // Used with the prefix-sum trick: mu2 = (total_mu - mu1_partial) / w2.
    let total_mu: f64 = (0..num_bins).map(|i| i as f64 * h[i]).sum();

    // -- Prefix-sum scan: O(N) threshold search --------------------------------
    // At threshold index t:
    //   Class 1 = bins [0, t-1],   Class 2 = bins [t, N-1].
    let mut best_sigma2 = 0.0_f64;
    let mut best_t = 0_usize;

    let mut w1 = 0.0_f64; // sum h[0..t-1]
    let mut mu1_partial = 0.0_f64; // sum i*h[i] for i in [0, t-1]

    for t in 1..num_bins {
        // Extend class 1 to include bin t-1.
        w1 += h[t - 1];
        mu1_partial += (t - 1) as f64 * h[t - 1];

        let w2 = 1.0 - w1;

        // Skip degenerate splits where one class is empty.
        if w1 < 1e-12 || w2 < 1e-12 {
            continue;
        }

        let mu1 = mu1_partial / w1;
        let mu2 = (total_mu - mu1_partial) / w2;

        let sigma2 = w1 * w2 * (mu1 - mu2) * (mu1 - mu2);

        if sigma2 > best_sigma2 {
            best_sigma2 = sigma2;
            best_t = t;
        }
    }

    // -- Convert best bin index to intensity units ----------------------------
    // t*_intensity = x_min + best_t / (N - 1) * range
    x_min + best_t as f32 / num_bins_f * range
}

#[cfg(test)]
#[path = "tests_otsu.rs"]
mod tests_otsu;
