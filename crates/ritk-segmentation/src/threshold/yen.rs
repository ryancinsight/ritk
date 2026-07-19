//! Yen's maximum correlation thresholding method.
//!
//! # Mathematical Specification
//!
//! Yen's method (Yen, Chang, & Chang 1995) selects the threshold t* that
//! maximises a correlation criterion derived from the second-order statistics
//! of the thresholded image:
//!
//!   C(t) = âˆ’log( A(t)Â² + B(t)Â² )
//!
//! where:
//! - A(t) = Î£_{i=0}^{t}   p(i)Â²
//! - B(t) = Î£_{i=t+1}^{Nâˆ’1} p(i)Â²
//! - p(i) = h\[i\] / n_total   (normalised histogram probability)
//!
//! The optimal threshold is:
//!
//!   t* = argmax_t C(t)
//!
//! with the intensity-domain mapping:
//!
//!   t*_intensity = centre of the selected bin (ITK histogram geometry; see
//!   `auto_threshold`)
//!
//! # Complexity
//! Histogram construction: O(n) voxels.
//! Threshold search:       O(N) bins using prefix sums of squared probabilities.
//! Total:                  O(n + N).
//!
//! # References
//! - Yen, J.-C., Chang, F.-J., & Chang, S. (1995). "A new criterion for
//!   automatic multilevel thresholding." *IEEE Trans. Image Process.*, 4(3),
//!   370â€“378.

use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

// â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Yen's maximum correlation threshold segmentation.
///
/// Selects a threshold t* that maximises the correlation criterion
/// C(t) = âˆ’log(A(t)Â² + B(t)Â²), then applies it to produce a binary mask.
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
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Yen threshold to produce a binary mask.
    ///
    /// - Pixels with intensity â‰¥ t* â†’ 1.0 (foreground).
    /// - Pixels with intensity <  t* â†’ 0.0 (background).
    ///
    /// Spatial metadata (origin, spacing, direction) is preserved exactly.
    ///
    /// Delegates to [`AutoThreshold::apply`].
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        <Self as AutoThreshold>::apply(self, image)
    }

    /// Apply the auto-threshold to a Coeus-native image.
    ///
    /// # Errors
    ///
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the native output image cannot be constructed.
    pub fn apply_native<B, const D: usize>(
        &self,
        image: &ritk_image::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        <Self as AutoThreshold>::apply_native(self, image, backend)
    }
}

impl Default for YenThreshold {
    fn default() -> Self {
        Self::new()
    }
}

// â”€â”€ AutoThreshold implementation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl AutoThreshold for YenThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Yen's maximum correlation criterion (ISO/ITK `YenThresholdCalculator`).
    ///
    /// # Algorithm
    /// 1. Normalise `hist` to probabilities `p[i] = count[i] / n_total`.
    /// 2. Cumulative probability `P1(t) = Î£_{iâ‰¤t} p(i)`, `P2(t) = 1 âˆ’ P1(t)`, and
    ///    cumulative squared probability `P1sq(t) = Î£_{iâ‰¤t} p(i)Â²`,
    ///    `P2sq(t) = total_sq âˆ’ P1sq(t)`.
    /// 3. For each t: `C(t) = âˆ’log(P1sqÂ·P2sq) + 2Â·log(P1Â·P2)` (each `log` term is
    ///    taken as 0 when its argument is â‰¤ 0, matching ITK).
    /// 4. t* = argmax C(t).
    /// 5. t*_intensity = centre of the selected bin (ITK `GetMeasurement`).
    ///
    /// Note: the criterion must keep the squared-probability masses `P1sq`/`P2sq`
    /// separate â€” their *sum* `P1sq + P2sq = total_sq` is constant in t, so the
    /// old `âˆ’log(P1sq + P2sq)` was degenerate (every bin scored identically,
    /// collapsing the threshold to the first bin).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let n: u64 = hist.iter().map(|&c| c as u64).sum();
        if n == 0 {
            return x_min;
        }

        // Normalise to probabilities and accumulate cumulative prob and probÂ².
        let p: Vec<f64> = hist.iter().map(|&c| c as f64 / n as f64).collect();
        let mut p1 = vec![0.0_f64; n_bins]; // Î£_{iâ‰¤t} p(i)
        let mut p1_sq = vec![0.0_f64; n_bins]; // Î£_{iâ‰¤t} p(i)Â²
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

        // ITK YenThresholdCalculator reports the centre of the selected bin.
        bin_center(x_min, itk_bin_width(x_min, x_max, n_bins), best_t)
    }
}

// â”€â”€ Convenience functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Convenience function: compute the Yen threshold with 256 bins.
pub fn yen_threshold<B: Backend, const D: usize>(image: &Image<f32, B, D>) -> f32 {
    YenThreshold::new().compute(image)
}

/// Compute the Yen threshold for a contiguous f32 intensity slice.
///
/// Delegates to the shared `threshold_from_slice` pipeline so it is
/// bit-identical to [`YenThreshold::compute`].
pub fn compute_yen_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&YenThreshold::with_bins(num_bins), slice)
}

#[cfg(test)]
#[path = "tests_yen.rs"]
mod tests;
