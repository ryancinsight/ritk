//! Moments (Tsai 1985) moment-preserving thresholding.
//!
//! Matches ITK's `MomentsThresholdCalculator` / the Fiji Auto_Threshold
//! "Moments" method: choose the threshold so the thresholded image preserves the
//! first three moments of the original grey-level histogram.

use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

/// Tsai moment-preserving threshold.
#[derive(Debug, Clone)]
pub struct MomentsThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl MomentsThreshold {
    /// Create a `MomentsThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `MomentsThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the Moments threshold intensity for `image`.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Moments threshold to produce a binary mask.
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

impl Default for MomentsThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoThreshold for MomentsThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Tsai's moment-preserving criterion.
    ///
    /// Solve for the fraction `p0` of pixels assigned to the background such that
    /// the binary (two-grey-level) image preserves the first three moments of the
    /// histogram, then pick the grey level whose cumulative probability first
    /// reaches `p0`. Reported as the bin centre (ITK `GetMeasurement`).
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let total: f64 = hist.iter().map(|&c| c as f64).sum();
        if total == 0.0 {
            return x_min;
        }
        let p: Vec<f64> = hist.iter().map(|&c| c as f64 / total).collect();

        // Moments m0..m3 over bin indices.
        let mut m1 = 0.0_f64;
        let mut m2 = 0.0_f64;
        let mut m3 = 0.0_f64;
        for (i, &pi) in p.iter().enumerate() {
            let fi = i as f64;
            m1 += fi * pi;
            m2 += fi * fi * pi;
            m3 += fi * fi * fi * pi;
        }
        let m0 = 1.0_f64;

        let cd = m0 * m2 - m1 * m1;
        if cd.abs() < f64::EPSILON {
            return bin_center(x_min, itk_bin_width(x_min, x_max, n_bins), 0);
        }
        let c0 = (-m2 * m2 + m1 * m3) / cd;
        let c1 = (m0 * (-m3) + m2 * m1) / cd;
        let disc = (c1 * c1 - 4.0 * c0).max(0.0).sqrt();
        let z0 = 0.5 * (-c1 - disc);
        let z1 = 0.5 * (-c1 + disc);
        // Fraction of the histogram assigned to the background grey level.
        let p0 = ((z1 - m1) / (z1 - z0)).clamp(0.0, 1.0);

        // First grey level whose cumulative probability exceeds p0.
        let mut cum = 0.0_f64;
        let mut threshold = n_bins - 1;
        for (i, &pi) in p.iter().enumerate() {
            cum += pi;
            if cum > p0 {
                threshold = i;
                break;
            }
        }

        bin_center(x_min, itk_bin_width(x_min, x_max, n_bins), threshold)
    }
}

/// Convenience function: compute the Moments threshold with 256 bins.
pub fn moments_threshold<B: Backend, const D: usize>(image: &Image<f32, B, D>) -> f32 {
    MomentsThreshold::new().compute(image)
}

/// Compute the Moments threshold directly from a flat `&[f32]` slice.
pub fn compute_moments_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&MomentsThreshold::with_bins(num_bins), slice)
}
