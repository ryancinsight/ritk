//! Shanbhag thresholding (Shanbhag 1994).
//!
//! Matches ITK's `ShanbhagThresholdCalculator`: choose the threshold that
//! balances the information measures of the background and foreground fuzzy
//! membership distributions (minimises `|ent_back âˆ’ ent_obj|`).

use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

const TOLERANCE: f64 = 2.220446049250313e-16;

/// Shanbhag information-measure threshold.
#[derive(Debug, Clone)]
pub struct ShanbhagThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl ShanbhagThreshold {
    /// Create a `ShanbhagThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `ShanbhagThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the Shanbhag threshold intensity for `image`.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Shanbhag threshold to produce a binary mask.
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

impl Default for ShanbhagThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoThreshold for ShanbhagThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let bw = itk_bin_width(x_min, x_max, n_bins);
        let total: f64 = hist.iter().map(|&c| c as f64).sum();
        if total == 0.0 {
            return x_min;
        }
        let norm: Vec<f64> = hist.iter().map(|&c| c as f64 / total).collect();

        // Cumulative background (P1) and foreground (P2) probabilities.
        let mut p1 = vec![0.0_f64; n_bins];
        let mut p2 = vec![0.0_f64; n_bins];
        p1[0] = norm[0];
        p2[0] = 1.0 - p1[0];
        for i in 1..n_bins {
            p1[i] = p1[i - 1] + norm[i];
            p2[i] = 1.0 - p1[i];
        }

        let first_bin = (0..n_bins).find(|&i| p1[i].abs() >= TOLERANCE).unwrap_or(0);
        let last_bin = (first_bin..n_bins)
            .rev()
            .find(|&i| p2[i].abs() >= TOLERANCE)
            .unwrap_or(n_bins - 1);

        let mut min_ent = f64::MAX;
        let mut threshold = first_bin;
        for it in first_bin..=last_bin {
            let mut ent_back = 0.0_f64;
            let mut term = 0.5 / p1[it];
            for ih in 1..=it {
                ent_back -= norm[ih] * (1.0 - term * p1[ih - 1]).ln();
            }
            ent_back *= term;

            let mut ent_obj = 0.0_f64;
            term = 0.5 / p2[it];
            for ih in (it + 1)..n_bins {
                ent_obj -= norm[ih] * (1.0 - term * p2[ih]).ln();
            }
            ent_obj *= term;

            let tot_ent = (ent_back - ent_obj).abs();
            if tot_ent < min_ent {
                min_ent = tot_ent;
                threshold = it;
            }
        }

        bin_center(x_min, bw, threshold)
    }
}

/// Convenience function: compute the Shanbhag threshold with 256 bins.
pub fn shanbhag_threshold<B: Backend, const D: usize>(image: &Image<f32, B, D>) -> f32 {
    ShanbhagThreshold::new().compute(image)
}

/// Compute the Shanbhag threshold directly from a flat `&[f32]` slice.
pub fn compute_shanbhag_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&ShanbhagThreshold::with_bins(num_bins), slice)
}
