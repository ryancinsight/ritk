//! Huang fuzzy thresholding (Huang & Wang 1995).
//!
//! Matches ITK's `HuangThresholdCalculator`: choose the threshold that minimises
//! a fuzzy-membership entropy measure, where each class's membership decays with
//! the distance from its mean grey level.

use burn::tensor::backend::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

/// Huang fuzzy-entropy threshold.
#[derive(Debug, Clone)]
pub struct HuangThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl HuangThreshold {
    /// Create a `HuangThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `HuangThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the Huang threshold intensity for `image`.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Huang threshold to produce a binary mask.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        <Self as AutoThreshold>::apply(self, image)
    }
}

impl Default for HuangThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoThreshold for HuangThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let bw = itk_bin_width(x_min, x_max, n_bins);
        let meas = |i: usize| bin_center(x_min, bw, i) as f64;
        let index_of =
            |v: f64| (((v - x_min as f64) / bw).floor().max(0.0) as usize).min(n_bins - 1);

        // Non-empty bin range [first, last].
        let first = (0..n_bins).find(|&i| hist[i] > 0);
        let first = match first {
            Some(f) => f,
            None => return x_min,
        };
        let last = (0..n_bins).rev().find(|&i| hist[i] > 0).unwrap_or(first);
        if last == first {
            return meas(first) as f32;
        }

        // Cumulative frequency S and cumulative measurement·frequency W.
        let mut s = vec![0.0_f64; last + 1];
        let mut w = vec![0.0_f64; last + 1];
        s[0] = hist[0] as f64;
        for i in first.max(1)..=last {
            s[i] = s[i - 1] + hist[i] as f64;
            w[i] = w[i - 1] + meas(i) * hist[i] as f64;
        }

        // Fuzzy-membership entropy lookup table over the class span.
        let c = (last - first) as f64;
        let smu_len = last + 1 - first;
        let mut smu = vec![0.0_f64; smu_len];
        for (i, slot) in smu.iter_mut().enumerate().skip(1) {
            let mu = 1.0 / (1.0 + i as f64 / c);
            *slot = -mu * mu.ln() - (1.0 - mu) * (1.0 - mu).ln();
        }
        let smu_at = |d: usize| smu[d.min(smu_len - 1)];

        let mut best_threshold = first;
        let mut best_entropy = f64::MAX;
        for threshold in first..last {
            let mut entropy = 0.0_f64;
            // Background class mean → its bin index.
            let mu = (w[threshold] / s[threshold]).round();
            let mu_idx = index_of(mu) as isize;
            for (i, &count) in hist[first..=threshold].iter().enumerate() {
                let i = i + first;
                let d = (i as isize - mu_idx).unsigned_abs();
                entropy += smu_at(d) * count as f64;
            }
            // Foreground class mean → its bin index.
            let mu2 = ((w[last] - w[threshold]) / (s[last] - s[threshold])).round();
            let mu2_idx = index_of(mu2) as isize;
            for (i, &count) in hist[threshold + 1..=last].iter().enumerate() {
                let i = i + threshold + 1;
                let d = (i as isize - mu2_idx).unsigned_abs();
                entropy += smu_at(d) * count as f64;
            }
            if entropy < best_entropy {
                best_entropy = entropy;
                best_threshold = threshold;
            }
        }

        bin_center(x_min, bw, best_threshold)
    }
}

/// Convenience function: compute the Huang threshold with 256 bins.
pub fn huang_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    HuangThreshold::new().compute(image)
}

/// Compute the Huang threshold directly from a flat `&[f32]` slice.
pub fn compute_huang_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&HuangThreshold::with_bins(num_bins), slice)
}
