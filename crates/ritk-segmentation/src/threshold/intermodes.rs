//! Intermodes thresholding (Prewitt & Mendelsohn 1966).
//!
//! Matches ITK's `IntermodesThresholdCalculator` (`UseInterMode = true`): smooth
//! the histogram with a 3-point running mean until it is bimodal (exactly two
//! local maxima), then take the average of the two mode positions.

use burn::tensor::backend::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

/// Maximum histogram-smoothing iterations (ITK default).
const MAX_SMOOTHING_ITERATIONS: usize = 10000;

/// Intermodes (average-of-modes) threshold.
#[derive(Debug, Clone)]
pub struct IntermodesThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl IntermodesThreshold {
    /// Create an `IntermodesThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create an `IntermodesThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the Intermodes threshold intensity for `image`.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Intermodes threshold to produce a binary mask.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        <Self as AutoThreshold>::apply(self, image)
    }
}

impl Default for IntermodesThreshold {
    fn default() -> Self {
        Self::new()
    }
}

/// Count local maxima; bimodal iff exactly two.
fn bimodal(h: &[f64]) -> bool {
    let mut modes = 0;
    for k in 1..h.len() - 1 {
        if h[k - 1] < h[k] && h[k + 1] < h[k] {
            modes += 1;
            if modes > 2 {
                return false;
            }
        }
    }
    modes == 2
}

impl AutoThreshold for IntermodesThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let bw = itk_bin_width(x_min, x_max, n_bins);
        let mut sm: Vec<f64> = hist.iter().map(|&c| c as f64).collect();

        // 3-point running-mean smoothing until the histogram is bimodal.
        let mut iters = 0;
        while !bimodal(&sm) {
            let mut previous;
            let mut current = 0.0_f64;
            let mut next = sm[0];
            for i in 0..sm.len() - 1 {
                previous = current;
                current = next;
                next = sm[i + 1];
                sm[i] = (previous + current + next) / 3.0;
            }
            let last = sm.len() - 1;
            sm[last] = (current + next) / 3.0;
            iters += 1;
            if iters > MAX_SMOOTHING_ITERATIONS {
                break;
            }
        }

        // Average of the two mode positions (UseInterMode = true).
        let mut tt = 0usize;
        for i in 1..sm.len() - 1 {
            if sm[i - 1] < sm[i] && sm[i + 1] < sm[i] {
                tt += i;
            }
        }
        tt /= 2;

        bin_center(x_min, bw, tt.min(n_bins - 1))
    }
}

/// Convenience function: compute the Intermodes threshold with 256 bins.
pub fn intermodes_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    IntermodesThreshold::new().compute(image)
}

/// Compute the Intermodes threshold directly from a flat `&[f32]` slice.
pub fn compute_intermodes_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&IntermodesThreshold::with_bins(num_bins), slice)
}
