//! IsoData (Ridler–Calvard iterative intermeans) thresholding.
//!
//! Matches ITK's `IsoDataThresholdCalculator` / the Fiji Auto_Threshold
//! "IsoData" method: iteratively refine the threshold to the midpoint of the
//! background and foreground mean bin indices until it stops moving.

use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

/// IsoData (Ridler–Calvard) iterative-intermeans threshold.
#[derive(Debug, Clone)]
pub struct IsoDataThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl IsoDataThreshold {
    /// Create an `IsoDataThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create an `IsoDataThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the IsoData threshold intensity for `image`.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the IsoData threshold to produce a binary mask.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
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
        image: &ritk_image::native::Image<f32, B, D>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, D>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        <Self as AutoThreshold>::apply_native(self, image, backend)
    }
}

impl Default for IsoDataThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoThreshold for IsoDataThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    /// Ridler–Calvard iteration, faithful to `itk::IsoDataThresholdCalculator`.
    ///
    /// The means are computed over the bin-centre **measurements** (not indices),
    /// the background is `[0, pos]` (inclusive) and the foreground `[pos+1, N)`,
    /// and the algorithm returns the measurement of the first bin whose value
    /// reaches the midpoint of the two class means. Falls back to the histogram
    /// mean if no such bin exists.
    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let bw = itk_bin_width(x_min, x_max, n_bins);
        let meas = |i: usize| bin_center(x_min, bw, i) as f64;
        let eps = f64::EPSILON;

        let mut pos = 0usize;
        loop {
            // Advance to the next non-empty bin.
            match (pos..n_bins).find(|&i| hist[i] > 0) {
                Some(p) => pos = p,
                None => return histogram_mean(hist, &meas, n_bins) as f32,
            }

            let (mut l, mut totl) = (0.0_f64, 0.0_f64);
            for (i, &count) in hist[..=pos].iter().enumerate() {
                let f = count as f64;
                totl += f;
                l += meas(i) * f;
            }
            let (mut h, mut toth) = (0.0_f64, 0.0_f64);
            for (i, &count) in hist[pos + 1..n_bins].iter().enumerate() {
                let f = count as f64;
                toth += f;
                h += meas(i + pos + 1) * f;
            }

            if totl > eps && toth > eps {
                l /= totl;
                h /= toth;
                if meas(pos) >= (l + h) * 0.5 {
                    return meas(pos) as f32;
                }
            }
            pos += 1;
            if pos >= n_bins {
                return histogram_mean(hist, &meas, n_bins) as f32;
            }
        }
    }
}

/// Histogram mean over bin-centre measurements (ITK `Histogram::Mean`).
fn histogram_mean(hist: &[u32], meas: &impl Fn(usize) -> f64, n_bins: usize) -> f64 {
    let (mut sum, mut total) = (0.0_f64, 0.0_f64);
    for (i, &count) in hist[..n_bins].iter().enumerate() {
        let f = count as f64;
        total += f;
        sum += meas(i) * f;
    }
    if total > 0.0 {
        sum / total
    } else {
        0.0
    }
}

/// Convenience function: compute the IsoData threshold with 256 bins.
pub fn isodata_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    IsoDataThreshold::new().compute(image)
}

/// Compute the IsoData threshold directly from a flat `&[f32]` slice.
pub fn compute_isodata_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&IsoDataThreshold::with_bins(num_bins), slice)
}
