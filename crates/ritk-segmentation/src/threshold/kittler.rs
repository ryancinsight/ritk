//! Kittler–Illingworth minimum-error thresholding (Kittler & Illingworth 1986).
//!
//! Matches ITK's `KittlerIllingworthThresholdCalculator`: model each class as a
//! Gaussian and iteratively solve the quadratic that minimises the
//! classification error until the threshold stops moving.

use burn::tensor::backend::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

/// Kittler–Illingworth minimum-error threshold.
#[derive(Debug, Clone)]
pub struct KittlerIllingworthThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl KittlerIllingworthThreshold {
    /// Create a `KittlerIllingworthThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `KittlerIllingworthThreshold` with a custom number of bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the Kittler–Illingworth threshold intensity for `image`.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Kittler–Illingworth threshold to produce a binary mask.
    pub fn apply<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        <Self as AutoThreshold>::apply(self, image)
    }
}

impl Default for KittlerIllingworthThreshold {
    fn default() -> Self {
        Self::new()
    }
}

impl AutoThreshold for KittlerIllingworthThreshold {
    fn num_bins(&self) -> usize {
        self.num_bins
    }

    fn compute_threshold(&self, hist: &[u32], n_bins: usize, x_min: f32, x_max: f32) -> f32 {
        let bw = itk_bin_width(x_min, x_max, n_bins);
        let meas = |i: usize| bin_center(x_min, bw, i) as f64;
        let index_of = |v: f64| {
            (((v - x_min as f64) / bw).floor().max(0.0) as usize).min(n_bins - 1) as isize
        };
        let eps = f64::EPSILON;

        // Cumulative moment functions A (freq), B (meas·freq), C (meas²·freq).
        let mut a = vec![0.0_f64; n_bins];
        let mut b = vec![0.0_f64; n_bins];
        let mut c = vec![0.0_f64; n_bins];
        let (mut ka, mut kb, mut kc) = (0.0_f64, 0.0_f64, 0.0_f64);
        for i in 0..n_bins {
            let f = hist[i] as f64;
            let m = meas(i);
            ka += f;
            kb += m * f;
            kc += m * m * f;
            a[i] = ka;
            b[i] = kb;
            c[i] = kc;
        }
        let tot = ka;
        if tot == 0.0 {
            return x_min;
        }
        let (as1, bs1, cs1) = (a[n_bins - 1], b[n_bins - 1], c[n_bins - 1]);

        let mut threshold = index_of(kb / tot); // start at the histogram mean's bin
        let mut tprev: isize = -2;
        let mut iters = 0usize;

        while threshold != tprev {
            iters += 1;
            if iters > 10_000 {
                break;
            }
            let t = threshold.clamp(0, n_bins as isize - 1) as usize;
            let (at, bt, ct) = (a[t], b[t], c[t]);
            if at.abs() < eps || (as1 - at).abs() < eps {
                break;
            }
            let mu = bt / at;
            let nu = (bs1 - bt) / (as1 - at);
            let p = at / as1;
            let q = (as1 - at) / as1;
            let sigma2 = ct / at - mu * mu;
            let tau2 = (cs1 - ct) / (as1 - at) - nu * nu;
            if sigma2 < eps || tau2.abs() < eps || p.abs() < eps {
                break;
            }
            let w0 = 1.0 / sigma2 - 1.0 / tau2;
            let w1 = mu / sigma2 - nu / tau2;
            let w2 =
                mu * mu / sigma2 - nu * nu / tau2 + (sigma2 * (q * q) / (tau2 * (p * p))).log10();
            let sqterm = w1 * w1 - w0 * w2;
            if sqterm < eps {
                break;
            }
            if w0.abs() < eps {
                threshold = index_of(-w2 / w1);
            } else {
                tprev = threshold;
                let temp = (w1 + sqterm.sqrt()) / w0;
                if temp.is_nan() {
                    threshold = tprev;
                    break;
                }
                threshold = index_of(temp);
            }
        }

        bin_center(x_min, bw, (threshold.max(0) as usize).min(n_bins - 1))
    }
}

/// Convenience function: compute the Kittler–Illingworth threshold with 256 bins.
pub fn kittler_illingworth_threshold<B: Backend, const D: usize>(image: &Image<B, D>) -> f32 {
    KittlerIllingworthThreshold::new().compute(image)
}

/// Compute the Kittler–Illingworth threshold directly from a flat `&[f32]` slice.
pub fn compute_kittler_illingworth_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&KittlerIllingworthThreshold::with_bins(num_bins), slice)
}
