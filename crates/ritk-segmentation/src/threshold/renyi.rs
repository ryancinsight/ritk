//! Renyi-entropy thresholding (Kapur/Sahoo/Wong generalisation, Renyi 1961).
//!
//! Matches ITK's `RenyiEntropyThresholdCalculator`: compute the maximum-entropy
//! threshold at three Renyi orders (Î± = 1, Â½, 2), then combine them with
//! proximity-dependent weights.

use ritk_image::tensor::Backend;
use ritk_image::Image;

use super::auto_threshold::{bin_center, itk_bin_width, threshold_from_slice, AutoThreshold};

const TOLERANCE: f64 = 2.220446049250313e-16;

/// Renyi-entropy threshold.
#[derive(Debug, Clone)]
pub struct RenyiEntropyThreshold {
    /// Number of equally-spaced histogram bins. Default 256.
    pub num_bins: usize,
}

impl RenyiEntropyThreshold {
    /// Create a `RenyiEntropyThreshold` with 256 histogram bins.
    pub fn new() -> Self {
        Self { num_bins: 256 }
    }

    /// Create a `RenyiEntropyThreshold` with a custom number of histogram bins.
    ///
    /// # Panics
    /// Panics if `num_bins < 2`.
    pub fn with_bins(num_bins: usize) -> Self {
        assert!(num_bins >= 2, "num_bins must be ≥ 2");
        Self { num_bins }
    }

    /// Compute the Renyi-entropy threshold intensity for `image`.
    pub fn compute<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> f32 {
        <Self as AutoThreshold>::compute(self, image)
    }

    /// Apply the Renyi-entropy threshold to produce a binary mask.
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

impl Default for RenyiEntropyThreshold {
    fn default() -> Self {
        Self::new()
    }
}

/// Shannon-entropy (Î± = 1) maximum-entropy threshold.
fn max_entropy_a1(
    norm: &[f64],
    p1: &[f64],
    p2: &[f64],
    first: usize,
    last: usize,
    n: usize,
) -> usize {
    let mut threshold = 0;
    let mut max_ent = f64::MIN;
    for it in first..=last {
        let mut ent_back = 0.0;
        for &nh in norm[..=it].iter() {
            if nh > 0.0 {
                let x = nh / p1[it];
                ent_back -= x * x.ln();
            }
        }
        let mut ent_obj = 0.0;
        for &nh in norm[it + 1..n].iter() {
            if nh > 0.0 {
                let x = nh / p2[it];
                ent_obj -= x * x.ln();
            }
        }
        let tot = ent_back + ent_obj;
        if max_ent < tot {
            max_ent = tot;
            threshold = it;
        }
    }
    threshold
}

/// Renyi Î± = Â½ maximum-entropy threshold.
fn max_entropy_a_half(
    norm: &[f64],
    p1: &[f64],
    p2: &[f64],
    first: usize,
    last: usize,
    n: usize,
) -> usize {
    let term = 1.0 / (1.0 - 0.5);
    let mut threshold = 0;
    let mut max_ent = f64::MIN;
    for it in first..=last {
        let mut ent_back = 0.0;
        for &nh in norm[..=it].iter() {
            ent_back += (nh / p1[it]).sqrt();
        }
        let mut ent_obj = 0.0;
        for &nh in norm[it + 1..n].iter() {
            ent_obj += (nh / p2[it]).sqrt();
        }
        let product = ent_back * ent_obj;
        let tot = if product > 0.0 {
            term * product.ln()
        } else {
            0.0
        };
        if tot > max_ent {
            max_ent = tot;
            threshold = it;
        }
    }
    threshold
}

/// Renyi Î± = 2 maximum-entropy threshold.
fn max_entropy_a2(
    norm: &[f64],
    p1: &[f64],
    p2: &[f64],
    first: usize,
    last: usize,
    n: usize,
) -> usize {
    let term = 1.0 / (1.0 - 2.0);
    let mut threshold = 0;
    let mut max_ent = 0.0_f64;
    for it in first..=last {
        let mut ent_back = 0.0;
        for &nh in norm[..=it].iter() {
            let x = nh / p1[it];
            ent_back += x * x;
        }
        let mut ent_obj = 0.0;
        for &nh in norm[it + 1..n].iter() {
            let x = nh / p2[it];
            ent_obj += x * x;
        }
        let product = ent_back * ent_obj;
        let tot = if product > 0.0 {
            term * product.ln()
        } else {
            0.0
        };
        if tot > max_ent {
            max_ent = tot;
            threshold = it;
        }
    }
    threshold
}

impl AutoThreshold for RenyiEntropyThreshold {
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

        let mut p1 = vec![0.0_f64; n_bins];
        let mut p2 = vec![0.0_f64; n_bins];
        p1[0] = norm[0];
        p2[0] = 1.0 - p1[0];
        for i in 1..n_bins {
            p1[i] = p1[i - 1] + norm[i];
            p2[i] = 1.0 - p1[i];
        }
        let first = (0..n_bins).find(|&i| p1[i].abs() >= TOLERANCE).unwrap_or(0);
        let last = (first..n_bins)
            .rev()
            .find(|&i| p2[i].abs() >= TOLERANCE)
            .unwrap_or(n_bins - 1);

        // Three Renyi-order thresholds, sorted ascending.
        let mut t1 = max_entropy_a_half(&norm, &p1, &p2, first, last, n_bins);
        let mut t2 = max_entropy_a1(&norm, &p1, &p2, first, last, n_bins);
        let mut t3 = max_entropy_a2(&norm, &p1, &p2, first, last, n_bins);
        if t2 < t1 {
            std::mem::swap(&mut t1, &mut t2);
        }
        if t3 < t2 {
            std::mem::swap(&mut t2, &mut t3);
        }
        if t2 < t1 {
            std::mem::swap(&mut t1, &mut t2);
        }

        // Proximity-dependent combination weights.
        let d12 = (t1 as f64 - t2 as f64).abs() <= 5.0;
        let d23 = (t2 as f64 - t3 as f64).abs() <= 5.0;
        let (beta1, beta2, beta3) = match (d12, d23) {
            (true, true) => (1.0, 2.0, 1.0),
            (true, false) => (0.0, 1.0, 3.0),
            (false, true) => (3.0, 1.0, 0.0),
            (false, false) => (1.0, 2.0, 1.0),
        };

        let omega = p1[t3] - p1[t1];
        let real = t1 as f64 * (p1[t1] + 0.25 * omega * beta1)
            + t2 as f64 * 0.25 * omega * beta2
            + t3 as f64 * (p2[t3] + 0.25 * omega * beta3);
        let opt = (real.max(0.0) as usize).min(n_bins - 1);

        bin_center(x_min, bw, opt)
    }
}

/// Convenience function: compute the Renyi-entropy threshold with 256 bins.
pub fn renyi_entropy_threshold<B: Backend, const D: usize>(image: &Image<f32, B, D>) -> f32 {
    RenyiEntropyThreshold::new().compute(image)
}

/// Compute the Renyi-entropy threshold directly from a flat `&[f32]` slice.
pub fn compute_renyi_entropy_threshold_from_slice(slice: &[f32], num_bins: usize) -> f32 {
    threshold_from_slice(&RenyiEntropyThreshold::with_bins(num_bins), slice)
}
