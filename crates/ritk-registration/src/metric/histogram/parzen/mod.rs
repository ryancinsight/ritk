use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use super::cache::HistogramCache;

pub(crate) mod compute;
pub(crate) mod oob;

#[cfg(test)]
mod tests;

pub(super) use oob::compute_oob_mask_3d;

/// Joint Histogram Calculator using Parzen windowing.
#[derive(Clone, Debug)]
pub struct ParzenJointHistogram<B: Backend> {
    /// Number of histogram bins
    pub num_bins: usize,
    /// Minimum intensity value (fixed-image axis)
    pub min_intensity: f32,
    /// Maximum intensity value (fixed-image axis)
    pub max_intensity: f32,
    /// Parzen window sigma for histogram smoothing (fixed-image axis)
    pub parzen_sigma: f32,
    /// Optional separate minimum intensity for the moving image.
    /// When `None`, falls back to `min_intensity` (shared-range behaviour).
    pub moving_min_intensity: Option<f32>,
    /// Optional separate maximum intensity for the moving image.
    /// When `None`, falls back to `max_intensity` (shared-range behaviour).
    pub moving_max_intensity: Option<f32>,
    /// Optional separate Parzen sigma for the moving image.
    /// When `None`, falls back to `parzen_sigma`.
    pub moving_parzen_sigma: Option<f32>,
    /// Cache for fixed image points to avoid recomputation
    pub(super) cache: Arc<Mutex<Option<HistogramCache<B>>>>,
    /// Phantom data
    _phantom: PhantomData<B>,
}

impl<B: Backend> ParzenJointHistogram<B> {
    /// Create a new Parzen Joint Histogram calculator.
    pub fn new(num_bins: usize, min_intensity: f32, max_intensity: f32, parzen_sigma: f32) -> Self {
        Self {
            num_bins,
            min_intensity,
            max_intensity,
            parzen_sigma,
            moving_min_intensity: None,
            moving_max_intensity: None,
            moving_parzen_sigma: None,
            cache: Arc::new(Mutex::new(None)),
            _phantom: PhantomData,
        }
    }

    /// Configure a separate intensity range for the moving image (elastix-style independent binning).
    ///
    /// When set, each axis of the joint histogram uses its own normalization:
    /// the fixed axis spans `[min_intensity, max_intensity]` and the moving axis
    /// spans `[moving_min, moving_max]`, giving each image the full bin resolution.
    ///
    /// `moving_parzen_sigma` is set to `(moving_max - moving_min).max(1e-6) / num_bins`
    /// (Mattes parameterization: sigma = bin_width).
    pub fn with_separate_moving_range(mut self, moving_min: f32, moving_max: f32) -> Self {
        let sigma = (moving_max - moving_min).max(1e-6) / self.num_bins as f32;
        self.moving_min_intensity = Some(moving_min);
        self.moving_max_intensity = Some(moving_max);
        self.moving_parzen_sigma = Some(sigma);
        self
    }

    /// Compute Entropy of a distribution P.
    pub fn compute_entropy(&self, p: Tensor<B, 1>) -> Tensor<B, 1> {
        let eps = 1e-10;
        let log_p = (p.clone() + eps).log();
        p.mul(log_p).sum().neg()
    }
}
