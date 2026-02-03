//! Mutual Information metric.
//!
//! This module provides mutual information metrics
//! based on differentiable soft histogramming.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_core::interpolation::LinearInterpolator;
use crate::metric::{Metric, histogram::ParzenJointHistogram};
use std::marker::PhantomData;

/// Mutual Information Metric.
///
/// Computes mutual information between two images using differentiable soft histogramming.
///
/// MI(X, Y) = H(X) + H(Y) - H(X, Y)
///
/// Returns negative MI as loss (to be minimized).
#[derive(Clone, Debug)]
pub struct MutualInformation<B: Backend> {
    /// Histogram calculator
    histogram_calculator: ParzenJointHistogram<B>,
    /// Sampling percentage (0.0 to 1.0) for stochastic optimization
    sampling_percentage: Option<f32>,
    /// Interpolator
    interpolator: LinearInterpolator,
    /// Phantom data
    _phantom: PhantomData<B>,
}

impl<B: Backend> MutualInformation<B> {
    /// Create a new Mutual Information metric.
    ///
    /// # Arguments
    /// * `num_bins` - Number of histogram bins
    /// * `min_intensity` - Minimum intensity value
    /// * `max_intensity` - Maximum intensity value
    /// * `parzen_sigma` - Parzen window sigma for smoothing (e.g. 1.0)
    pub fn new(num_bins: usize, min_intensity: f32, max_intensity: f32, parzen_sigma: f32) -> Self {
        Self {
            histogram_calculator: ParzenJointHistogram::new(num_bins, min_intensity, max_intensity, parzen_sigma),
            sampling_percentage: None,
            interpolator: LinearInterpolator::new(),
            _phantom: PhantomData,
        }
    }

    /// Set the sampling percentage for stochastic optimization.
    ///
    /// # Arguments
    /// * `percentage` - Percentage of pixels to sample (0.0 to 1.0)
    pub fn with_sampling(mut self, percentage: f32) -> Self {
        if percentage > 0.0 && percentage < 1.0 {
            self.sampling_percentage = Some(percentage);
        } else {
            self.sampling_percentage = None;
        }
        self
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(32, 0.0, 255.0, 1.0)
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for MutualInformation<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // 1. Compute Joint Histogram (using shared logic with chunking/sampling)
        let joint_hist = self.histogram_calculator.compute_image_joint_histogram(
            fixed,
            moving,
            transform,
            &self.interpolator,
            self.sampling_percentage,
        );
        
        // 2. Normalize to PDF
        let sum = joint_hist.clone().sum();
        let p_xy = joint_hist / (sum.unsqueeze_dim(1) + 1e-10);

        // 3. Compute Marginals
        let p_x = p_xy.clone().sum_dim(1).squeeze(1); // Sum over moving -> P(fixed)
        let p_y = p_xy.clone().sum_dim(0).squeeze(0); // Sum over fixed -> P(moving)

        // 4. Compute Entropies
        let h_x = self.histogram_calculator.compute_entropy(p_x);
        let h_y = self.histogram_calculator.compute_entropy(p_y);
        
        // Joint Entropy H(X,Y) = -sum(p_xy * log(p_xy))
        let eps = 1e-10;
        let log_p_xy = (p_xy.clone() + eps).log();
        let h_xy = p_xy.mul(log_p_xy).sum().neg();

        // 5. Mutual Information
        let mi = h_x + h_y - h_xy;

        // Return negative MI (loss)
        mi.neg()
    }

    fn name(&self) -> &'static str {
        "Mutual Information"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_mi_creation() {
        let metric = MutualInformation::<TestBackend>::default_params();
        assert_eq!(metric.histogram_calculator.num_bins, 32);
        assert_eq!(metric.histogram_calculator.min_intensity, 0.0);
        assert_eq!(metric.histogram_calculator.max_intensity, 255.0);
        assert_eq!(metric.histogram_calculator.parzen_sigma, 1.0);
    }

    #[test]
    fn test_mi_name() {
        let metric = MutualInformation::<TestBackend>::default_params();
        assert_eq!(<MutualInformation<TestBackend> as Metric<TestBackend, 3>>::name(&metric), "Mutual Information");
    }
}
