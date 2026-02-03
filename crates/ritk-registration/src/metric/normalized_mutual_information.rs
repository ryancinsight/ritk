//! Normalized Mutual Information metric.
//!
//! This module provides normalized mutual information
//! based on differentiable soft histogramming.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use crate::metric::{Metric, histogram::ParzenJointHistogram};
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_core::interpolation::LinearInterpolator;
use std::marker::PhantomData;

/// Normalized Mutual Information Metric.
///
/// Computes normalized mutual information between two images.
/// Normalization ensures the metric is in the range [0, 1] (or similar, depending on method).
///
/// NMI = (H(X) + H(Y)) / H(X,Y)  (using JointEntropy method)
/// Returns negative NMI as loss.
#[derive(Clone, Debug)]
pub struct NormalizedMutualInformation<B: Backend> {
    /// Histogram calculator
    histogram_calculator: ParzenJointHistogram<B>,
    /// Sampling percentage (0.0 to 1.0) for stochastic optimization
    sampling_percentage: Option<f32>,
    /// Normalization method
    normalization_method: NormalizationMethod,
    /// Interpolator
    interpolator: LinearInterpolator,
    /// Phantom data
    _phantom: PhantomData<B>,
}

/// Normalization method for NMI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Normalize by joint entropy: (H(X) + H(Y)) / H(X,Y)
    JointEntropy,
    /// Normalize by average of marginal entropies: 2 * MI / (H(X) + H(Y))
    AverageEntropy,
    /// Normalize by minimum of marginal entropies: MI / min(H(X), H(Y))
    MinEntropy,
    /// Normalize by maximum of marginal entropies: MI / max(H(X), H(Y))
    MaxEntropy,
}

impl<B: Backend> NormalizedMutualInformation<B> {
    /// Create a new Normalized Mutual Information metric.
    ///
    /// # Arguments
    /// * `num_bins` - Number of histogram bins
    /// * `min_intensity` - Minimum intensity value
    /// * `max_intensity` - Maximum intensity value
    /// * `parzen_sigma` - Parzen window sigma for smoothing
    /// * `normalization_method` - Normalization method
    pub fn new(
        num_bins: usize,
        min_intensity: f32,
        max_intensity: f32,
        parzen_sigma: f32,
        normalization_method: NormalizationMethod,
    ) -> Self {
        Self {
            histogram_calculator: ParzenJointHistogram::new(num_bins, min_intensity, max_intensity, parzen_sigma),
            sampling_percentage: None,
            normalization_method,
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
        Self::new(32, 0.0, 255.0, 1.0, NormalizationMethod::JointEntropy)
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for NormalizedMutualInformation<B> {
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

        // 3. Marginals
        let p_x = p_xy.clone().sum_dim(1).squeeze(1); // P(fixed)
        let p_y = p_xy.clone().sum_dim(0).squeeze(0); // P(moving)

        // 4. Entropies
        let h_x = self.histogram_calculator.compute_entropy(p_x);
        let h_y = self.histogram_calculator.compute_entropy(p_y);
        
        // Joint Entropy H(X,Y)
        let eps = 1e-10;
        let log_p_xy = (p_xy.clone() + eps).log();
        let h_xy = p_xy.mul(log_p_xy).sum().neg();

        // 5. Compute NMI based on method
        let nmi = match self.normalization_method {
            NormalizationMethod::JointEntropy => {
                // (H(X) + H(Y)) / H(X,Y)
                (h_x.clone() + h_y.clone()) / (h_xy + eps)
            }
            NormalizationMethod::AverageEntropy => {
                // 2 * MI / (H(X) + H(Y))
                // MI = H(X) + H(Y) - H(X,Y)
                let mi = h_x.clone() + h_y.clone() - h_xy;
                (mi * 2.0) / (h_x + h_y + eps)
            }
            NormalizationMethod::MinEntropy => {
                // MI / min(H(X), H(Y))
                let mi = h_x.clone() + h_y.clone() - h_xy;
                // min(a, b) = 0.5 * (a + b - |a - b|)
                let sum_h = h_x.clone() + h_y.clone();
                let diff_h = (h_x - h_y).abs();
                let min_h = (sum_h - diff_h) * 0.5;
                mi / (min_h + eps)
            }
            NormalizationMethod::MaxEntropy => {
                // MI / max(H(X), H(Y))
                let mi = h_x.clone() + h_y.clone() - h_xy;
                // max(a, b) = 0.5 * (a + b + |a - b|)
                let sum_h = h_x.clone() + h_y.clone();
                let diff_h = (h_x - h_y).abs();
                let max_h = (sum_h + diff_h) * 0.5;
                mi / (max_h + eps)
            }
        };

        // Return negative NMI (loss)
        nmi.neg()
    }

    fn name(&self) -> &'static str {
        "Normalized Mutual Information"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_nmi_creation() {
        let metric = NormalizedMutualInformation::<TestBackend>::default_params();
        assert_eq!(metric.histogram_calculator.num_bins, 32);
        assert_eq!(metric.histogram_calculator.min_intensity, 0.0);
        assert_eq!(metric.histogram_calculator.max_intensity, 255.0);
        assert_eq!(metric.histogram_calculator.parzen_sigma, 1.0);
        assert_eq!(metric.normalization_method, NormalizationMethod::JointEntropy);
    }

    #[test]
    fn test_nmi_name() {
        let metric = NormalizedMutualInformation::<TestBackend>::default_params();
        assert_eq!(<NormalizedMutualInformation<TestBackend> as Metric<TestBackend, 3>>::name(&metric), "Normalized Mutual Information");
    }

    #[test]
    fn test_normalization_methods() {
        let metric1 = NormalizedMutualInformation::<TestBackend>::new(
            32, 0.0, 255.0, 1.0, NormalizationMethod::JointEntropy,
        );
        let metric2 = NormalizedMutualInformation::<TestBackend>::new(
            32, 0.0, 255.0, 1.0, NormalizationMethod::AverageEntropy,
        );
        let metric3 = NormalizedMutualInformation::<TestBackend>::new(
            32, 0.0, 255.0, 1.0, NormalizationMethod::MinEntropy,
        );
        let metric4 = NormalizedMutualInformation::<TestBackend>::new(
            32, 0.0, 255.0, 1.0, NormalizationMethod::MaxEntropy,
        );

        assert_eq!(metric1.normalization_method, NormalizationMethod::JointEntropy);
        assert_eq!(metric2.normalization_method, NormalizationMethod::AverageEntropy);
        assert_eq!(metric3.normalization_method, NormalizationMethod::MinEntropy);
        assert_eq!(metric4.normalization_method, NormalizationMethod::MaxEntropy);
    }
    
    #[test]
    fn test_with_sampling() {
        let metric = NormalizedMutualInformation::<TestBackend>::default_params().with_sampling(0.5);
        assert_eq!(metric.sampling_percentage, Some(0.5));
        
        let metric_invalid = NormalizedMutualInformation::<TestBackend>::default_params().with_sampling(1.5);
        assert_eq!(metric_invalid.sampling_percentage, None);
    }
}
