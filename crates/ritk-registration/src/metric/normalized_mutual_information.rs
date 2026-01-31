//! Normalized Mutual Information metric.
//!
//! This module provides normalized mutual information
//! based on differentiable soft histogramming.

use burn::tensor::{Tensor, Int};
use burn::tensor::backend::Backend;
use crate::metric::{Metric, trait_::utils};
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
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
    /// Number of histogram bins
    num_bins: usize,
    /// Minimum intensity value
    min_intensity: f32,
    /// Maximum intensity value
    max_intensity: f32,
    /// Parzen window sigma for histogram smoothing
    parzen_sigma: f32,
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
            num_bins,
            min_intensity,
            max_intensity,
            parzen_sigma,
            normalization_method,
            interpolator: LinearInterpolator::new(),
            _phantom: PhantomData,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(32, 0.0, 255.0, 1.0, NormalizationMethod::JointEntropy)
    }

    /// Compute soft joint histogram between two images (vectorized).
    fn compute_joint_histogram(&self, fixed: &Tensor<B, 1>, moving: &Tensor<B, 1>) -> Tensor<B, 2> {
        let device = fixed.device();
        let n = fixed.dims()[0];
        
        // Normalize intensities to [0, num_bins-1]
        let normalize = |t: Tensor<B, 1>| -> Tensor<B, 1> {
            let t = t - self.min_intensity;
            let t = t / (self.max_intensity - self.min_intensity);
            let t = t * (self.num_bins as f32 - 1.0);
            t.clamp(0.0, self.num_bins as f32 - 1.0)
        };

        let fixed_norm = normalize(fixed.clone());
        let moving_norm = normalize(moving.clone());

        // Create bin centers [Bins]
        let bins = Tensor::<B, 1, Int>::arange(0..self.num_bins as i64, &device).float();
        
        // Vectorized Weight Computation
        // weights: [N, Bins]
        // Use Gaussian kernel for Parzen windowing:
        // W[i, b] = exp(-0.5 * ((val[i] - b) / sigma)^2)
        
        let compute_weights = |vals: Tensor<B, 1>| -> Tensor<B, 2> {
            let vals_exp = vals.reshape([n, 1]); // [N, 1]
            let bins_exp = bins.clone().reshape([1, self.num_bins]); // [1, Bins]
            
            let diff = vals_exp - bins_exp; // [N, Bins]
            let sigma_sq = self.parzen_sigma * self.parzen_sigma;
            let exponent = diff.powf_scalar(2.0) * (-0.5 / sigma_sq);
            exponent.exp()
        };
        
        let w_fixed = compute_weights(fixed_norm); // [N, Bins]
        let w_moving = compute_weights(moving_norm); // [N, Bins]
        
        // Joint Histogram = W_fixed^T * W_moving
        // [Bins, N] * [N, Bins] -> [Bins, Bins]
        w_fixed.transpose().matmul(w_moving)
    }

    /// Compute entropy from histogram (as probabilities).
    fn compute_entropy(&self, p: Tensor<B, 1>) -> Tensor<B, 1> {
        let eps = 1e-10;
        let log_p = (p.clone() + eps).log();
        p.mul(log_p).sum().neg()
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for NormalizedMutualInformation<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // 1. Sampling (same as other metrics)
        let fixed_shape = fixed.shape();
        let device = fixed.data().device();
        let fixed_indices = utils::generate_grid(fixed_shape, &device);
        let fixed_points = fixed.index_to_world_tensor(fixed_indices.clone());
        let moving_points = transform.transform_points(fixed_points);
        let moving_indices = moving.world_to_index_tensor(moving_points);
        let moving_values = self.interpolator.interpolate(moving.data(), moving_indices);
        let fixed_values = fixed.data().clone().reshape([fixed_indices.dims()[0]]);

        // 2. Compute Joint Histogram
        let joint_hist = self.compute_joint_histogram(&fixed_values, &moving_values);

        // 3. Normalize to PDF
        let sum = joint_hist.clone().sum();
        let p_xy = joint_hist / (sum.unsqueeze_dim(1) + 1e-10);

        // 4. Marginals
        let p_x = p_xy.clone().sum_dim(1).squeeze(1); // P(fixed)
        let p_y = p_xy.clone().sum_dim(0).squeeze(0); // P(moving)

        // 5. Entropies
        let h_x = self.compute_entropy(p_x);
        let h_y = self.compute_entropy(p_y);
        
        // Joint Entropy H(X,Y)
        let eps = 1e-10;
        let log_p_xy = (p_xy.clone() + eps).log();
        let h_xy = p_xy.mul(log_p_xy).sum().neg();

        // 6. Compute NMI based on method
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
        assert_eq!(metric.num_bins, 32);
        assert_eq!(metric.min_intensity, 0.0);
        assert_eq!(metric.max_intensity, 255.0);
        assert_eq!(metric.parzen_sigma, 1.0);
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
}
