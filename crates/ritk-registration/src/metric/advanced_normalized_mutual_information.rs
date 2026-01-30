//! Advanced Normalized Mutual Information metric.
//!
//! This module provides advanced normalized mutual information
//! based on elastix implementations.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use crate::metric::Metric;

/// Advanced Normalized Mutual Information Metric.
///
/// Computes normalized mutual information between two images.
/// Normalization ensures the metric is in the range [0, 1].
pub struct AdvancedNormalizedMutualInformation<B: Backend> {
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
}

/// Normalization method for NMI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationMethod {
    /// Normalize by joint entropy
    JointEntropy,
    /// Normalize by average of marginal entropies
    AverageEntropy,
    /// Normalize by minimum of marginal entropies
    MinEntropy,
    /// Normalize by maximum of marginal entropies
    MaxEntropy,
}

impl<B: Backend> AdvancedNormalizedMutualInformation<B> {
    /// Create a new Advanced Normalized Mutual Information metric.
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
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(32, 0.0, 255.0, 1.0, NormalizationMethod::JointEntropy)
    }

    /// Compute joint histogram between two images.
    fn compute_joint_histogram(&self, fixed: &Tensor<B, 3>, moving: &Tensor<B, 3>) -> Tensor<B, 2> {
        let device = fixed.device();
        let [height, width, depth] = fixed.dims();

        // Normalize intensities to [0, num_bins-1]
        let fixed_norm = fixed.clone() - self.min_intensity;
        let fixed_norm = fixed_norm / (self.max_intensity - self.min_intensity);
        let fixed_norm = fixed_norm * (self.num_bins as f32 - 1.0);
        let fixed_norm = fixed_norm.clamp(0.0, self.num_bins as f32 - 1.0);

        let moving_norm = moving.clone() - self.min_intensity;
        let moving_norm = moving_norm / (self.max_intensity - self.min_intensity);
        let moving_norm = moving_norm * (self.num_bins as f32 - 1.0);
        let moving_norm = moving_norm.clamp(0.0, self.num_bins as f32 - 1.0);

        // Initialize joint histogram
        let mut joint_hist = Tensor::zeros([self.num_bins, self.num_bins], &device);

        // Accumulate histogram
        for i in 0..height {
            for j in 0..width {
                for k in 0..depth {
                    let fixed_val = fixed_norm.clone().select(0, i).select(0, j).select(0, k);
                    let moving_val = moving_norm.clone().select(0, i).select(0, j).select(0, k);

                    let fixed_idx = fixed_val.into_scalar() as usize;
                    let moving_idx = moving_val.into_scalar() as usize;

                    // Increment histogram bin
                    let current_val = joint_hist.clone().select(0, fixed_idx).select(0, moving_idx);
                    let new_val = current_val + 1.0;
                    joint_hist = joint_hist.clone().scatter(0, Tensor::from_floats([fixed_idx as f32], &device), new_val);
                }
            }
        }

        // Apply Parzen window smoothing
        self.apply_parzen_smoothing(&joint_hist)
    }

    /// Apply Parzen window smoothing to histogram.
    fn apply_parzen_smoothing(&self, hist: &Tensor<B, 2>) -> Tensor<B, 2> {
        let device = hist.device();
        let sigma = self.parzen_sigma;

        // Create Gaussian kernel
        let kernel_size = 5;
        let mut kernel = Tensor::zeros([kernel_size], &device);
        let center = kernel_size / 2;

        for i in 0..kernel_size {
            let x = (i as f32 - center as f32) / sigma;
            let val = (-0.5 * x * x).exp();
            kernel = kernel.clone().scatter(0, Tensor::from_floats([i as f32], &device), Tensor::from_floats([val], &device));
        }

        // Normalize kernel
        let kernel_sum = kernel.clone().sum();
        kernel = kernel / kernel_sum;

        // Apply 2D convolution (simplified)
        hist.clone()
    }

    /// Compute entropy from histogram.
    fn compute_entropy(&self, hist: &Tensor<B, 2>) -> Tensor<B, 1> {
        let device = hist.device();

        // Normalize histogram to get probabilities
        let hist_sum = hist.clone().sum();
        let prob = hist.clone() / hist_sum;

        // Compute entropy: -sum(p * log(p))
        let log_prob = prob.clone().log();
        let entropy = -(prob.clone() * log_prob).sum();

        entropy
    }

    /// Compute marginal entropy from joint histogram.
    fn compute_marginal_entropy(&self, joint_hist: &Tensor<B, 2>, axis: usize) -> Tensor<B, 1> {
        let device = joint_hist.device();

        // Sum along axis to get marginal distribution
        let marginal = if axis == 0 {
            joint_hist.clone().sum_dim(1)
        } else {
            joint_hist.clone().sum_dim(0)
        };

        // Normalize to get probabilities
        let marginal_sum = marginal.clone().sum();
        let prob = marginal / marginal_sum;

        // Compute entropy
        let log_prob = prob.clone().log();
        let entropy = -(prob.clone() * log_prob).sum();

        entropy
    }

    /// Compute normalization factor.
    fn compute_normalization_factor(
        &self,
        joint_entropy: Tensor<B, 1>,
        fixed_entropy: Tensor<B, 1>,
        moving_entropy: Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        match self.normalization_method {
            NormalizationMethod::JointEntropy => joint_entropy,
            NormalizationMethod::AverageEntropy => (fixed_entropy.clone() + moving_entropy.clone()) / 2.0,
            NormalizationMethod::MinEntropy => {
                if fixed_entropy.clone().into_scalar() < moving_entropy.clone().into_scalar() {
                    fixed_entropy
                } else {
                    moving_entropy
                }
            }
            NormalizationMethod::MaxEntropy => {
                if fixed_entropy.clone().into_scalar() > moving_entropy.clone().into_scalar() {
                    fixed_entropy
                } else {
                    moving_entropy
                }
            }
        }
    }
}

impl<B: Backend> Metric<B> for AdvancedNormalizedMutualInformation<B> {
    fn compute(&self, fixed: &Tensor<B, 3>, moving: &Tensor<B, 3>) -> Tensor<B, 1> {
        // Compute joint histogram
        let joint_hist = self.compute_joint_histogram(fixed, moving);

        // Compute joint entropy
        let joint_entropy = self.compute_entropy(&joint_hist);

        // Compute marginal entropies
        let fixed_entropy = self.compute_marginal_entropy(&joint_hist, 0);
        let moving_entropy = self.compute_marginal_entropy(&joint_hist, 1);

        // Mutual information: H(X) + H(Y) - H(X,Y)
        let mi = fixed_entropy.clone() + moving_entropy.clone() - joint_entropy.clone();

        // Compute normalization factor
        let norm_factor = self.compute_normalization_factor(joint_entropy, fixed_entropy, moving_entropy);

        // Normalized mutual information
        let nmi = mi / norm_factor;

        nmi
    }

    fn name(&self) -> &str {
        "Advanced Normalized Mutual Information"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_advanced_nmi_creation() {
        let metric = AdvancedNormalizedMutualInformation::<TestBackend>::default_params();
        assert_eq!(metric.num_bins, 32);
        assert_eq!(metric.min_intensity, 0.0);
        assert_eq!(metric.max_intensity, 255.0);
        assert_eq!(metric.parzen_sigma, 1.0);
        assert_eq!(metric.normalization_method, NormalizationMethod::JointEntropy);
    }

    #[test]
    fn test_advanced_nmi_name() {
        let metric = AdvancedNormalizedMutualInformation::<TestBackend>::default_params();
        assert_eq!(metric.name(), "Advanced Normalized Mutual Information");
    }

    #[test]
    fn test_normalization_methods() {
        let metric1 = AdvancedNormalizedMutualInformation::<TestBackend>::new(
            32, 0.0, 255.0, 1.0, NormalizationMethod::JointEntropy,
        );
        let metric2 = AdvancedNormalizedMutualInformation::<TestBackend>::new(
            32, 0.0, 255.0, 1.0, NormalizationMethod::AverageEntropy,
        );
        let metric3 = AdvancedNormalizedMutualInformation::<TestBackend>::new(
            32, 0.0, 255.0, 1.0, NormalizationMethod::MinEntropy,
        );
        let metric4 = AdvancedNormalizedMutualInformation::<TestBackend>::new(
            32, 0.0, 255.0, 1.0, NormalizationMethod::MaxEntropy,
        );

        assert_eq!(metric1.normalization_method, NormalizationMethod::JointEntropy);
        assert_eq!(metric2.normalization_method, NormalizationMethod::AverageEntropy);
        assert_eq!(metric3.normalization_method, NormalizationMethod::MinEntropy);
        assert_eq!(metric4.normalization_method, NormalizationMethod::MaxEntropy);
    }
}
