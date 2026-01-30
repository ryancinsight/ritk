//! Advanced Correlation Ratio metric.
//!
//! This module provides advanced correlation ratio
//! based on elastix implementations.

use burn::tensor::{Tensor, Int};
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use super::trait_::{Metric, utils};
use std::marker::PhantomData;

/// Advanced Correlation Ratio Metric.
///
/// Computes correlation ratio between two images using differentiable soft histogramming
/// (Parzen window / Linear Partial Volume Estimation).
/// This metric is asymmetric and measures how well one image
/// can predict the other.
#[derive(Clone, Debug)]
pub struct AdvancedCorrelationRatio<B: Backend> {
    /// Number of histogram bins
    num_bins: usize,
    /// Minimum intensity value
    min_intensity: f32,
    /// Maximum intensity value
    max_intensity: f32,
    /// Direction of correlation
    direction: CorrelationDirection,
    /// Interpolator for resampling
    interpolator: LinearInterpolator,
    /// Phantom data for backend
    _phantom: PhantomData<B>,
}

/// Direction of correlation ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrelationDirection {
    /// Correlation of moving image given fixed image
    MovingGivenFixed,
    /// Correlation of fixed image given moving image
    FixedGivenMoving,
}

impl<B: Backend> AdvancedCorrelationRatio<B> {
    /// Create a new Advanced Correlation Ratio metric.
    ///
    /// # Arguments
    /// * `num_bins` - Number of histogram bins
    /// * `min_intensity` - Minimum intensity value
    /// * `max_intensity` - Maximum intensity value
    /// * `direction` - Direction of correlation
    pub fn new(
        num_bins: usize,
        min_intensity: f32,
        max_intensity: f32,
        direction: CorrelationDirection,
    ) -> Self {
        Self {
            num_bins,
            min_intensity,
            max_intensity,
            direction,
            interpolator: LinearInterpolator::new(),
            _phantom: PhantomData,
        }
    }

    /// Create with default parameters.
    pub fn default_params() -> Self {
        Self::new(32, 0.0, 255.0, CorrelationDirection::MovingGivenFixed)
    }

    /// Compute soft joint histogram between two images (flattened).
    /// Uses linear kernel (triangle) for differentiability.
    fn compute_joint_histogram(&self, fixed: &Tensor<B, 1>, moving: &Tensor<B, 1>) -> Tensor<B, 2> {
        // Normalize intensities to [0, num_bins-1]
        let normalize = |t: Tensor<B, 1>| -> Tensor<B, 1> {
            let t = t - self.min_intensity;
            let t = t / (self.max_intensity - self.min_intensity);
            let t = t * (self.num_bins as f32 - 1.0);
            t.clamp(0.0, self.num_bins as f32 - 1.0)
        };

        let fixed_norm = normalize(fixed.clone());
        let moving_norm = normalize(moving.clone());
        
        let n = fixed.dims()[0];
        let bins = Tensor::<B, 1, Int>::arange(0..self.num_bins as i64, &B::Device::default()).float();
        
        // Helper to compute weights [N, num_bins]
        // w[i, b] = max(0, 1 - |val[i] - bin[b]|)
        let compute_weights = |vals: Tensor<B, 1>| -> Tensor<B, 2> {
            let vals_exp = vals.reshape([n, 1]);
            let bins_exp = bins.clone().reshape([1, self.num_bins]);
            let dist = (vals_exp - bins_exp).abs();
            (dist.neg() + 1.0).clamp_min(0.0)
        };

        let w_fixed = compute_weights(fixed_norm);
        let w_moving = compute_weights(moving_norm);

        // Joint histogram H [num_bins, num_bins] = W_fixed^T * W_moving
        w_fixed.transpose().matmul(w_moving)
    }

    /// Compute marginal distribution from joint histogram.
    fn compute_marginal(&self, joint_hist: &Tensor<B, 2>, axis: usize) -> Tensor<B, 1> {
        if axis == 0 {
            joint_hist.clone().sum_dim(1).squeeze(1)
        } else {
            joint_hist.clone().sum_dim(0).squeeze(0)
        }
    }

    /// Compute conditional mean.
    fn compute_conditional_mean(&self, joint_hist: &Tensor<B, 2>, axis: usize) -> Tensor<B, 1> {
        let device = joint_hist.device();
        
        let indices = Tensor::arange(0..self.num_bins as i64, &device).float(); // [bins]
        
        if axis == 0 {
            // P(m|f) -> compute mean of m for each f
            let marginal = joint_hist.clone().sum_dim(1).squeeze(1); // [bins]
            
            // Weighted sum: sum(hist * j)
            let indices_2d = indices.unsqueeze_dim(0).repeat(&[self.num_bins, 1]); // [bins, bins]
            let weighted = joint_hist.clone().mul(indices_2d);
            let weighted_sum = weighted.sum_dim(1).squeeze(1); // [bins]
            
            // Avoid division by zero
            let mask = marginal.clone().equal_elem(0.0).float();
            let safe_marginal = marginal.clone() + mask;
            
            weighted_sum / safe_marginal
        } else {
            // Axis 1: Moving. E[X|Y=y].
            let marginal = joint_hist.clone().sum_dim(0).squeeze(0); // [bins]
            
            let indices_2d = indices.unsqueeze_dim(1).repeat(&[1, self.num_bins]); // [bins, bins] (col vector repeated)
            let weighted = joint_hist.clone().mul(indices_2d);
            let weighted_sum = weighted.sum_dim(0).squeeze(0);
            
            let mask = marginal.clone().equal_elem(0.0).float();
            let safe_marginal = marginal.clone() + mask;
            
            weighted_sum / safe_marginal
        }
    }

    /// Compute conditional variance.
    fn compute_conditional_variance(&self, joint_hist: &Tensor<B, 2>, axis: usize) -> Tensor<B, 1> {
        let device = joint_hist.device();
        let conditional_mean = self.compute_conditional_mean(joint_hist, axis);
        let indices = Tensor::arange(0..self.num_bins as i64, &device).float();
        
        if axis == 0 {
            // Var(Y|X=x) = E[(Y - E[Y|X])^2 | X=x]
            let marginal = joint_hist.clone().sum_dim(1).squeeze(1);
            
            // Expand mean to [bins, bins] (repeat across cols)
            let mean_2d = conditional_mean.unsqueeze_dim(1).repeat(&[1, self.num_bins]);
            
            // Expand indices to [bins, bins] (0..N across cols)
            let indices_2d = indices.unsqueeze_dim(0).repeat(&[self.num_bins, 1]);
            
            let diff = indices_2d - mean_2d;
            let diff_sq = diff.powf_scalar(2.0);
            
            let weighted = diff_sq.mul(joint_hist.clone());
            let sum_sq = weighted.sum_dim(1).squeeze(1);
            
            let mask = marginal.clone().equal_elem(0.0).float();
            let safe_marginal = marginal + mask;
            
            sum_sq / safe_marginal
        } else {
            let marginal = joint_hist.clone().sum_dim(0).squeeze(0);
            
            let mean_2d = conditional_mean.unsqueeze_dim(0).repeat(&[self.num_bins, 1]);
            let indices_2d = indices.unsqueeze_dim(1).repeat(&[1, self.num_bins]);
            
            let diff = indices_2d - mean_2d;
            let diff_sq = diff.powf_scalar(2.0);
            
            let weighted = diff_sq.mul(joint_hist.clone());
            let sum_sq = weighted.sum_dim(0).squeeze(0);
            
            let mask = marginal.clone().equal_elem(0.0).float();
            let safe_marginal = marginal + mask;
            
            sum_sq / safe_marginal
        }
    }

    /// Compute correlation ratio.
    fn compute_correlation_ratio(
        &self,
        joint_hist: &Tensor<B, 2>,
        axis: usize,
    ) -> Tensor<B, 1> {
        // eta^2 = Var(E[Y|X]) / Var(Y)
        // OR 1 - E[Var(Y|X)] / Var(Y)
        
        let marginal_x = self.compute_marginal(joint_hist, axis); // P(x) * N
        let cond_var = self.compute_conditional_variance(joint_hist, axis); // Var(Y|x)
        
        let total_count = joint_hist.clone().sum().into_scalar(); // N
        
        // Expected Conditional Variance: Sum(P(x) * Var(Y|x))
        let weighted_cond_var = marginal_x.mul(cond_var).sum();
        let expected_cond_var = weighted_cond_var / total_count;
        
        // Total Variance of Y
        let other_axis = if axis == 0 { 1 } else { 0 };
        let marginal_y = self.compute_marginal(joint_hist, other_axis);
        let indices = Tensor::arange(0..self.num_bins as i64, &joint_hist.device()).float();
        
        let sum_y = marginal_y.clone().mul(indices.clone()).sum();
        let mean_y = sum_y / total_count;
        
        let sum_sq_y = marginal_y.mul(indices.powf_scalar(2.0)).sum();
        let mean_sq_y = sum_sq_y / total_count;
        
        let var_y = mean_sq_y - mean_y.powf_scalar(2.0);
        
        // CR = 1 - E[Var]/Var
        let result = Tensor::from_floats([1.0], &joint_hist.device()) - (expected_cond_var / (var_y + 1e-6));
        
        result
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for AdvancedCorrelationRatio<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // 1. Generate grid of points in fixed image space (indices).
        let fixed_shape = fixed.shape();
        let device = fixed.data().device();
        let fixed_indices = utils::generate_grid(fixed_shape, &device); // [N, D]

        // 2. Transform fixed indices to physical points.
        let fixed_points = fixed.index_to_world_tensor(fixed_indices.clone()); // [N, D]

        // 3. Apply Transform to get corresponding points in moving image physical space.
        let moving_points = transform.transform_points(fixed_points); // [N, D]

        // 4. Transform moving physical points to moving image indices.
        let moving_indices = moving.world_to_index_tensor(moving_points); // [N, D]

        // 5. Sample moving image at moving_indices.
        let moving_values = self.interpolator.interpolate(moving.data(), moving_indices); // [N]

        // 6. Get fixed image values
        // Flatten fixed data to match grid order [N]
        let fixed_values = fixed.data().clone().reshape([fixed_indices.dims()[0]]);

        // 7. Compute Joint Histogram
        let joint_hist = self.compute_joint_histogram(&fixed_values, &moving_values);

        // 8. Compute CR based on direction
        let cr = match self.direction {
            CorrelationDirection::MovingGivenFixed => {
                self.compute_correlation_ratio(&joint_hist, 0)
            }
            CorrelationDirection::FixedGivenMoving => {
                self.compute_correlation_ratio(&joint_hist, 1)
            }
        };

        // Return negative CR (loss to minimize)
        cr.neg()
    }

    fn name(&self) -> &'static str {
        "Advanced Correlation Ratio"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_advanced_cr_creation() {
        let metric = AdvancedCorrelationRatio::<TestBackend>::default_params();
        assert_eq!(metric.num_bins, 32);
        assert_eq!(metric.min_intensity, 0.0);
        assert_eq!(metric.max_intensity, 255.0);
        assert_eq!(metric.direction, CorrelationDirection::MovingGivenFixed);
    }

    #[test]
    fn test_advanced_cr_name() {
        let metric = AdvancedCorrelationRatio::<TestBackend>::default_params();
        assert_eq!(<AdvancedCorrelationRatio<TestBackend> as Metric<TestBackend, 3>>::name(&metric), "Advanced Correlation Ratio");
    }

    #[test]
    fn test_correlation_directions() {
        let metric1 = AdvancedCorrelationRatio::<TestBackend>::new(
            32, 0.0, 255.0, CorrelationDirection::MovingGivenFixed,
        );
        let metric2 = AdvancedCorrelationRatio::<TestBackend>::new(
            32, 0.0, 255.0, CorrelationDirection::FixedGivenMoving,
        );

        assert_eq!(metric1.direction, CorrelationDirection::MovingGivenFixed);
        assert_eq!(metric2.direction, CorrelationDirection::FixedGivenMoving);
    }
}
