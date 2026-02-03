//! Correlation Ratio metric.
//!
//! This module provides correlation ratio
//! based on elastix implementations.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_core::interpolation::LinearInterpolator;
use crate::metric::{Metric, histogram::ParzenJointHistogram};
use std::marker::PhantomData;

/// Correlation Ratio Metric.
///
/// Computes correlation ratio between two images using differentiable soft histogramming
/// (Parzen window / Linear Partial Volume Estimation).
/// This metric is asymmetric and measures how well one image
/// can predict the other.
#[derive(Clone, Debug)]
pub struct CorrelationRatio<B: Backend> {
    /// Histogram calculator
    histogram_calculator: ParzenJointHistogram<B>,
    /// Direction of correlation
    direction: CorrelationDirection,
    /// Sampling percentage (0.0 to 1.0)
    sampling_percentage: Option<f32>,
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

impl<B: Backend> CorrelationRatio<B> {
    /// Create a new Correlation Ratio metric.
    ///
    /// # Arguments
    /// * `num_bins` - Number of histogram bins
    /// * `min_intensity` - Minimum intensity value
    /// * `max_intensity` - Maximum intensity value
    /// * `parzen_sigma` - Parzen window sigma (e.g. 1.0)
    /// * `direction` - Direction of correlation
    pub fn new(
        num_bins: usize,
        min_intensity: f32,
        max_intensity: f32,
        parzen_sigma: f32,
        direction: CorrelationDirection,
    ) -> Self {
        Self {
            histogram_calculator: ParzenJointHistogram::new(num_bins, min_intensity, max_intensity, parzen_sigma),
            direction,
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
        Self::new(32, 0.0, 255.0, 1.0, CorrelationDirection::MovingGivenFixed)
    }

    /// Compute conditional mean.
    fn compute_conditional_mean(&self, joint_hist: &Tensor<B, 2>, axis: usize) -> Tensor<B, 1> {
        let device = joint_hist.device();
        let num_bins = self.histogram_calculator.num_bins;
        
        let indices = Tensor::arange(0..num_bins as i64, &device).float(); // [bins]
        
        if axis == 0 {
            // P(m|f) -> compute mean of m for each f
            let marginal = joint_hist.clone().sum_dim(1).squeeze(1); // [bins]
            
            // Weighted sum: sum(hist * j)
            let indices_2d = indices.unsqueeze_dim(0).repeat(&[num_bins, 1]); // [bins, bins]
            let weighted = joint_hist.clone().mul(indices_2d);
            let weighted_sum = weighted.sum_dim(1).squeeze(1); // [bins]
            
            // Avoid division by zero
            let mask = marginal.clone().equal_elem(0.0).float();
            let safe_marginal = marginal.clone() + mask;
            
            weighted_sum / safe_marginal
        } else {
            // Axis 1: Moving. E[X|Y=y].
            let marginal = joint_hist.clone().sum_dim(0).squeeze(0); // [bins]
            
            let indices_2d = indices.unsqueeze_dim(1).repeat(&[1, num_bins]); // [bins, bins] (col vector repeated)
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
        let num_bins = self.histogram_calculator.num_bins;
        let conditional_mean = self.compute_conditional_mean(joint_hist, axis);
        let indices = Tensor::arange(0..num_bins as i64, &device).float();
        
        if axis == 0 {
            // Var(Y|X=x) = E[(Y - E[Y|X])^2 | X=x]
            let marginal = joint_hist.clone().sum_dim(1).squeeze(1);
            
            // Expand mean to [bins, bins] (repeat across cols)
            let mean_2d = conditional_mean.unsqueeze_dim(1).repeat(&[1, num_bins]);
            
            // Expand indices to [bins, bins] (0..N across cols)
            let indices_2d = indices.unsqueeze_dim(0).repeat(&[num_bins, 1]);
            
            let diff = indices_2d - mean_2d;
            let diff_sq = diff.powf_scalar(2.0);
            
            let weighted = diff_sq.mul(joint_hist.clone());
            let sum_sq = weighted.sum_dim(1).squeeze(1);
            
            let mask = marginal.clone().equal_elem(0.0).float();
            let safe_marginal = marginal + mask;
            
            sum_sq / safe_marginal
        } else {
            let marginal = joint_hist.clone().sum_dim(0).squeeze(0);
            
            let mean_2d = conditional_mean.unsqueeze_dim(0).repeat(&[num_bins, 1]);
            let indices_2d = indices.unsqueeze_dim(1).repeat(&[1, num_bins]);
            
            let diff = indices_2d - mean_2d;
            let diff_sq = diff.powf_scalar(2.0);
            
            let weighted = diff_sq.mul(joint_hist.clone());
            let sum_sq = weighted.sum_dim(0).squeeze(0);
            
            let mask = marginal.clone().equal_elem(0.0).float();
            let safe_marginal = marginal + mask;
            
            sum_sq / safe_marginal
        }
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for CorrelationRatio<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // 1. Compute Joint Histogram (using shared logic)
        let joint_hist = self.histogram_calculator.compute_image_joint_histogram(
            fixed,
            moving,
            transform,
            &self.interpolator,
            self.sampling_percentage,
        );

        // 2. Compute Correlation Ratio
        // CR = 1 - Var(Y|X) / Var(Y)
        // or
        // CR = Var(E[Y|X]) / Var(Y)
        
        // Let's use 1 - sum(p(x) * var(y|x)) / var(y)
        
        // Normalize to PDF
        let sum = joint_hist.clone().sum();
        let p_xy = joint_hist / (sum.unsqueeze_dim(1) + 1e-10);
        
        match self.direction {
            CorrelationDirection::MovingGivenFixed => {
                // Fixed is X (rows), Moving is Y (cols)
                // We want CR(Y|X)
                
                // Var(Y)
                let p_y = p_xy.clone().sum_dim(0).squeeze(0);
                let indices = Tensor::arange(0..self.histogram_calculator.num_bins as i64, &p_y.device()).float();
                
                let mean_y = p_y.clone().mul(indices.clone()).sum();
                let mean_y_sq = p_y.clone().mul(indices.clone().powf_scalar(2.0)).sum();
                let var_y = mean_y_sq - mean_y.powf_scalar(2.0);
                
                // Conditional Variance Var(Y|X)
                let cond_var = self.compute_conditional_variance(&p_xy, 0);
                
                // P(X)
                let p_x = p_xy.clone().sum_dim(1).squeeze(1);
                
                // Expected conditional variance
                let expected_cond_var = p_x.mul(cond_var).sum();
                
                // CR = 1 - E[Var(Y|X)] / Var(Y)
                let cr = (expected_cond_var / (var_y + 1e-10)).neg().add_scalar(1.0);
                
                // Return negative CR (loss)
                cr.neg()
            }
            CorrelationDirection::FixedGivenMoving => {
                // Fixed is X (rows), Moving is Y (cols)
                // We want CR(X|Y)
                
                // Var(X)
                let p_x = p_xy.clone().sum_dim(1).squeeze(1);
                let indices = Tensor::arange(0..self.histogram_calculator.num_bins as i64, &p_x.device()).float();
                
                let mean_x = p_x.clone().mul(indices.clone()).sum();
                let mean_x_sq = p_x.clone().mul(indices.clone().powf_scalar(2.0)).sum();
                let var_x = mean_x_sq - mean_x.powf_scalar(2.0);
                
                // Conditional Variance Var(X|Y)
                let cond_var = self.compute_conditional_variance(&p_xy, 1);
                
                // P(Y)
                let p_y = p_xy.clone().sum_dim(0).squeeze(0);
                
                // Expected conditional variance
                let expected_cond_var = p_y.mul(cond_var).sum();
                
                // CR = 1 - E[Var(X|Y)] / Var(X)
                let cr = (expected_cond_var / (var_x + 1e-10)).neg().add_scalar(1.0);
                
                // Return negative CR (loss)
                cr.neg()
            }
        }
    }

    fn name(&self) -> &'static str {
        "Correlation Ratio"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_cr_creation() {
        let metric = CorrelationRatio::<TestBackend>::default_params();
        assert_eq!(metric.histogram_calculator.num_bins, 32);
        assert_eq!(metric.direction, CorrelationDirection::MovingGivenFixed);
    }
    
    #[test]
    fn test_cr_name() {
        let metric = CorrelationRatio::<TestBackend>::default_params();
        assert_eq!(<CorrelationRatio<TestBackend> as Metric<TestBackend, 3>>::name(&metric), "Correlation Ratio");
    }
}
