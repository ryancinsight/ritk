//! Correlation Ratio metric.
//!
//! This module provides correlation ratio
//! based on elastix implementations.

use crate::metric::sampling::SamplingConfig;
use crate::metric::{histogram::ParzenJointHistogram, Metric};
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_statistics::IntensityRange;
use ritk_image::Image;
use ritk_interpolation::LinearInterpolator;
use ritk_transform::Transform;

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
    /// Sampling configuration
    sampling: SamplingConfig,
    /// Interpolator for resampling
    interpolator: LinearInterpolator,
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
    /// * `range` - Validated intensity range `[min, max]`
    /// * `parzen_sigma` - Parzen window sigma (e.g. 1.0)
    /// * `direction` - Direction of correlation
    pub fn new(
        num_bins: usize,
        range: IntensityRange<f32>,
        parzen_sigma: f32,
        direction: CorrelationDirection,
        device: &B::Device,
    ) -> Self {
        Self {
            histogram_calculator: ParzenJointHistogram::new(
                num_bins,
                range.min(),
                range.max(),
                parzen_sigma,
                device,
            ),
            direction,
            sampling: SamplingConfig::uniform(1.0),
            interpolator: LinearInterpolator::new_zero_pad(),
        }
    }

    /// Set the sampling percentage for stochastic optimization.
    ///
    /// # Arguments
    /// * `percentage` - Percentage of pixels to sample (0.0 to 1.0)
    pub fn with_sampling(mut self, percentage: f32) -> Self {
        self.sampling = SamplingConfig::uniform(percentage);
        self
    }

    /// Create with default parameters.
    pub fn default_params(device: &B::Device) -> Self {
        Self::new(
            32,
            IntensityRange::new_unchecked(0.0_f32, 255.0_f32),
            1.0,
            CorrelationDirection::MovingGivenFixed,
            device,
        )
    }

    /// Compute a safe marginal PDF where zero entries are replaced by 1
    /// to avoid division by zero in conditional mean/variance.
    ///
    /// Pre-computing once per marginal and sharing across
    /// `compute_conditional_mean` and `compute_conditional_variance`
    /// eliminates redundant mask recomputation.
    fn safe_marginal(marginal: &Tensor<B, 1>) -> Tensor<B, 1> {
        let mask = marginal.clone().equal_elem(0.0).float();
        marginal.clone() + mask
    }

    /// Compute conditional mean.
    ///
    /// Accepts `joint_hist` by reference and a pre-computed `safe_marginal`
    /// to minimize tensor clones. A single `clone()` is unavoidable: Burn tensor
    /// operations consume `self`, so the weighted-sum `.mul()` requires an
    /// owned tensor.
    fn compute_conditional_mean(
        &self,
        joint_hist: &Tensor<B, 2>,
        axis: usize,
        safe_marginal: &Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let device = joint_hist.device();
        let num_bins = self.histogram_calculator.num_bins;
        let indices = Tensor::arange(0..num_bins as i64, &device).float(); // [bins]

        // Single clone: consumed by .mul below (Burn ops take ownership).
        let jh = joint_hist.clone();

        if axis == 0 {
            // P(m|f) -> compute mean of m for each f
            // Weighted sum: sum(hist * j)
            let indices_2d = indices.unsqueeze_dim(0).repeat(&[num_bins, 1]); // [bins, bins]
            let weighted = jh.mul(indices_2d);
            let weighted_sum = weighted.sum_dim(1).squeeze::<1>(); // [bins]

            weighted_sum / safe_marginal.clone()
        } else {
            // Axis 1: Moving. E[X|Y=y].
            let indices_2d = indices.unsqueeze_dim(1).repeat(&[1, num_bins]); // [bins, bins]
            let weighted = jh.mul(indices_2d);
            let weighted_sum = weighted.sum_dim(0).squeeze::<1>();

            weighted_sum / safe_marginal.clone()
        }
    }

    /// Compute conditional variance.
    ///
    /// Accepts `joint_hist` by reference and a pre-computed `safe_marginal`
    /// to minimize tensor clones. A single `clone()` is unavoidable: Burn tensor
    /// operations consume `self`, so the weighted-sum `.mul()` requires an
    /// owned tensor.
    fn compute_conditional_variance(
        &self,
        joint_hist: &Tensor<B, 2>,
        axis: usize,
        safe_marginal: &Tensor<B, 1>,
    ) -> Tensor<B, 1> {
        let device = joint_hist.device();
        let num_bins = self.histogram_calculator.num_bins;
        let conditional_mean = self.compute_conditional_mean(joint_hist, axis, safe_marginal);
        let indices = Tensor::arange(0..num_bins as i64, &device).float();

        // Single clone: consumed by .mul below (Burn ops take ownership).
        let jh = joint_hist.clone();

        if axis == 0 {
            // Var(Y|X=x) = E[(Y - E[Y|X])^2 | X=x]
            // Expand mean to [bins, bins] (repeat across cols)
            let mean_2d = conditional_mean.unsqueeze_dim(1).repeat(&[1, num_bins]);

            // Expand indices to [bins, bins] (0..N across cols)
            let indices_2d = indices.unsqueeze_dim(0).repeat(&[num_bins, 1]);

            let diff = indices_2d - mean_2d;
            let diff_sq = diff.powf_scalar(2.0);

            let weighted = diff_sq.mul(jh);
            let sum_sq = weighted.sum_dim(1).squeeze::<1>();

            sum_sq / safe_marginal.clone()
        } else {
            let mean_2d = conditional_mean.unsqueeze_dim(0).repeat(&[num_bins, 1]);
            let indices_2d = indices.unsqueeze_dim(1).repeat(&[1, num_bins]);

            let diff = indices_2d - mean_2d;
            let diff_sq = diff.powf_scalar(2.0);

            let weighted = diff_sq.mul(jh);
            let sum_sq = weighted.sum_dim(0).squeeze::<1>();

            sum_sq / safe_marginal.clone()
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
            self.sampling,
        );

        // 2. Compute Correlation Ratio
        // CR = 1 - Var(Y|X) / Var(Y)
        // or
        // CR = Var(E[Y|X]) / Var(Y)

        // Let's use 1 - sum(p(x) * var(y|x)) / var(y)

        // Normalize to PDF: clone joint_hist once for sum,
        // then consume original for division.
        let total = joint_hist.clone().sum();
        let p_xy = joint_hist / (total.unsqueeze_dim(1) + 1e-10);

        // Pre-compute both marginal PDFs from p_xy once.
        // p_xy.clone() feeds sum_dim which consumes self;
        // each branch only uses one marginal as primary and
        // the other for weighting, so clone twice total
        // (once per marginal) instead of inside each branch.
        let p_x = p_xy.clone().sum_dim(1).squeeze::<1>(); // rows  → P(X)
        let p_y = p_xy.clone().sum_dim(0).squeeze::<1>(); // cols  → P(Y)

        // Shared index vector for variance computation.
        let num_bins = self.histogram_calculator.num_bins;
        let indices = Tensor::arange(0..num_bins as i64, &p_xy.device()).float();

        // Pre-compute safe marginals (zero-mask applied once per axis)
        // to avoid redundant mask computation inside conditional mean/variance.
        let safe_p_x = Self::safe_marginal(&p_x);
        let safe_p_y = Self::safe_marginal(&p_y);

        match self.direction {
            CorrelationDirection::MovingGivenFixed => {
                // Fixed is X (rows), Moving is Y (cols)
                // We want CR(Y|X)

                // Var(Y): E[Y^2] - (E[Y])^2
                let mean_y = p_y.clone().mul(indices.clone()).sum();
                let indices_sq = indices.powf_scalar(2.0);
                let mean_y_sq = p_y.mul(indices_sq).sum();
                let var_y = mean_y_sq - mean_y.powf_scalar(2.0);

                // Conditional Variance Var(Y|X)
                let cond_var = self.compute_conditional_variance(&p_xy, 0, &safe_p_x);

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

                // Var(X): E[X^2] - (E[X])^2
                let mean_x = p_x.clone().mul(indices.clone()).sum();
                let indices_sq = indices.powf_scalar(2.0);
                let mean_x_sq = p_x.mul(indices_sq).sum();
                let var_x = mean_x_sq - mean_x.powf_scalar(2.0);

                // Conditional Variance Var(X|Y)
                let cond_var = self.compute_conditional_variance(&p_xy, 1, &safe_p_y);

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
#[path = "tests_correlation_ratio.rs"]
mod tests;
