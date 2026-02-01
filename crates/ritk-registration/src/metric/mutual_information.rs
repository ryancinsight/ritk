//! Mutual Information metric.
//!
//! This module provides mutual information metrics
//! based on differentiable soft histogramming.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use super::trait_::{Metric, utils};
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
    /// Number of histogram bins
    num_bins: usize,
    /// Minimum intensity value
    min_intensity: f32,
    /// Maximum intensity value
    max_intensity: f32,
    /// Parzen window sigma for histogram smoothing (in bin units)
    parzen_sigma: f32,
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
            num_bins,
            min_intensity,
            max_intensity,
            parzen_sigma,
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

    /// Compute soft joint histogram between two images (vectorized).
    /// Uses linear kernel (triangle) for differentiability.
    fn compute_joint_histogram(&self, fixed: &Tensor<B, 1>, moving: &Tensor<B, 1>) -> Tensor<B, 2> {
        let device = fixed.device();
        let [n] = fixed.dims();
        
        // Normalize intensities to [0, num_bins-1]
        let normalize = |t: Tensor<B, 1>| -> Tensor<B, 1> {
            let t = t - self.min_intensity;
            let t = t / (self.max_intensity - self.min_intensity);
            let t = t * (self.num_bins as f32 - 1.0);
            t.clamp(0.0, self.num_bins as f32 - 1.0)
        };

        // Create bin centers [Bins]
        let bins = Tensor::<B, 1, burn::tensor::Int>::arange(0..self.num_bins as i64, &device).float();
        let bins_exp = bins.clone().reshape([1, self.num_bins]); // [1, Bins]
        let sigma_sq = self.parzen_sigma * self.parzen_sigma;

        // Vectorized Weight Computation
        // weights: [N, Bins]
        // Use Gaussian kernel for Parzen windowing:
        // W[i, b] = exp(-0.5 * ((val[i] - b) / sigma)^2)
        
        let compute_weights = |vals: Tensor<B, 1>, size: usize| -> Tensor<B, 2> {
            let vals_exp = vals.reshape([size, 1]); // [N, 1]
            
            let diff = vals_exp - bins_exp.clone(); // [N, Bins]
            let exponent = diff.powf_scalar(2.0) * (-0.5 / sigma_sq);
            exponent.exp()
        };
        
        // WGPU dispatch limit workaround
        // The matmul (Bins, N) * (N, Bins) reduces along N.
        // If N is too large, it exceeds dispatch limits.
        const CHUNK_SIZE: usize = 32768;

        if n <= CHUNK_SIZE {
            let fixed_norm = normalize(fixed.clone());
            let moving_norm = normalize(moving.clone());
            
            let w_fixed = compute_weights(fixed_norm, n);
            let w_moving = compute_weights(moving_norm, n);
            
            w_fixed.transpose().matmul(w_moving)
        } else {
            let mut joint_hist = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);
            let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);
                let current_chunk_size = end - start;
                
                let fixed_chunk = fixed.clone().slice([start..end]);
                let moving_chunk = moving.clone().slice([start..end]);

                let fixed_norm = normalize(fixed_chunk);
                let moving_norm = normalize(moving_chunk);

                let w_fixed = compute_weights(fixed_norm, current_chunk_size);
                let w_moving = compute_weights(moving_norm, current_chunk_size);
                
                joint_hist = joint_hist + w_fixed.transpose().matmul(w_moving);
            }
            joint_hist
        }
    }

    /// Compute Entropy of a distribution P.
    fn compute_entropy(&self, p: Tensor<B, 1>) -> Tensor<B, 1> {
        let eps = 1e-10;
        let log_p = (p.clone() + eps).log();
        let entropy = p.mul(log_p).sum().neg();
        entropy
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for MutualInformation<B> {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        let fixed_shape = fixed.shape();
        let device = fixed.data().device();
        
        // 1. Generate Coordinates (Full Grid or Random Sample)
        let (fixed_indices, n, use_sampling) = if let Some(p) = self.sampling_percentage {
            let total_voxels = fixed_shape.iter().product::<usize>();
            let num_samples = (total_voxels as f32 * p) as usize;
            let indices = utils::generate_random_points(fixed_shape, num_samples, &device);
            (indices, num_samples, true)
        } else {
            let indices = utils::generate_grid(fixed_shape, &device);
            let [n, _] = indices.dims();
            (indices, n, false)
        };
        
        // Use a chunk size that respects wgpu dispatch limits.
        // The limit is often 65535 for one dimension. 
        // We choose a safe size that works for all intermediate operations (transforms, interpolation).
        const CHUNK_SIZE: usize = 32768; 

        let joint_hist = if n <= CHUNK_SIZE {
            // Process all at once
            let fixed_points = fixed.index_to_world_tensor(fixed_indices.clone());
            let moving_points = transform.transform_points(fixed_points);
            let moving_indices = moving.world_to_index_tensor(moving_points);
            let moving_values = self.interpolator.interpolate(moving.data(), moving_indices);
            
            let fixed_values = if use_sampling {
                // If sampling, interpolate fixed values at random coordinates
                self.interpolator.interpolate(fixed.data(), fixed_indices)
            } else {
                // For full grid, we can slice directly (optimization)
                fixed.data().clone().reshape([n])
            };
            
            self.compute_joint_histogram(&fixed_values, &moving_values)
        } else {
            // Process in chunks
            let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut joint_hist_acc = Tensor::<B, 2>::zeros([self.num_bins, self.num_bins], &device);
            
            // Flatten fixed data once if not sampling
            let fixed_data_flat = if !use_sampling {
                Some(fixed.data().clone().reshape([n]))
            } else {
                None
            };

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);
                
                // Slice indices for this chunk
                let chunk_indices = fixed_indices.clone().slice([start..end]);
                
                // Pipeline for this chunk
                let chunk_fixed_points = fixed.index_to_world_tensor(chunk_indices.clone());
                let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                let chunk_moving_values = self.interpolator.interpolate(moving.data(), chunk_moving_indices);
                
                // Get fixed values
                let chunk_fixed_values = if use_sampling {
                    self.interpolator.interpolate(fixed.data(), chunk_indices)
                } else {
                    fixed_data_flat.as_ref().unwrap().clone().slice([start..end])
                };
                
                // Compute partial histogram
                let chunk_hist = self.compute_joint_histogram(&chunk_fixed_values, &chunk_moving_values);
                joint_hist_acc = joint_hist_acc + chunk_hist;
            }
            joint_hist_acc
        };

        // 3. Normalize to PDF
        let sum = joint_hist.clone().sum();
        let p_xy = joint_hist / (sum.unsqueeze_dim(1) + 1e-10);

        // 4. Compute Marginals
        let p_x = p_xy.clone().sum_dim(1).squeeze(1); // Sum over moving -> P(fixed)
        let p_y = p_xy.clone().sum_dim(0).squeeze(0); // Sum over fixed -> P(moving)

        // 5. Compute Entropies
        let h_x = self.compute_entropy(p_x);
        let h_y = self.compute_entropy(p_y);
        
        // Joint Entropy H(X,Y) = -sum(p_xy * log(p_xy))
        let eps = 1e-10;
        let log_p_xy = (p_xy.clone() + eps).log();
        let h_xy = p_xy.mul(log_p_xy).sum().neg();

        // 6. Mutual Information
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
        assert_eq!(metric.num_bins, 32);
        assert_eq!(metric.min_intensity, 0.0);
        assert_eq!(metric.max_intensity, 255.0);
        assert_eq!(metric.parzen_sigma, 1.0);
    }

    #[test]
    fn test_mi_name() {
        let metric = MutualInformation::<TestBackend>::default_params();
        assert_eq!(<MutualInformation<TestBackend> as Metric<TestBackend, 3>>::name(&metric), "Mutual Information");
    }
}
