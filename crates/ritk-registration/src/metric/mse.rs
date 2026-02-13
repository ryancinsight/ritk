//! Mean Squared Error metric implementation.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use ritk_core::image::Image;
use ritk_core::image::grid;
use ritk_core::transform::Transform;
use ritk_core::interpolation::{Interpolator, LinearInterpolator};
use super::trait_::Metric;

/// Mean Squared Error Metric.
///
/// Computes the mean squared difference between pixel intensities:
/// MSE = (1/N) * sum((Fixed(x) - Moving(T(x)))^2)
#[derive(Clone)]
pub struct MeanSquaredError {
    interpolator: LinearInterpolator,
}

impl MeanSquaredError {
    /// Create a new MSE metric.
    pub fn new() -> Self {
        Self {
            interpolator: LinearInterpolator::new(),
        }
    }
}

impl Default for MeanSquaredError {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend, const D: usize> Metric<B, D> for MeanSquaredError {
    fn forward(
        &self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: &impl Transform<B, D>,
    ) -> Tensor<B, 1> {
        // 1. Generate grid of points in fixed image space (indices).
        let fixed_shape = fixed.shape();
        let device = fixed.data().device();
        let fixed_indices = grid::generate_grid(fixed_shape, &device); // [N, D]
        let [n, _] = fixed_indices.dims();

        // 2. Transform and interpolate with chunking to avoid WGPU dispatch limits
        const CHUNK_SIZE: usize = 32768;

        let moving_values = if n <= CHUNK_SIZE {
            // Transform fixed indices to physical points
            let fixed_points = fixed.index_to_world_tensor(fixed_indices); // [N, D]
            
            // Apply Transform to get corresponding points in moving image physical space
            let moving_points = transform.transform_points(fixed_points); // [N, D]
            
            // Transform moving physical points to moving image indices
            let moving_indices = moving.world_to_index_tensor(moving_points); // [N, D]
            
            // Sample moving image at moving_indices
            self.interpolator.interpolate(moving.data(), moving_indices) // [N]
        } else {
            let num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
            let mut chunks = Vec::with_capacity(num_chunks);

            for i in 0..num_chunks {
                let start = i * CHUNK_SIZE;
                let end = std::cmp::min(start + CHUNK_SIZE, n);

                let chunk_indices = fixed_indices.clone().slice([start..end]);
                let chunk_fixed_points = fixed.index_to_world_tensor(chunk_indices);
                let chunk_moving_points = transform.transform_points(chunk_fixed_points);
                let chunk_moving_indices = moving.world_to_index_tensor(chunk_moving_points);
                let chunk_values = self.interpolator.interpolate(moving.data(), chunk_moving_indices);
                chunks.push(chunk_values);
            }
            Tensor::cat(chunks, 0)
        };

        // 3. Get fixed image values
        // Flatten fixed data to match grid order [N]
        let fixed_values = fixed.data().clone().reshape([n]);

        // 4. Calculate MSE
        let diff = moving_values - fixed_values;
        diff.powf_scalar(2.0).mean()
    }

    fn name(&self) -> &'static str {
        "MeanSquaredError"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_core::spatial::{Point3, Spacing3, Direction3};
    use ritk_core::transform::TranslationTransform;

    type B = NdArray<f32>;

    #[test]
    fn test_mse_identity() {
        let device = Default::default();

        // Create synthetic data: 5x5x5 cube with gradient
        let d = 5;
        let mut data_vec = Vec::with_capacity(d * d * d);
        for z in 0..d {
            for y in 0..d {
                for x in 0..d {
                    data_vec.push((x + y + z) as f32);
                }
            }
        }
        let data = Tensor::<B, 1>::from_floats(data_vec.as_slice(), &device)
            .reshape([d, d, d]);

        let origin = Point3::new([0.0, 0.0, 0.0]);
        let spacing = Spacing3::new([1.0, 1.0, 1.0]);
        let direction = Direction3::identity();

        let fixed = Image::new(data.clone(), origin, spacing, direction);
        let moving = Image::new(data.clone(), origin, spacing, direction);

        let t_tensor = Tensor::from_floats([0.0, 0.0, 0.0], &device);
        let transform = TranslationTransform::new(t_tensor);
        let metric = MeanSquaredError::new();

        let loss = metric.forward(&fixed, &moving, &transform);
        let loss_val = loss.into_scalar();

        assert!(loss_val < 1e-5, "MSE should be 0 for identical images, got {}", loss_val);
    }
}
