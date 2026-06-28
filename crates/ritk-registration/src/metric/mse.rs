//! Mean Squared Error metric implementation.

use super::trait_::Metric;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_image::grid;
use ritk_image::Image;
use ritk_interpolation::{transform_and_interpolate, LinearInterpolator};
use ritk_transform::Transform;

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

        // 2. Transform, interpolate, and accumulate squared differences chunk-by-chunk
        // to avoid allocating massive concatenated tensors on the heap.
        let sum_sq_diff = if n <= ritk_wgpu_compat::WGPU_CHUNK_SIZE {
            // Transform fixed indices to physical points
            let fixed_points = fixed.index_to_world_tensor(fixed_indices); // [N, D]

            let m = transform_and_interpolate(fixed_points, transform, moving, &self.interpolator)
                .values;
            let f = fixed.data().clone().reshape([n]); // [N]

            let diff = m - f;
            diff.powf_scalar(2.0).sum()
        } else {
            let num_chunks = n.div_ceil(ritk_wgpu_compat::WGPU_CHUNK_SIZE);
            let mut acc_sq_diff = Tensor::zeros([1], &device);
            let fixed_values_flat = fixed.data().clone().reshape([n]);

            for i in 0..num_chunks {
                let start = i * ritk_wgpu_compat::WGPU_CHUNK_SIZE;
                let end = std::cmp::min(start + ritk_wgpu_compat::WGPU_CHUNK_SIZE, n);

                let chunk_range = start..end;
                let chunk_indices = fixed_indices.clone().slice([chunk_range.clone()]);
                let f = fixed_values_flat.clone().slice([chunk_range]);

                let chunk_fixed_points = fixed.index_to_world_tensor(chunk_indices);
                let m = transform_and_interpolate(
                    chunk_fixed_points,
                    transform,
                    moving,
                    &self.interpolator,
                )
                .values;

                let diff = m - f;
                acc_sq_diff = acc_sq_diff + diff.powf_scalar(2.0).sum();
            }
            acc_sq_diff
        };

        // 3. Return mean squared error
        sum_sq_diff / (n as f32)
    }

    fn name(&self) -> &'static str {
        "MeanSquaredError"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;
    use ritk_spatial::{Direction3, Point3, Spacing3};
    use ritk_transform::TranslationTransform;

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
        let data = Tensor::<B, 1>::from_floats(data_vec.as_slice(), &device).reshape([d, d, d]);

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

        assert!(
            loss_val < 1e-5,
            "MSE should be 0 for identical images, got {}",
            loss_val
        );
    }
}
