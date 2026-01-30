//! Nearest neighbor interpolation implementation.
//!
//! This module provides nearest neighbor interpolation for 2D and 3D data.

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use super::trait_::Interpolator;

/// Nearest Neighbor Interpolator.
///
/// Performs nearest neighbor interpolation (rounds to nearest integer coordinate).
pub struct NearestNeighborInterpolator;

impl NearestNeighborInterpolator {
    /// Create a new nearest neighbor interpolator.
    pub fn new() -> Self {
        Self
    }
}

impl Default for NearestNeighborInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for NearestNeighborInterpolator {
    fn interpolate<const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        if D == 3 {
            self.interpolate_3d(data, indices)
        } else if D == 2 {
            self.interpolate_2d(data, indices)
        } else {
            panic!("NearestNeighborInterpolator only supports 2D and 3D tensors");
        }
    }
}

impl NearestNeighborInterpolator {
    fn interpolate_3d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0];
        let d1 = shape.dims[1];
        let d2 = shape.dims[2];

        let x = indices.clone().slice([0..indices.dims()[0], 0..1]).squeeze::<1>(1);
        let y = indices.clone().slice([0..indices.dims()[0], 1..2]).squeeze::<1>(1);
        let z = indices.clone().slice([0..indices.dims()[0], 2..3]).squeeze::<1>(1);

        // Round to nearest integer
        let x_i = x.round().clamp(0.0, (d0 - 1) as f64).int();
        let y_i = y.round().clamp(0.0, (d1 - 1) as f64).int();
        let z_i = z.round().clamp(0.0, (d2 - 1) as f64).int();

        let stride_x = (d1 * d2) as i32;
        let stride_y = d2 as i32;
        let stride_z = 1;

        let idx = x_i * stride_x + y_i * stride_y + z_i * stride_z;
        let flat_data = data.clone().reshape([d0 * d1 * d2]);
        flat_data.gather(0, idx)
    }

    fn interpolate_2d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0];
        let d1 = shape.dims[1];

        let x = indices.clone().slice([0..indices.dims()[0], 0..1]).squeeze::<1>(1);
        let y = indices.clone().slice([0..indices.dims()[0], 1..2]).squeeze::<1>(1);

        // Round to nearest integer
        let x_i = x.round().clamp(0.0, (d0 - 1) as f64).int();
        let y_i = y.round().clamp(0.0, (d1 - 1) as f64).int();

        let stride_x = d1 as i32;
        let stride_y = 1;

        let idx = x_i * stride_x + y_i * stride_y;
        let flat_data = data.clone().reshape([d0 * d1]);
        flat_data.gather(0, idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_nearest_neighbor_interpolator_2d() {
        let device = Default::default();
        let data = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 1.0], [2.0, 3.0]],
            &device,
        );
        let interpolator = NearestNeighborInterpolator::new();

        // Test at integer coordinates
        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 0.0], [1.0, 1.0]],
            &device,
        );
        let values = interpolator.interpolate(&data, indices);
        let data_slice = values.to_data();
        let data_slice_ref = data_slice.as_slice::<f32>().unwrap();

        assert_eq!(data_slice_ref[0], 0.0);
        assert_eq!(data_slice_ref[1], 3.0);
    }

    #[test]
    fn test_nearest_neighbor_interpolator_rounding() {
        let device = Default::default();
        let data = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 1.0], [2.0, 3.0]],
            &device,
        );
        let interpolator = NearestNeighborInterpolator::new();

        // Test at half coordinates (should round to nearest)
        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[0.4, 0.4], [0.6, 0.6]],
            &device,
        );
        let values = interpolator.interpolate(&data, indices);
        let data_slice = values.to_data();
        let data_slice_ref = data_slice.as_slice::<f32>().unwrap();

        // 0.4 rounds to 0, 0.6 rounds to 1
        assert_eq!(data_slice_ref[0], 0.0);
        assert_eq!(data_slice_ref[1], 3.0);
    }
}
