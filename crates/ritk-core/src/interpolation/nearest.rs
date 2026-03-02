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
        match D {
            4 => self.interpolate_4d(data, indices),
            3 => self.interpolate_3d(data, indices),
            2 => self.interpolate_2d(data, indices),
            1 => self.interpolate_1d(data, indices),
            _ => panic!("NearestNeighborInterpolator only supports 1D, 2D, 3D and 4D tensors"),
        }
    }
}

impl NearestNeighborInterpolator {
    fn interpolate_4d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // W
        let d1 = shape.dims[1]; // Z
        let d2 = shape.dims[2]; // Y
        let d3 = shape.dims[3]; // X

        // indices: (x, y, z, w)
        let x = indices.clone().slice([0..indices.dims()[0], 0..1]).squeeze::<1>(1);
        let y = indices.clone().slice([0..indices.dims()[0], 1..2]).squeeze::<1>(1);
        let z = indices.clone().slice([0..indices.dims()[0], 2..3]).squeeze::<1>(1);
        let w = indices.clone().slice([0..indices.dims()[0], 3..4]).squeeze::<1>(1);

        // Round to nearest integer and clamp
        let x_i = x.round().clamp(0.0, (d3 - 1) as f64).int();
        let y_i = y.round().clamp(0.0, (d2 - 1) as f64).int();
        let z_i = z.round().clamp(0.0, (d1 - 1) as f64).int();
        let w_i = w.round().clamp(0.0, (d0 - 1) as f64).int();

        // Strides for [W, Z, Y, X]
        let stride_w = (d1 * d2 * d3) as i32;
        let stride_z = (d2 * d3) as i32;
        let stride_y = d3 as i32;
        let stride_x = 1;

        let idx = w_i * stride_w + z_i * stride_z + y_i * stride_y + x_i * stride_x;
        let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);
        flat_data.gather(0, idx)
    }

    fn interpolate_3d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // Z
        let d1 = shape.dims[1]; // Y
        let d2 = shape.dims[2]; // X

        // indices: (x, y, z)
        let x = indices.clone().slice([0..indices.dims()[0], 0..1]).squeeze::<1>(1);
        let y = indices.clone().slice([0..indices.dims()[0], 1..2]).squeeze::<1>(1);
        let z = indices.clone().slice([0..indices.dims()[0], 2..3]).squeeze::<1>(1);

        // Round to nearest integer and clamp
        let x_i = x.round().clamp(0.0, (d2 - 1) as f64).int();
        let y_i = y.round().clamp(0.0, (d1 - 1) as f64).int();
        let z_i = z.round().clamp(0.0, (d0 - 1) as f64).int();

        // Strides for [Z, Y, X]
        let stride_z = (d1 * d2) as i32;
        let stride_y = d2 as i32;
        let stride_x = 1;

        let idx = z_i * stride_z + y_i * stride_y + x_i * stride_x;
        let flat_data = data.clone().reshape([d0 * d1 * d2]);
        flat_data.gather(0, idx)
    }

    fn interpolate_2d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // Y
        let d1 = shape.dims[1]; // X

        // indices: (x, y)
        let x = indices.clone().slice([0..indices.dims()[0], 0..1]).squeeze::<1>(1);
        let y = indices.clone().slice([0..indices.dims()[0], 1..2]).squeeze::<1>(1);

        // Round to nearest integer
        let x_i = x.round().clamp(0.0, (d1 - 1) as f64).int();
        let y_i = y.round().clamp(0.0, (d0 - 1) as f64).int();

        // Strides for [Y, X]
        let stride_y = d1 as i32;
        let stride_x = 1;

        let idx = y_i * stride_y + x_i * stride_x;
        let flat_data = data.clone().reshape([d0 * d1]);
        flat_data.gather(0, idx)
    }

    fn interpolate_1d<B: Backend, const D: usize>(&self, data: &Tensor<B, D>, indices: Tensor<B, 2>) -> Tensor<B, 1> {
        let shape = data.shape();
        let d0 = shape.dims[0]; // X

        // indices: (x)
        let x = indices.clone().slice([0..indices.dims()[0], 0..1]).squeeze::<1>(1);

        // Round to nearest integer
        let x_i = x.round().clamp(0.0, (d0 - 1) as f64).int();

        let idx = x_i;
        let flat_data = data.clone().reshape([d0]);
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

    #[test]
    fn test_nearest_neighbor_interpolator_2d_axes() {
        let device = Default::default();
        // data: [[0, 1],
        //        [2, 3]]
        // Y=0: 0, 1. Y=1: 2, 3.
        let data = Tensor::<TestBackend, 2>::from_floats(
            [[0.0, 1.0], [2.0, 3.0]],
            &device,
        );
        let interpolator = NearestNeighborInterpolator::new();

        // indices: (x, y)
        // (1, 0) -> x=1, y=0. Should correspond to col 1, row 0. Value 1.0.
        let indices = Tensor::<TestBackend, 2>::from_floats(
            [[1.0, 0.0]],
            &device,
        );
        let values = interpolator.interpolate(&data, indices);
        let val = values.into_data().as_slice::<f32>().unwrap()[0];

        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_nearest_neighbor_interpolator_1d() {
        let device = Default::default();
        let data = Tensor::<TestBackend, 1>::from_floats([10.0, 20.0, 30.0], &device);
        let interpolator = NearestNeighborInterpolator::new();

        // x=1.0 -> 20.0
        let indices = Tensor::<TestBackend, 2>::from_floats([[1.0]], &device);
        let val = interpolator.interpolate(&data, indices).into_data().as_slice::<f32>().unwrap()[0];
        assert_eq!(val, 20.0);
    }

    #[test]
    fn test_nearest_neighbor_interpolator_4d() {
        let device = Default::default();
        let mut data_vec = vec![0.0; 16];
        data_vec[15] = 100.0; // Last element (1,1,1,1)

        let data = Tensor::<TestBackend, 4>::from_data(
            burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2, 2])),
            &device
        );
        let interpolator = NearestNeighborInterpolator::new();

        let indices = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
        let val = interpolator.interpolate(&data, indices).into_data().as_slice::<f32>().unwrap()[0];
        assert_eq!(val, 100.0);
    }
}
