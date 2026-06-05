//! Nearest neighbor interpolation implementation.
//!
//! This module provides nearest neighbor interpolation for 1D, 2D, 3D, and 4D data.

use super::dispatch::dispatch_nearest;
use super::trait_::Interpolator;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

/// Nearest Neighbor Interpolator.
///
/// Performs nearest neighbor interpolation (rounds to nearest integer coordinate).
pub struct NearestNeighborInterpolator {
    /// If `true`, samples outside the volume boundary return `0.0` instead of
    /// the nearest-edge value. Mirrors [`LinearInterpolator::zero_pad`].
    pub zero_pad: bool,
}

impl NearestNeighborInterpolator {
    /// Create a new nearest neighbor interpolator with edge-clamping (default).
    pub fn new() -> Self {
        Self { zero_pad: false }
    }

    /// Create a nearest neighbor interpolator that returns `0.0` for OOB samples.
    pub fn new_zero_pad() -> Self {
        Self { zero_pad: true }
    }

    /// Builder-style setter for `zero_pad`.
    pub fn with_zero_pad(mut self, zero_pad: bool) -> Self {
        self.zero_pad = zero_pad;
        self
    }
}

impl Default for NearestNeighborInterpolator {
    fn default() -> Self {
        Self::new()
    }
}

impl<B: Backend> Interpolator<B> for NearestNeighborInterpolator {
    fn interpolate<const D: usize>(
        &self,
        data: &Tensor<B, D>,
        indices: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        dispatch_nearest(data, indices, self.zero_pad)
    }
}

// ── Dimension-specific free functions ───────────────────────────────────

pub(crate) fn interpolate_1d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    zero_pad: bool,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // X

    // indices: (x)
    let n = indices.dims()[0];
    let x = indices.slice([0..n, 0..1]).flatten::<1>(0, 1);

    // floor(coord + 0.5) gives standard round-to-nearest behavior.
    let x_f = (x + 0.5).floor();
    let x_i = x_f.clone().clamp(0.0, (d0 - 1) as f64).int();

    let result = data.clone().reshape([d0]).gather(0, x_i);

    if zero_pad {
        let x_in = x_f.clone().equal(x_f.clamp(0.0, (d0 - 1) as f64)).float();
        result * x_in
    } else {
        result
    }
}

pub(crate) fn interpolate_2d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    zero_pad: bool,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // Y
    let d1 = shape.dims[1]; // X

    // indices: (x, y)
    let n = indices.dims()[0];
    let x = indices.clone().slice([0..n, 0..1]).flatten::<1>(0, 1);
    let y = indices.slice([0..n, 1..2]).flatten::<1>(0, 1);

    // floor(coord + 0.5) gives standard round-to-nearest behavior.
    let x_f = (x + 0.5).floor();
    let y_f = (y + 0.5).floor();

    let x_i = x_f.clone().clamp(0.0, (d1 - 1) as f64).int();
    let y_i = y_f.clone().clamp(0.0, (d0 - 1) as f64).int();

    // Strides for [Y, X]
    let stride_y = d1 as i32;
    let stride_x = 1;

    let idx = y_i * stride_y + x_i * stride_x;
    let flat_data = data.clone().reshape([d0 * d1]);
    let result = flat_data.gather(0, idx);

    if zero_pad {
        let x_in = x_f.clone().equal(x_f.clamp(0.0, (d1 - 1) as f64)).float();
        let y_in = y_f.clone().equal(y_f.clamp(0.0, (d0 - 1) as f64)).float();
        result * (x_in * y_in)
    } else {
        result
    }
}

pub(crate) fn interpolate_3d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    zero_pad: bool,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // Z
    let d1 = shape.dims[1]; // Y
    let d2 = shape.dims[2]; // X

    // indices: (x, y, z)
    let n = indices.dims()[0];
    let x = indices.clone().slice([0..n, 0..1]).flatten::<1>(0, 1);
    let y = indices.clone().slice([0..n, 1..2]).flatten::<1>(0, 1);
    let z = indices.slice([0..n, 2..3]).flatten::<1>(0, 1);

    // floor(coord + 0.5) gives standard round-to-nearest behavior.
    let x_f = (x + 0.5).floor();
    let y_f = (y + 0.5).floor();
    let z_f = (z + 0.5).floor();

    let x_i = x_f.clone().clamp(0.0, (d2 - 1) as f64).int();
    let y_i = y_f.clone().clamp(0.0, (d1 - 1) as f64).int();
    let z_i = z_f.clone().clamp(0.0, (d0 - 1) as f64).int();

    // Strides for [Z, Y, X]
    let stride_z = (d1 * d2) as i32;
    let stride_y = d2 as i32;
    let stride_x = 1;

    let idx = z_i * stride_z + y_i * stride_y + x_i * stride_x;
    let flat_data = data.clone().reshape([d0 * d1 * d2]);
    let result = flat_data.gather(0, idx);

    if zero_pad {
        // x_f.clone().equal(x_f.clamp(...)).float() → 1.0 if coord was in-bounds, 0.0 otherwise
        let x_in = x_f.clone().equal(x_f.clamp(0.0, (d2 - 1) as f64)).float();
        let y_in = y_f.clone().equal(y_f.clamp(0.0, (d1 - 1) as f64)).float();
        let z_in = z_f.clone().equal(z_f.clamp(0.0, (d0 - 1) as f64)).float();
        result * (x_in * y_in * z_in)
    } else {
        result
    }
}

pub(crate) fn interpolate_4d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    zero_pad: bool,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // W
    let d1 = shape.dims[1]; // Z
    let d2 = shape.dims[2]; // Y
    let d3 = shape.dims[3]; // X

    let n = indices.dims()[0];
    let x = indices.clone().slice([0..n, 0..1]).flatten::<1>(0, 1);
    let y = indices.clone().slice([0..n, 1..2]).flatten::<1>(0, 1);
    let z = indices.clone().slice([0..n, 2..3]).flatten::<1>(0, 1);
    let w = indices.slice([0..n, 3..4]).flatten::<1>(0, 1);

    // floor(coord + 0.5) gives standard round-to-nearest behavior.
    let x_f = (x + 0.5).floor();
    let y_f = (y + 0.5).floor();
    let z_f = (z + 0.5).floor();
    let w_f = (w + 0.5).floor();

    let x_i = x_f.clone().clamp(0.0, (d3 - 1) as f64).int();
    let y_i = y_f.clone().clamp(0.0, (d2 - 1) as f64).int();
    let z_i = z_f.clone().clamp(0.0, (d1 - 1) as f64).int();
    let w_i = w_f.clone().clamp(0.0, (d0 - 1) as f64).int();

    // Strides for [W, Z, Y, X]
    let stride_w = (d1 * d2 * d3) as i32;
    let stride_z = (d2 * d3) as i32;
    let stride_y = d3 as i32;
    let stride_x = 1;

    let idx = w_i * stride_w + z_i * stride_z + y_i * stride_y + x_i * stride_x;
    let flat_data = data.clone().reshape([d0 * d1 * d2 * d3]);
    let result = flat_data.gather(0, idx);

    if zero_pad {
        let x_in = x_f.clone().equal(x_f.clamp(0.0, (d3 - 1) as f64)).float();
        let y_in = y_f.clone().equal(y_f.clamp(0.0, (d2 - 1) as f64)).float();
        let z_in = z_f.clone().equal(z_f.clamp(0.0, (d1 - 1) as f64)).float();
        let w_in = w_f.clone().equal(w_f.clamp(0.0, (d0 - 1) as f64)).float();
        result * (x_in * y_in * z_in * w_in)
    } else {
        result
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
        let data = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
        let interpolator = NearestNeighborInterpolator::new();

        // Test at integer coordinates
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0], [1.0, 1.0]], &device);
        let values = interpolator.interpolate(&data, indices);
        let data_slice = values.to_data();
        let data_slice_ref = data_slice.as_slice::<f32>().unwrap();

        assert_eq!(data_slice_ref[0], 0.0);
        assert_eq!(data_slice_ref[1], 3.0);
    }

    #[test]
    fn test_nearest_neighbor_interpolator_rounding() {
        let device = Default::default();
        let data = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
        let interpolator = NearestNeighborInterpolator::new();

        // Test at half coordinates (should round to nearest)
        let indices = Tensor::<TestBackend, 2>::from_floats([[0.4, 0.4], [0.6, 0.6]], &device);
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
        let data = Tensor::<TestBackend, 2>::from_floats([[0.0, 1.0], [2.0, 3.0]], &device);
        let interpolator = NearestNeighborInterpolator::new();

        // indices: (x, y)
        // (1, 0) -> x=1, y=0. Should correspond to col 1, row 0. Value 1.0.
        let indices = Tensor::<TestBackend, 2>::from_floats([[1.0, 0.0]], &device);
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
        let val = interpolator
            .interpolate(&data, indices)
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert_eq!(val, 20.0);
    }

    #[test]
    fn test_nearest_neighbor_interpolator_4d() {
        let device = Default::default();
        let mut data_vec = vec![0.0; 16];
        data_vec[15] = 100.0; // Last element (1,1,1,1)

        let data = Tensor::<TestBackend, 4>::from_data(
            burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2, 2])),
            &device,
        );
        let interpolator = NearestNeighborInterpolator::new();

        let indices = Tensor::<TestBackend, 2>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
        let val = interpolator
            .interpolate(&data, indices)
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert_eq!(val, 100.0);
    }

    #[test]
    fn test_nearest_neighbor_zero_pad_3d_oob_returns_zero() {
        let device = Default::default();
        let data_vec = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let data = Tensor::<TestBackend, 3>::from_data(
            burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2])),
            &device,
        );
        let interp = NearestNeighborInterpolator::new_zero_pad();

        // Far outside: should be 0.0
        let oob = Tensor::<TestBackend, 2>::from_floats(
            [[-5.0, -5.0, -5.0], [10.0, 10.0, 10.0]],
            &device,
        );
        let result = interp.interpolate(&data, oob);
        let s = result.into_data().as_slice::<f32>().unwrap().to_vec();
        assert!(s[0].abs() < 1e-6, "OOB 3D should give 0, got {}", s[0]);
        assert!(s[1].abs() < 1e-6, "OOB 3D should give 0, got {}", s[1]);
    }

    #[test]
    fn test_nearest_neighbor_zero_pad_3d_inbounds_unchanged() {
        let device = Default::default();
        let data_vec = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let data = Tensor::<TestBackend, 3>::from_data(
            burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2, 2])),
            &device,
        );
        let interp = NearestNeighborInterpolator::new_zero_pad();

        // In-bounds corner at (0,0,0) should return data[0,0,0] = 10.0
        let corner = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0]], &device);
        let val = interp
            .interpolate(&data, corner)
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        assert!(
            (val - 10.0).abs() < 1e-5,
            "In-bounds corner should give 10.0, got {}",
            val
        );
    }

    #[test]
    fn test_nearest_neighbor_no_zero_pad_clamps_edge() {
        // Verify backward compat: without zero_pad, OOB clamps to edge.
        let device = Default::default();
        let data_vec = vec![1.0_f32, 2.0, 3.0, 4.0];
        let data = Tensor::<TestBackend, 2>::from_data(
            burn::tensor::TensorData::new(data_vec, burn::tensor::Shape::new([2, 2])),
            &device,
        );
        let interp = NearestNeighborInterpolator::new(); // zero_pad = false
        let oob = Tensor::<TestBackend, 2>::from_floats([[-100.0, -100.0]], &device);
        let val = interp
            .interpolate(&data, oob)
            .into_data()
            .as_slice::<f32>()
            .unwrap()[0];
        // Should clamp to (0,0) -> data[0,0] = 1.0
        assert!(
            (val - 1.0).abs() < 1e-5,
            "Edge clamp should give 1.0, got {}",
            val
        );
    }
}
