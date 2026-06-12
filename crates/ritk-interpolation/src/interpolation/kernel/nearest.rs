//! Nearest neighbor interpolation implementation.
//!
//! This module provides nearest neighbor interpolation for 1D, 2D, 3D, and 4D data.

use super::BoundsPolicy;
use crate::interpolation::dispatch::dispatch_nearest;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;
use ritk_core::interpolation::Interpolator;

use crate::interpolation::shared::{in_bounds_mask, OutOfBoundsMode};

/// Nearest Neighbor Interpolator.
///
/// Performs nearest neighbor interpolation (rounds to nearest integer coordinate).
///
/// When [`BoundsPolicy::Extend`] (the default), out-of-bounds samples clamp to the nearest
/// edge voxel. When [`BoundsPolicy::ZeroPad`], out-of-bounds samples return `0.0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct NearestNeighborInterpolator {
    /// Boundary handling policy. Default: `Extend`.
    pub bounds_policy: BoundsPolicy,
}

impl NearestNeighborInterpolator {
    /// Create a new nearest neighbor interpolator with edge-clamping (default).
    pub fn new() -> Self {
        Self {
            bounds_policy: BoundsPolicy::Extend,
        }
    }

    /// Create a nearest neighbor interpolator that returns `0.0` for OOB samples.
    pub fn new_zero_pad() -> Self {
        Self {
            bounds_policy: BoundsPolicy::ZeroPad,
        }
    }

    /// Builder-style setter for the bounds policy.
    pub fn with_bounds_policy(mut self, policy: BoundsPolicy) -> Self {
        self.bounds_policy = policy;
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
        dispatch_nearest(data, indices, self.bounds_policy.as_out_of_bounds_mode())
    }
}

// ── Dimension-specific free functions ───────────────────────────────────

pub(crate) fn interpolate_1d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // X

    // indices: (x)
    let n = indices.dims()[0];
    let x = indices.slice([0..n, 0..1]).flatten::<1>(0, 1);

    // floor(coord + 0.5) gives standard round-to-nearest behavior.
    let x_f = (x + 0.5).floor();
    let x_i = x_f.clone().clamp(0.0, (d0 - 1) as f64).int();

    // reshape consumes self, gather consumes self — chain directly since only one gather.
    let result = data.clone().reshape([d0]).gather(0, x_i);

    if let Some(mask) = in_bounds_mask(x_f, (d0 - 1) as f64, mode) {
        result * mask
    } else {
        result
    }
}

pub(crate) fn interpolate_2d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // Y
    let d1 = shape.dims[1]; // X

    // indices: (x, y)
    let n = indices.dims()[0];
    // slice consumes self, so clone indices once and slice each column.
    let indices_local = indices;
    let x = indices_local.clone().slice([0..n, 0..1]).flatten::<1>(0, 1);
    let y = indices_local.slice([0..n, 1..2]).flatten::<1>(0, 1);

    // floor(coord + 0.5) gives standard round-to-nearest behavior.
    let x_f = (x + 0.5).floor();
    let y_f = (y + 0.5).floor();
    let x_i = x_f.clone().clamp(0.0, (d1 - 1) as f64).int();
    let y_i = y_f.clone().clamp(0.0, (d0 - 1) as f64).int();

    // Strides for [Y, X]
    let stride_y = d1 as i32;
    let stride_x = 1;
    let idx = y_i * stride_y + x_i * stride_x;
    // reshape consumes self, gather consumes self — chain directly since only one gather.
    let result = data.clone().reshape([d0 * d1]).gather(0, idx);

    let x_mask = in_bounds_mask(x_f, (d1 - 1) as f64, mode);
    let y_mask = in_bounds_mask(y_f, (d0 - 1) as f64, mode);

    match (x_mask, y_mask) {
        (Some(xm), Some(ym)) => result * xm * ym,
        _ => result,
    }
}

pub(crate) fn interpolate_3d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // Z
    let d1 = shape.dims[1]; // Y
    let d2 = shape.dims[2]; // X

    // indices: (x, y, z)
    let n = indices.dims()[0];
    // slice consumes self, so clone indices once and slice each column.
    let indices_local = indices;
    let x = indices_local.clone().slice([0..n, 0..1]).flatten::<1>(0, 1);
    let y = indices_local.clone().slice([0..n, 1..2]).flatten::<1>(0, 1);
    let z = indices_local.slice([0..n, 2..3]).flatten::<1>(0, 1);

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
    // reshape consumes self, gather consumes self — chain directly since only one gather.
    let result = data.clone().reshape([d0 * d1 * d2]).gather(0, idx);

    let x_mask = in_bounds_mask(x_f, (d2 - 1) as f64, mode);
    let y_mask = in_bounds_mask(y_f, (d1 - 1) as f64, mode);
    let z_mask = in_bounds_mask(z_f, (d0 - 1) as f64, mode);

    match (x_mask, y_mask, z_mask) {
        (Some(xm), Some(ym), Some(zm)) => result * xm * ym * zm,
        _ => result,
    }
}

pub(crate) fn interpolate_4d<B: Backend, const D: usize>(
    data: &Tensor<B, D>,
    indices: Tensor<B, 2>,
    mode: OutOfBoundsMode,
) -> Tensor<B, 1> {
    let shape = data.shape();
    let d0 = shape.dims[0]; // W
    let d1 = shape.dims[1]; // Z
    let d2 = shape.dims[2]; // Y
    let d3 = shape.dims[3]; // X

    let n = indices.dims()[0];
    // slice consumes self, so clone indices once and slice each column.
    let indices_local = indices;
    let x = indices_local.clone().slice([0..n, 0..1]).flatten::<1>(0, 1);
    let y = indices_local.clone().slice([0..n, 1..2]).flatten::<1>(0, 1);
    let z = indices_local.clone().slice([0..n, 2..3]).flatten::<1>(0, 1);
    let w = indices_local.slice([0..n, 3..4]).flatten::<1>(0, 1);

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
    // reshape consumes self, gather consumes self — chain directly since only one gather.
    let result = data.clone().reshape([d0 * d1 * d2 * d3]).gather(0, idx);

    let x_mask = in_bounds_mask(x_f, (d3 - 1) as f64, mode);
    let y_mask = in_bounds_mask(y_f, (d2 - 1) as f64, mode);
    let z_mask = in_bounds_mask(z_f, (d1 - 1) as f64, mode);
    let w_mask = in_bounds_mask(w_f, (d0 - 1) as f64, mode);

    match (x_mask, y_mask, z_mask, w_mask) {
        (Some(xm), Some(ym), Some(zm), Some(wm)) => result * xm * ym * zm * wm,
        _ => result,
    }
}

// ═════════════════════════════════════════════════════════════════════
//  Const-generic shape specialization (Sprint 361 — 351-01-NN-TYPED)
// ═════════════════════════════════════════════════════════════════════
//
// Parallel to `interpolate_3d` above, but takes the volume shape as
// `const D0: usize, const D1: usize, const D2: usize`. This enables
// compile-time bounds, mask inlining, and per-shape monomorphization.
// Generated via the [`ritk_macros::interp_dim_template_nearest_typed!`]
// proc-macro, which provides the nearest-neighbor rounding prelude
// (`floor(coord + 0.5)`) and the pre-clamp mask application.
ritk_macros::interp_dim_template_nearest_typed!(
    3,
    interpolate_nearest_3d_typed,
    x,
    y,
    z,
    wx,
    wy,
    wz,
    D2 - 1,
    D1 - 1,
    D0 - 1,
    D0,
    D1,
    D2,
    {
        // Compute the flat gather index from the per-axis nearest indices.
        // The strides are pre-computed by the proc-macro prelude.
        let flat_data = data.clone().reshape([d0 * d1 * d2]);
        let idx = z_i * stride_z + y_i * stride_y + x_i;
        flat_data.gather(0, idx)
    }
);

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
