//! Tests for the per-shape runtime dispatcher (audit §8 351-01).
//!
//! Verifies that [`dispatch_linear`] correctly routes 3-D tensors
//! with cube shapes (64³, 128³, 256³, 512³) through the
//! const-generic typed instantiations, and falls through to the
//! generic path for non-cube shapes. Also tests the sealed
//! [`DispatchByShape`] trait method directly.
use super::*;
use crate::interpolation::shared::OutOfBoundsMode;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

fn build_cube(side: usize) -> Tensor<TestBackend, 3> {
    // Fill with 0.0 except a single 1.0 at the center voxel — easy
    // to spot-check after interpolation.
    let mut data = vec![0.0_f32; side * side * side];
    let mid = side / 2;
    data[mid * side * side + mid * side + mid] = 1.0;
    let device = Default::default();
    Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data, burn::tensor::Shape::new([side, side, side])),
        &device,
    )
}

fn query_near_center(side: usize) -> Tensor<TestBackend, 2> {
    let device = Default::default();
    let mid = side as f32 / 2.0;
    Tensor::<TestBackend, 2>::from_floats([[mid, mid, mid]], &device)
}

mod rank1;
mod rank2;
mod rank3;
mod rank3_extended;
mod rank4;
