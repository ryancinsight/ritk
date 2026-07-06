//! Tests for the per-shape runtime dispatcher (audit §8 351-01).
//!
//! Verifies that [`dispatch_linear`] correctly routes 3-D tensors
//! with cube shapes (64³, 128³, 256³, 512³) through the
//! const-generic typed instantiations, and falls through to the
//! generic path for non-cube shapes. Also tests the sealed
//! [`DispatchByShape`] trait method directly.
use super::*;
use crate::interpolation::shared::OutOfBoundsMode;
use ritk_image::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

fn build_cube(side: usize) -> Tensor<TestBackend, 3> {
    let device = Default::default();
    let data = Tensor::<TestBackend, 3>::zeros([side, side, side], &device);
    let mid = side / 2;
    let ones = Tensor::<TestBackend, 3>::ones([1, 1, 1], &device);
    data.slice_assign([mid..mid + 1, mid..mid + 1, mid..mid + 1], ones)
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
