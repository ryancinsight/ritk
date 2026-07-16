// Integration tests for `dispatch_linear` / `dispatch_nearest` routing.
// Per-dimension unit tests live alongside kernel implementations in
// `interpolation::kernel::linear` and `interpolation::kernel::nearest`.
// This file covers cross-dimension routing smoke tests.

use burn_ndarray::NdArray;
use ritk_image::tensor::{Shape, Tensor, TensorData};

use super::{dispatch_linear, dispatch_nearest};
use crate::interpolation::shared::OutOfBoundsMode;

type B = NdArray<f32>;

// ── Helpers ───────────────────────────────────────────────────────────────

fn make_1d(vals: &[f32]) -> Tensor<B, 1> {
    let n = vals.len();
    Tensor::<B, 1>::from_data(
        TensorData::new(vals.to_vec(), Shape::new([n])),
        &Default::default(),
    )
}

fn make_3d(vals: &[f32], shape: [usize; 3]) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        TensorData::new(vals.to_vec(), Shape::new(shape)),
        &Default::default(),
    )
}

fn make_indices(rows: Vec<Vec<f32>>) -> Tensor<B, 2> {
    let n_points = rows.len();
    let n_dims = if rows.is_empty() { 0 } else { rows[0].len() };
    let flat: Vec<f32> = rows.into_iter().flatten().collect();
    Tensor::<B, 2>::from_data(
        TensorData::new(flat, Shape::new([n_points, n_dims])),
        &Default::default(),
    )
}

// ── dispatch_linear ───────────────────────────────────────────────────────

#[test]
fn dispatch_linear_1d_routes_without_panic() {
    let data = make_1d(&[1.0, 2.0, 3.0, 4.0]);
    // Query at index 1.5 — between elements at indices 1 and 2.
    let indices = make_indices(vec![vec![1.5]]);
    let _result = dispatch_linear::<B, 1>(&data, indices, OutOfBoundsMode::Clamp);
}

#[test]
fn dispatch_linear_3d_routes_without_panic() {
    let side = 4usize;
    let vals: Vec<f32> = (0..side.pow(3) as u32).map(|v| v as f32).collect();
    let data = make_3d(&vals, [side, side, side]);
    let cx = side as f32 / 2.0;
    let indices = make_indices(vec![vec![cx, cx, cx]]);
    let _result = dispatch_linear::<B, 3>(&data, indices, OutOfBoundsMode::Clamp);
}

// ── dispatch_nearest ──────────────────────────────────────────────────────

#[test]
fn dispatch_nearest_3d_routes_without_panic() {
    let side = 4usize;
    let vals: Vec<f32> = (0..side.pow(3) as u32).map(|v| v as f32).collect();
    let data = make_3d(&vals, [side, side, side]);
    let cx = side as f32 / 2.0;
    let indices = make_indices(vec![vec![cx, cx, cx]]);
    let _result = dispatch_nearest::<B, 3>(&data, indices, OutOfBoundsMode::Clamp);
}
