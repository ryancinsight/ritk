//! 4-D dispatch tests (Sprint 359 — 351-01-ND-TYPED).
//!
//! Verifies that the per-shape runtime dispatcher routes 4-D tensors
//! with common shapes to the const-generic typed instantiations, and
//! falls through to the generic path for uncommon shapes.

use super::*;
use crate::interpolation::dispatch::Dispatch4DTyped;

fn build_4d(side: usize) -> Tensor<TestBackend, 4> {
    // Fill with 0.0 except a single 1.0 at the center voxel.
    let mut data = vec![0.0_f32; side * side * side * side];
    let mid = side / 2;
    data[mid * side * side * side + mid * side * side + mid * side + mid] = 1.0;
    let device = Default::default();
    Tensor::<TestBackend, 4>::from_data(
        TensorData::new(data, burn::tensor::Shape::new([side, side, side, side])),
        &device,
    )
}

fn query_4d(side: usize) -> Tensor<TestBackend, 2> {
    let device = Default::default();
    let mid = side as f32 / 2.0;
    Tensor::<TestBackend, 2>::from_floats([[mid, mid, mid, mid]], &device)
}

#[test]
fn dispatch_4d_routes_64_to_typed_path() {
    let data = build_4d(64);
    let indices = query_4d(64);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "4-D 64⁴ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_4d_routes_128_to_typed_path() {
    let data = build_4d(128);
    let indices = query_4d(128);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "4-D 128⁴ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_4d_falls_through_to_generic_for_uncommon_shape() {
    // Uncommon 4-D shape (e.g. 16) falls through to the generic path.
    let data = build_4d(16);
    let indices = query_4d(16);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "4-D 16⁴ center (fallback) should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_for_shape_convenience_routes_typed() {
    let data = build_4d(64);
    let indices = query_4d(64);
    let result = dispatch_for_shape::<TestBackend, 4>(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "4-D 64⁴ via dispatch_for_shape should give 1.0, got {}",
        val
    );
}

#[test]
fn type_narrowing_wrapper_4d_routes_through_sealed_trait() {
    let data = build_4d(128);
    let indices = query_4d(128);
    let result: Tensor<TestBackend, 1> = data.dispatch_4d_typed(indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "4-D 128⁴ via Dispatch4DTyped wrapper should give 1.0, got {}",
        val
    );
}
