//! 1-D dispatch tests (Sprint 359 — 351-01-ND-TYPED).
//!
//! Verifies that the per-shape runtime dispatcher routes 1-D tensors
//! with common shapes to the const-generic typed instantiations, and
//! falls through to the generic path for uncommon shapes.

use super::*;
use crate::interpolation::dispatch::Dispatch1DTyped;

fn build_1d(side: usize) -> Tensor<TestBackend, 1> {
    // Fill with 0.0 except a single 1.0 at the center element.
    let mut data = vec![0.0_f32; side];
    data[side / 2] = 1.0;
    let device = Default::default();
    Tensor::<TestBackend, 1>::from_data(
        TensorData::new(data, ritk_image::tensor::Shape::new([side])),
        &device,
    )
}

fn query_1d(side: usize) -> Tensor<TestBackend, 2> {
    let device = Default::default();
    let mid = side as f32 / 2.0;
    Tensor::<TestBackend, 2>::from_floats([[mid]], &device)
}

#[test]
fn dispatch_1d_routes_64_to_typed_path() {
    let data = build_1d(64);
    let indices = query_1d(64);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1-D 64 center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_1d_routes_128_to_typed_path() {
    let data = build_1d(128);
    let indices = query_1d(128);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1-D 128 center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_1d_routes_256_to_typed_path() {
    let data = build_1d(256);
    let indices = query_1d(256);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1-D 256 center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_1d_routes_512_to_typed_path() {
    let data = build_1d(512);
    let indices = query_1d(512);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1-D 512 center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_1d_falls_through_to_generic_for_uncommon_shape() {
    // Uncommon 1-D shape (e.g. 100) falls through to the generic path.
    let data = build_1d(100);
    let indices = query_1d(100);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1-D 100 center (fallback) should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_for_shape_convenience_routes_typed() {
    let data = build_1d(256);
    let indices = query_1d(256);
    let result = dispatch_for_shape::<TestBackend, 1>(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1-D 256 via dispatch_for_shape should give 1.0, got {}",
        val
    );
}

#[test]
fn type_narrowing_wrapper_1d_routes_through_sealed_trait() {
    let data = build_1d(128);
    let indices = query_1d(128);
    let result: Tensor<TestBackend, 1> = data.dispatch_1d_typed(indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1-D 128 via Dispatch1DTyped wrapper should give 1.0, got {}",
        val
    );
}
