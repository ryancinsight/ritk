//! 2-D dispatch tests (Sprint 359 — 351-01-ND-TYPED).
//!
//! Verifies that the per-shape runtime dispatcher routes 2-D tensors
//! with common square shapes to the const-generic typed instantiations,
//! and falls through to the generic path for non-square shapes.

use super::*;
use crate::interpolation::dispatch::Dispatch2DTyped;

fn build_2d(side: usize) -> Tensor<TestBackend, 2> {
    // Fill with 0.0 except a single 1.0 at the center pixel.
    let mut data = vec![0.0_f32; side * side];
    let mid = side / 2;
    data[mid * side + mid] = 1.0;
    let device = Default::default();
    Tensor::<TestBackend, 2>::from_data(
        TensorData::new(data, burn::tensor::Shape::new([side, side])),
        &device,
    )
}

fn query_2d(side: usize) -> Tensor<TestBackend, 2> {
    let device = Default::default();
    let mid = side as f32 / 2.0;
    Tensor::<TestBackend, 2>::from_floats([[mid, mid]], &device)
}

#[test]
fn dispatch_2d_routes_64x64_to_typed_path() {
    let data = build_2d(64);
    let indices = query_2d(64);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "2-D 64×64 center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_2d_routes_128x128_to_typed_path() {
    let data = build_2d(128);
    let indices = query_2d(128);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "2-D 128×128 center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_2d_routes_256x256_to_typed_path() {
    let data = build_2d(256);
    let indices = query_2d(256);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "2-D 256×256 center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_2d_routes_512x512_to_typed_path() {
    let data = build_2d(512);
    let indices = query_2d(512);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "2-D 512×512 center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_2d_falls_through_to_generic_for_non_square() {
    // Non-square 2-D shape (e.g. 100×150) falls through to the generic path.
    let device = Default::default();
    let mut data = vec![0.0_f32; 100 * 150];
    data[50 * 150 + 75] = 1.0;
    let data = Tensor::<TestBackend, 2>::from_data(
        TensorData::new(data, burn::tensor::Shape::new([100, 150])),
        &device,
    );
    let indices = Tensor::<TestBackend, 2>::from_floats([[75.0, 50.0]], &device);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "2-D 100×150 center (fallback) should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_2d_for_shape_convenience_routes_typed() {
    let data = build_2d(256);
    let indices = query_2d(256);
    let result = dispatch_2d_for_shape(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "2-D 256×256 via dispatch_2d_for_shape should give 1.0, got {}",
        val
    );
}

#[test]
fn type_narrowing_wrapper_2d_routes_through_sealed_trait() {
    let data = build_2d(128);
    let indices = query_2d(128);
    let result: Tensor<TestBackend, 1> = data.dispatch_2d_typed(indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "2-D 128×128 via Dispatch2DTyped wrapper should give 1.0, got {}",
        val
    );
}
