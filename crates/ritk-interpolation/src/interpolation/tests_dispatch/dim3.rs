//! 3-D dispatch tests — linear interpolation, sealed trait, and nearest.
//!
//! Verifies that [`dispatch_linear`] correctly routes 3-D tensors
//! with cube shapes (64³, 128³, 256³, 512³) through the
//! const-generic typed instantiations, and falls through to the
//! generic path for non-cube shapes. Also tests the sealed
//! [`DispatchByShape`] trait method directly, and the nearest-neighbor
//! sealed-trait wrappers.

use super::*;
use crate::interpolation::dispatch::{Dispatch3DTyped, DispatchNearest3DTyped};

// ══════════════════════════════════════════════════════════════════════
// Linear dispatch — cube shapes
// ══════════════════════════════════════════════════════════════════════

#[test]
fn dispatch_routes_64_cube_to_typed_path() {
    let data = build_cube(64);
    let indices = query_near_center(64);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    // The center voxel is 1.0; trilinear interpolation at the exact
    // center should return 1.0 (it samples the single 1.0 voxel).
    assert!(
        (val - 1.0).abs() < 1e-5,
        "64³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_128_cube_to_typed_path() {
    let data = build_cube(128);
    let indices = query_near_center(128);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "128³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_256_cube_to_typed_path() {
    let data = build_cube(256);
    let indices = query_near_center(256);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "256³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_512_cube_to_typed_path() {
    let data = build_cube(512);
    let indices = query_near_center(512);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "512³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_falls_through_to_generic_for_non_cube() {
    // Non-cube shape (e.g. 100×150×200) falls through to the generic
    // interpolate_3d path. The result should still be correct.
    let device = Default::default();
    let data_vec: Vec<f32> = (0..100 * 150 * 200).map(|i| i as f32).collect();
    let data = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data_vec, burn::tensor::Shape::new([100, 150, 200])),
        &device,
    );
    let indices = Tensor::<TestBackend, 2>::from_floats([[10.0, 20.0, 30.0]], &device);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    // Should be a valid interpolated value (somewhere in the data range).
    assert!(
        val.is_finite(),
        "Non-cube result should be finite, got {}",
        val
    );
    assert!(
        (0.0..(100 * 150 * 200) as f32).contains(&val),
        "Non-cube result should be in data range, got {}",
        val
    );
}

// ══════════════════════════════════════════════════════════════════════
// Sealed trait tests (Sprint 357)
// ══════════════════════════════════════════════════════════════════════

/// Verifies the sealed [`DispatchByShape`] trait method directly.
/// This is the trait that `dispatch_3d_for_shape` delegates to.
#[test]
fn sealed_trait_dispatch_by_shape_routes_typed_path() {
    let data = build_cube(256);
    let indices = query_near_center(256);
    // Call the sealed trait method directly via the public wrapper.
    let result = dispatch_3d_for_shape(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Sealed trait method 256³ center should give 1.0, got {}",
        val
    );
}

/// Verifies the type-narrowing wrapper trait [`Dispatch3DTyped`].
/// The trait is implemented for any `Tensor<B, D>`, but only the
/// `D == 3` branch is meaningful.
#[test]
fn type_narrowing_wrapper_routes_3d_through_sealed_trait() {
    let data = build_cube(128);
    let indices = query_near_center(128);
    // Call the type-narrowing wrapper directly.
    let result: Tensor<TestBackend, 1> = data.dispatch_3d_typed(indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Type-narrowing wrapper 128³ center should give 1.0, got {}",
        val
    );
}

// ══════════════════════════════════════════════════════════════════════
// Nearest-neighbor sealed trait tests (Sprint 358)
// ══════════════════════════════════════════════════════════════════════

/// Verifies that the sealed [`DispatchNearestByShape`] trait method
/// correctly dispatches 3-D nearest-neighbor through the wrapper.
/// Currently falls through to the generic `interpolate_3d` for all
/// shapes — the test verifies the wrapper is wired correctly.
#[test]
fn nearest_sealed_trait_dispatches_3d() {
    let data = build_cube(64);
    // Nearest-neighbor at the exact center of a cube with a 1.0 at the
    // center voxel should return 1.0 (it rounds to the center voxel).
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[32.0, 32.0, 32.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 64³ at center should give 1.0, got {}",
        val
    );
}

/// Verifies the type-narrowing wrapper trait [`DispatchNearest3DTyped`]
/// for nearest-neighbor. The trait is implemented for any
/// `Tensor<B, D>`, but only the `D == 3` branch is meaningful.
#[test]
fn nearest_type_narrowing_wrapper_routes_3d() {
    let data = build_cube(64);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[32.0, 32.0, 32.0]], &device);
    // Call the type-narrowing wrapper directly.
    let result: Tensor<TestBackend, 1> =
        data.dispatch_nearest_3d_typed(indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest type-narrowing wrapper 64³ at center should give 1.0, got {}",
        val
    );
}

/// Verifies the [`dispatch_nearest_3d_for_shape`] convenience function
/// (parallel to [`dispatch_3d_for_shape`] for linear).
#[test]
fn nearest_dispatch_3d_for_shape_routes_correctly() {
    let data = build_cube(256);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[128.0, 128.0, 128.0]], &device);
    let result = dispatch_nearest_3d_for_shape(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 256³ at center via dispatch_nearest_3d_for_shape should give 1.0, got {}",
        val
    );
}
