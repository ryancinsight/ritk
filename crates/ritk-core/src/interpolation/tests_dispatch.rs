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

// ════════════════════════════════════════════════════════════════════
//  Sealed trait tests (Sprint 357)
// ════════════════════════════════════════════════════════════════════

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
    use super::Dispatch3DTyped;
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

// ════════════════════════════════════════════════════════════════════
//  Nearest-neighbor sealed trait tests (Sprint 358)
// ════════════════════════════════════════════════════════════════════

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
    use super::DispatchNearest3DTyped;
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

// ═════════════════════════════════════════════════════════════════════
//  N-D typed dispatcher tests (Sprint 359 — 351-01-ND-TYPED)
// ═════════════════════════════════════════════════════════════════════
//
// Verifies that the per-shape runtime dispatcher routes 1-D, 2-D, and
// 4-D tensors with common shapes to the const-generic typed
// instantiations, and falls through to the generic path for uncommon
// shapes. Parallel to the 3-D tests above, but exercises the
// `DispatchByShape` impls for `Tensor<B, 1>`, `Tensor<B, 2>`, and
// `Tensor<B, 4>`.

// ── 1-D tests ──────────────────────────────────────────────────────

fn build_1d(side: usize) -> Tensor<TestBackend, 1> {
    // Fill with 0.0 except a single 1.0 at the center element.
    let mut data = vec![0.0_f32; side];
    data[side / 2] = 1.0;
    let device = Default::default();
    Tensor::<TestBackend, 1>::from_data(
        TensorData::new(data, burn::tensor::Shape::new([side])),
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
fn dispatch_1d_for_shape_convenience_routes_typed() {
    let data = build_1d(256);
    let indices = query_1d(256);
    let result = dispatch_1d_for_shape(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1-D 256 via dispatch_1d_for_shape should give 1.0, got {}",
        val
    );
}

#[test]
fn type_narrowing_wrapper_1d_routes_through_sealed_trait() {
    use super::Dispatch1DTyped;
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

// ── 2-D tests ──────────────────────────────────────────────────────

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
    use super::Dispatch2DTyped;
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

// ── 4-D tests ──────────────────────────────────────────────────────

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
fn dispatch_4d_for_shape_convenience_routes_typed() {
    let data = build_4d(64);
    let indices = query_4d(64);
    let result = dispatch_4d_for_shape(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "4-D 64⁴ via dispatch_4d_for_shape should give 1.0, got {}",
        val
    );
}

#[test]
fn type_narrowing_wrapper_4d_routes_through_sealed_trait() {
    use super::Dispatch4DTyped;
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

// ═════════════════════════════════════════════════════════════════════
//  Extended medical-imaging shape list (Sprint 360 — 351-01-SHAPE-LIST)
// ═════════════════════════════════════════════════════════════════════
//
// Verifies the extended shape-routing match in
// [`DispatchByShape::dispatch_by_shape`] for the new medical-imaging
// shapes: 6 additional cubes (32³, 48³, 96³, 192³, 384³, 1024³) and
// 2 non-cube clinical shapes (256×256×128 CT, 192×256×256 MRI). Each
// test verifies that the routing reaches the const-generic typed
// instantiation and returns the correct interpolated value at the
// center voxel.

fn build_rect(dims: [usize; 3]) -> Tensor<TestBackend, 3> {
    // Fill with 0.0 except a single 1.0 at the center voxel.
    let mut data = vec![0.0_f32; dims[0] * dims[1] * dims[2]];
    let mid = [dims[0] / 2, dims[1] / 2, dims[2] / 2];
    let idx = mid[0] * dims[1] * dims[2] + mid[1] * dims[2] + mid[2];
    data[idx] = 1.0;
    let device = Default::default();
    Tensor::<TestBackend, 3>::from_data(
        TensorData::new(data, burn::tensor::Shape::new(dims)),
        &device,
    )
}

fn query_rect_center(dims: [usize; 3]) -> Tensor<TestBackend, 2> {
    let device = Default::default();
    // Indices tensor columns are interpreted by the kernel as [X, Y, Z]:
    //   column 0 → X (innermost, stride 1, bounds = dims[2])
    //   column 1 → Y (middle,    stride dims[2], bounds = dims[1])
    //   column 2 → Z (outermost, stride dims[1]*dims[2], bounds = dims[0])
    // The data buffer (`build_rect`) places 1.0 at [Z_mid, Y_mid, X_mid] in
    // [Z, Y, X] row-major order, so the query must return [X_mid, Y_mid, Z_mid]
    // to land on the same voxel.
    let mid = [
        dims[2] as f32 / 2.0, // X
        dims[1] as f32 / 2.0, // Y
        dims[0] as f32 / 2.0, // Z
    ];
    Tensor::<TestBackend, 2>::from_floats([mid], &device)
}

// ── Linear dispatcher — extended cube shapes ──────────────────────

#[test]
fn dispatch_routes_32_cube_to_typed_path() {
    let data = build_cube(32);
    let indices = query_near_center(32);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "32³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_48_cube_to_typed_path() {
    let data = build_cube(48);
    let indices = query_near_center(48);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "48³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_96_cube_to_typed_path() {
    let data = build_cube(96);
    let indices = query_near_center(96);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "96³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_192_cube_to_typed_path() {
    let data = build_cube(192);
    let indices = query_near_center(192);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "192³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_384_cube_to_typed_path() {
    let data = build_cube(384);
    let indices = query_near_center(384);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "384³ center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_1024_cube_to_typed_path() {
    let data = build_cube(1024);
    let indices = query_near_center(1024);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "1024³ center should give 1.0, got {}",
        val
    );
}

// ── Linear dispatcher — non-cube clinical shapes ─────────────────

#[test]
fn dispatch_routes_256x256x128_ct_to_typed_path() {
    let data = build_rect([256, 256, 128]);
    let indices = query_rect_center([256, 256, 128]);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "256×256×128 CT center should give 1.0, got {}",
        val
    );
}

#[test]
fn dispatch_routes_192x256x256_mri_to_typed_path() {
    let data = build_rect([192, 256, 256]);
    let indices = query_rect_center([192, 256, 256]);
    let result = dispatch_linear(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "192×256×256 MRI center should give 1.0, got {}",
        val
    );
}

// ── Nearest dispatcher — extended cube shapes ────────────────────

#[test]
fn nearest_dispatch_routes_32_cube_to_typed_path() {
    let data = build_cube(32);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[16.0, 16.0, 16.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 32³ at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_dispatch_routes_48_cube_to_typed_path() {
    let data = build_cube(48);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[24.0, 24.0, 24.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 48³ at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_dispatch_routes_96_cube_to_typed_path() {
    let data = build_cube(96);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[48.0, 48.0, 48.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 96³ at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_dispatch_routes_192_cube_to_typed_path() {
    let data = build_cube(192);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[96.0, 96.0, 96.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 192³ at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_dispatch_routes_384_cube_to_typed_path() {
    let data = build_cube(384);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[192.0, 192.0, 192.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 384³ at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_dispatch_routes_1024_cube_to_typed_path() {
    let data = build_cube(1024);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[512.0, 512.0, 512.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 1024³ at center should give 1.0, got {}",
        val
    );
}

// ── Nearest dispatcher — non-cube clinical shapes ────────────────

#[test]
fn nearest_dispatch_routes_256x256x128_ct_to_typed_path() {
    let data = build_rect([256, 256, 128]);
    let device = Default::default();
    // Indices are in [X, Y, Z] order. The 1.0 is at [X=64, Y=128, Z=128].
    let indices = Tensor::<TestBackend, 2>::from_floats([[64.0, 128.0, 128.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 256×256×128 CT at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_dispatch_routes_192x256x256_mri_to_typed_path() {
    let data = build_rect([192, 256, 256]);
    let device = Default::default();
    // Indices are in [X, Y, Z] order. The 1.0 is at [X=128, Y=128, Z=96].
    let indices = Tensor::<TestBackend, 2>::from_floats([[128.0, 128.0, 96.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest 192×256×256 MRI at center should give 1.0, got {}",
        val
    );
}

// ═════════════════════════════════════════════════════════════════════
//  Typed nearest-neighbor routing (Sprint 361 — 351-01-NN-TYPED)
// ═════════════════════════════════════════════════════════════════════
//
// Verifies that the per-shape runtime dispatcher routes 64³, 128³,
// 256³, and 512³ nearest-neighbor calls through the const-generic
// typed instantiations generated by the
// [`ritk_macros::interp_dim_template_nearest_typed!`] proc-macro.

#[test]
fn nearest_typed_routes_64_cube() {
    let data = build_cube(64);
    let device = Default::default();
    // Query at exact center voxel (32, 32, 32) — nearest should round to it.
    let indices = Tensor::<TestBackend, 2>::from_floats([[32.0, 32.0, 32.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest typed 64³ at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_typed_routes_128_cube() {
    let data = build_cube(128);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[64.0, 64.0, 64.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest typed 128³ at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_typed_routes_256_cube() {
    let data = build_cube(256);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[128.0, 128.0, 128.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest typed 256³ at center should give 1.0, got {}",
        val
    );
}

#[test]
fn nearest_typed_routes_512_cube() {
    let data = build_cube(512);
    let device = Default::default();
    let indices = Tensor::<TestBackend, 2>::from_floats([[256.0, 256.0, 256.0]], &device);
    let result = dispatch_nearest(&data, indices, OutOfBoundsMode::Clamp);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (val - 1.0).abs() < 1e-5,
        "Nearest typed 512³ at center should give 1.0, got {}",
        val
    );
}
