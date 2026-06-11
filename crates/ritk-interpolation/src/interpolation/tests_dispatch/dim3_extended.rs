//! Extended 3-D shape dispatch tests (Sprint 360 — 351-01-SHAPE-LIST, Sprint 361 — 351-01-NN-TYPED).
//!
//! Verifies the extended shape-routing match in
//! [`DispatchByShape::dispatch_by_shape`] for medical-imaging shapes:
//! 6 additional cubes (32³, 48³, 96³, 192³, 384³, 1024³) and 2
//! non-cube clinical shapes (256×256×128 CT, 192×256×256 MRI) for both
//! linear and nearest-neighbor dispatch. Also tests the typed
//! nearest-neighbor routing for 64³, 128³, 256³, and 512³.

use super::*;

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
    // column 0 → X (innermost, stride 1, bounds = dims[2])
    // column 1 → Y (middle, stride dims[2], bounds = dims[1])
    // column 2 → Z (outermost, stride dims[1]*dims[2], bounds = dims[0])
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
// Typed nearest-neighbor routing (Sprint 361 — 351-01-NN-TYPED)
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
