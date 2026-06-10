use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray;

/// Helper: create a 2D displacement field [1, C, 4, 4] from a flat f32 slice.
fn disp_2d(values: &[f32], c: usize) -> Tensor<B, 4> {
    let device = Default::default();
    Tensor::<B, 4>::from_data(
        TensorData::new(values.to_vec(), Shape::new([1, c, 4, 4])),
        &device,
    )
}

/// Helper: create a 3D displacement field [1, C, 4, 4, 4] from a flat f32 slice.
fn disp_3d(values: &[f32], c: usize) -> Tensor<B, 5> {
    let device = Default::default();
    Tensor::<B, 5>::from_data(
        TensorData::new(values.to_vec(), Shape::new([1, c, 4, 4, 4])),
        &device,
    )
}

/// Helper: assert a scalar loss is finite.
fn assert_finite(loss: f32, label: &str) {
    assert!(loss.is_finite(), "{label} should be finite, got {loss}");
}

// ── Bending energy ───────────────────────────────────────────────────────────

#[test]
fn bending_energy_2d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_bending_energy::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "bending_energy 2D zero");
    assert!(
        loss.abs() < 1e-6,
        "bending_energy 2D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn bending_energy_3d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_bending_energy::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "bending_energy 3D zero");
    assert!(
        loss.abs() < 1e-6,
        "bending_energy 3D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn bending_energy_2d_nonzero_is_finite_and_positive() {
    // Ramp in x: u_x increases linearly → non-zero second derivative
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = disp_2d(&vals, 2);
    let loss: f32 = super::dispatch_bending_energy::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "bending_energy 2D nonzero");
    assert!(
        loss >= 0.0,
        "bending_energy should be non-negative, got {loss}"
    );
}

#[test]
fn bending_energy_3d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = disp_3d(&vals, 3);
    let loss: f32 = super::dispatch_bending_energy::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "bending_energy 3D nonzero");
    assert!(
        loss >= 0.0,
        "bending_energy should be non-negative, got {loss}"
    );
}

// ── Curvature ────────────────────────────────────────────────────────────────

#[test]
fn curvature_2d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_curvature::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "curvature 2D zero");
    assert!(
        loss.abs() < 1e-6,
        "curvature 2D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn curvature_3d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_curvature::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "curvature 3D zero");
    assert!(
        loss.abs() < 1e-6,
        "curvature 3D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn curvature_2d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = disp_2d(&vals, 2);
    let loss: f32 = super::dispatch_curvature::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "curvature 2D nonzero");
    assert!(loss >= 0.0, "curvature should be non-negative, got {loss}");
}

#[test]
fn curvature_3d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = disp_3d(&vals, 3);
    let loss: f32 = super::dispatch_curvature::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "curvature 3D nonzero");
    assert!(loss >= 0.0, "curvature should be non-negative, got {loss}");
}

// ── Diffusion ────────────────────────────────────────────────────────────────

#[test]
fn diffusion_2d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_diffusion::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "diffusion 2D zero");
    assert!(
        loss.abs() < 1e-6,
        "diffusion 2D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn diffusion_3d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_diffusion::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "diffusion 3D zero");
    assert!(
        loss.abs() < 1e-6,
        "diffusion 3D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn diffusion_2d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = disp_2d(&vals, 2);
    let loss: f32 = super::dispatch_diffusion::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "diffusion 2D nonzero");
    assert!(loss >= 0.0, "diffusion should be non-negative, got {loss}");
}

#[test]
fn diffusion_3d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = disp_3d(&vals, 3);
    let loss: f32 = super::dispatch_diffusion::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "diffusion 3D nonzero");
    assert!(loss >= 0.0, "diffusion should be non-negative, got {loss}");
}

// ── Elastic ──────────────────────────────────────────────────────────────────

#[test]
fn elastic_2d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_elastic::<B, 4>(displacement, 1.0, 1.0).into_scalar();
    assert_finite(loss, "elastic 2D zero");
    assert!(
        loss.abs() < 1e-6,
        "elastic 2D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn elastic_3d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_elastic::<B, 5>(displacement, 1.0, 1.0).into_scalar();
    assert_finite(loss, "elastic 3D zero");
    assert!(
        loss.abs() < 1e-6,
        "elastic 3D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn elastic_2d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = disp_2d(&vals, 2);
    let loss: f32 = super::dispatch_elastic::<B, 4>(displacement, 1.0, 1.0).into_scalar();
    assert_finite(loss, "elastic 2D nonzero");
    assert!(loss >= 0.0, "elastic should be non-negative, got {loss}");
}

#[test]
fn elastic_3d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = disp_3d(&vals, 3);
    let loss: f32 = super::dispatch_elastic::<B, 5>(displacement, 1.0, 1.0).into_scalar();
    assert_finite(loss, "elastic 3D nonzero");
    assert!(loss >= 0.0, "elastic should be non-negative, got {loss}");
}

// ── Total variation ──────────────────────────────────────────────────────────

#[test]
fn total_variation_2d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_total_variation::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "total_variation 2D zero");
    assert!(
        loss.abs() < 1e-6,
        "total_variation 2D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn total_variation_3d_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_total_variation::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "total_variation 3D zero");
    assert!(
        loss.abs() < 1e-6,
        "total_variation 3D zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn total_variation_2d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = disp_2d(&vals, 2);
    let loss: f32 = super::dispatch_total_variation::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "total_variation 2D nonzero");
    assert!(
        loss >= 0.0,
        "total_variation should be non-negative, got {loss}"
    );
}

#[test]
fn total_variation_3d_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = disp_3d(&vals, 3);
    let loss: f32 = super::dispatch_total_variation::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "total_variation 3D nonzero");
    assert!(
        loss >= 0.0,
        "total_variation should be non-negative, got {loss}"
    );
}

// ── Bending energy ≡ curvature (shared laplacian_squared) ────────────────────

#[test]
fn bending_energy_equals_curvature_same_input_2d() {
    let vals: Vec<f32> = (0..32).map(|i| (i as f32 * 0.3).sin()).collect();
    let d1 = disp_2d(&vals, 2);
    let d2 = disp_2d(&vals, 2);
    let be: f32 = super::dispatch_bending_energy::<B, 4>(d1, 1.0).into_scalar();
    let cu: f32 = super::dispatch_curvature::<B, 4>(d2, 1.0).into_scalar();
    assert_finite(be, "bending_energy 2D");
    assert_finite(cu, "curvature 2D");
    assert!(
        (be - cu).abs() < 1e-6,
        "bending_energy and curvature should match for same input, got be={be}, cu={cu}"
    );
}

#[test]
fn bending_energy_equals_curvature_same_input_3d() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.1).cos()).collect();
    let d1 = disp_3d(&vals, 3);
    let d2 = disp_3d(&vals, 3);
    let be: f32 = super::dispatch_bending_energy::<B, 5>(d1, 1.0).into_scalar();
    let cu: f32 = super::dispatch_curvature::<B, 5>(d2, 1.0).into_scalar();
    assert_finite(be, "bending_energy 3D");
    assert_finite(cu, "curvature 3D");
    assert!(
        (be - cu).abs() < 1e-6,
        "bending_energy and curvature should match for same input, got be={be}, cu={cu}"
    );
}
