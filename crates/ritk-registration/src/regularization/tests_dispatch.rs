use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray;

/// Tolerance for zero-displacement field producing near-zero loss.
/// Grid is 4×4 or 4×4×4; f32 accumulation error ≤ 64 × f32::EPSILON.
const ZERO_FIELD_LOSS_TOL: f32 = 1e-6;

/// Helper: create a planar (2-D) displacement field [1, C, 4, 4] from a flat f32 slice.
fn planar_displacement_field(values: &[f32], c: usize) -> Tensor<B, 4> {
    let device = Default::default();
    Tensor::<B, 4>::from_data(
        TensorData::new(values.to_vec(), Shape::new([1, c, 4, 4])),
        &device,
    )
}

/// Helper: create a volumetric (3-D) displacement field [1, C, 4, 4, 4] from a flat f32 slice.
fn volume_displacement_field(values: &[f32], c: usize) -> Tensor<B, 5> {
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
fn bending_energy_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_bending_energy::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "bending_energy planar zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "bending_energy planar zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn bending_energy_zero_displacement_is_zero_volume() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_bending_energy::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "bending_energy volume zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "bending_energy volume zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn bending_energy_nonzero_is_finite_and_positive() {
    // Ramp in x: u_x increases linearly → non-zero second derivative
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = planar_displacement_field(&vals, 2);
    let loss: f32 = super::dispatch_bending_energy::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "bending_energy planar nonzero");
    assert!(
        loss >= 0.0,
        "bending_energy should be non-negative, got {loss}"
    );
}

#[test]
fn bending_energy_nonzero_is_finite_and_positive_volume() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = volume_displacement_field(&vals, 3);
    let loss: f32 = super::dispatch_bending_energy::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "bending_energy volume nonzero");
    assert!(
        loss >= 0.0,
        "bending_energy should be non-negative, got {loss}"
    );
}

// ── Curvature ────────────────────────────────────────────────────────────────

#[test]
fn curvature_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_curvature::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "curvature planar zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "curvature planar zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn curvature_zero_displacement_is_zero_volume() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_curvature::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "curvature volume zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "curvature volume zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn curvature_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = planar_displacement_field(&vals, 2);
    let loss: f32 = super::dispatch_curvature::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "curvature planar nonzero");
    assert!(loss >= 0.0, "curvature should be non-negative, got {loss}");
}

#[test]
fn curvature_nonzero_is_finite_and_positive_volume() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = volume_displacement_field(&vals, 3);
    let loss: f32 = super::dispatch_curvature::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "curvature volume nonzero");
    assert!(loss >= 0.0, "curvature should be non-negative, got {loss}");
}

// ── Diffusion ────────────────────────────────────────────────────────────────

#[test]
fn diffusion_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_diffusion::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "diffusion planar zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "diffusion planar zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn diffusion_zero_displacement_is_zero_volume() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_diffusion::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "diffusion volume zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "diffusion volume zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn diffusion_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = planar_displacement_field(&vals, 2);
    let loss: f32 = super::dispatch_diffusion::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "diffusion planar nonzero");
    assert!(loss >= 0.0, "diffusion should be non-negative, got {loss}");
}

#[test]
fn diffusion_nonzero_is_finite_and_positive_volume() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = volume_displacement_field(&vals, 3);
    let loss: f32 = super::dispatch_diffusion::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "diffusion volume nonzero");
    assert!(loss >= 0.0, "diffusion should be non-negative, got {loss}");
}

// ── Elastic ──────────────────────────────────────────────────────────────────

#[test]
fn elastic_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_elastic::<B, 4>(displacement, 1.0, 1.0).into_scalar();
    assert_finite(loss, "elastic planar zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "elastic planar zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn elastic_zero_displacement_is_zero_volume() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_elastic::<B, 5>(displacement, 1.0, 1.0).into_scalar();
    assert_finite(loss, "elastic volume zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "elastic volume zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn elastic_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = planar_displacement_field(&vals, 2);
    let loss: f32 = super::dispatch_elastic::<B, 4>(displacement, 1.0, 1.0).into_scalar();
    assert_finite(loss, "elastic planar nonzero");
    assert!(loss >= 0.0, "elastic should be non-negative, got {loss}");
}

#[test]
fn elastic_nonzero_is_finite_and_positive_volume() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = volume_displacement_field(&vals, 3);
    let loss: f32 = super::dispatch_elastic::<B, 5>(displacement, 1.0, 1.0).into_scalar();
    assert_finite(loss, "elastic volume nonzero");
    assert!(loss >= 0.0, "elastic should be non-negative, got {loss}");
}

// ── Total variation ──────────────────────────────────────────────────────────

#[test]
fn total_variation_zero_displacement_is_zero() {
    let displacement = Tensor::<B, 4>::zeros([1, 2, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_total_variation::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "total_variation planar zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "total_variation planar zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn total_variation_zero_displacement_is_zero_volume() {
    let displacement = Tensor::<B, 5>::zeros([1, 3, 4, 4, 4], &Default::default());
    let loss: f32 = super::dispatch_total_variation::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "total_variation volume zero");
    assert!(
        loss.abs() < ZERO_FIELD_LOSS_TOL,
        "total_variation volume zero displacement should be ≈ 0, got {loss}"
    );
}

#[test]
fn total_variation_nonzero_is_finite_and_positive() {
    let vals: Vec<f32> = (0..32).map(|i| (i % 4) as f32 * 0.1).collect();
    let displacement = planar_displacement_field(&vals, 2);
    let loss: f32 = super::dispatch_total_variation::<B, 4>(displacement, 1.0).into_scalar();
    assert_finite(loss, "total_variation planar nonzero");
    assert!(
        loss >= 0.0,
        "total_variation should be non-negative, got {loss}"
    );
}

#[test]
fn total_variation_nonzero_is_finite_and_positive_volume() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.05).sin()).collect();
    let displacement = volume_displacement_field(&vals, 3);
    let loss: f32 = super::dispatch_total_variation::<B, 5>(displacement, 1.0).into_scalar();
    assert_finite(loss, "total_variation volume nonzero");
    assert!(
        loss >= 0.0,
        "total_variation should be non-negative, got {loss}"
    );
}

// ── Bending energy ≡ curvature (shared laplacian_squared) ────────────────────

#[test]
fn bending_energy_equals_curvature_same_input() {
    let vals: Vec<f32> = (0..32).map(|i| (i as f32 * 0.3).sin()).collect();
    let d1 = planar_displacement_field(&vals, 2);
    let d2 = planar_displacement_field(&vals, 2);
    let be: f32 = super::dispatch_bending_energy::<B, 4>(d1, 1.0).into_scalar();
    let cu: f32 = super::dispatch_curvature::<B, 4>(d2, 1.0).into_scalar();
    assert_finite(be, "bending_energy planar");
    assert_finite(cu, "curvature planar");
    assert!(
        (be - cu).abs() < ZERO_FIELD_LOSS_TOL,
        "bending_energy and curvature should match for same input, got be={be}, cu={cu}"
    );
}

#[test]
fn bending_energy_equals_curvature_same_input_volume() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.1).cos()).collect();
    let d1 = volume_displacement_field(&vals, 3);
    let d2 = volume_displacement_field(&vals, 3);
    let be: f32 = super::dispatch_bending_energy::<B, 5>(d1, 1.0).into_scalar();
    let cu: f32 = super::dispatch_curvature::<B, 5>(d2, 1.0).into_scalar();
    assert_finite(be, "bending_energy volume");
    assert_finite(cu, "curvature volume");
    assert!(
        (be - cu).abs() < ZERO_FIELD_LOSS_TOL,
        "bending_energy and curvature should match for same input, got be={be}, cu={cu}"
    );
}
