//! Verification of the native (Coeus) regularizer dispatch kernels.
//!
//! Evidence tier: analytical (closed-form value oracles). Each finite-difference
//! regularizer is checked against a displacement field whose penalty has a
//! hand-derived exact value, plus the structural laws it must obey
//! (zero-field → zero, bending ≡ curvature, weight linearity).
//!
//! Backend: the deterministic single-threaded [`SequentialBackend`], so the
//! fused stencil/reduction has no reduction-order variance to bound.

use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;

type B = SequentialBackend;

/// Zero-field losses are exactly zero (all differences vanish); the tolerance
/// only absorbs the final division, so it is tight.
const ZERO_FIELD_LOSS_TOL: f32 = 1e-6;

/// Closed-form oracle tolerance. The oracle fields have ≤192 f32 voxels; the
/// worst-case accumulated rounding is `N·f32::EPSILON·max_term` with `N ≤ 192`
/// and `f32::EPSILON ≈ 1.19e-7`, i.e. `< 3e-5` relative on the `O(1)` oracle
/// values. `1e-5` absolute bounds it, and the integer-valued quadratic oracles
/// are in fact exact in f32.
const FD_LOSS_TOL: f32 = 1e-5;

/// Planar (2-D) displacement field `[1, C, 4, 4]` from a flat row-major slice
/// (`[channel][height][width]`, width fastest).
fn planar(values: &[f32], c: usize) -> Tensor<f32, B> {
    Tensor::<f32, B>::from_slice_on([1, c, 4, 4], values, &SequentialBackend)
}

/// Volumetric (3-D) displacement field `[1, C, 4, 4, 4]` from a flat slice.
fn volume(values: &[f32], c: usize) -> Tensor<f32, B> {
    Tensor::<f32, B>::from_slice_on([1, c, 4, 4, 4], values, &SequentialBackend)
}

/// A field that is a linear ramp in the width axis: `u = 0.1·w`, identical
/// across channels/rows/slices. Its width forward-difference is `0.1` at the
/// three interior columns and `0` (zero pad) at the last, so `mean(g_w²) =
/// (3/4)·0.1² = 0.0075` and `mean(|g_w|) = (3/4)·0.1 = 0.075`; every other axis
/// gradient is zero.
fn width_ramp(len: usize) -> Vec<f32> {
    (0..len).map(|m| (m % 4) as f32 * 0.1).collect()
}

/// A field quadratic in the width axis: `u = w²`. Its second width difference is
/// `(w−1)² + (w+1)² − 2w² = 2` everywhere, so the discrete Laplacian is `2` at
/// every interior voxel and `0` on the zero border.
fn width_quadratic(len: usize) -> Vec<f32> {
    (0..len).map(|m| ((m % 4) as f32).powi(2)).collect()
}

fn assert_close(got: f32, expected: f32, tol: f32, label: &str) {
    assert!(
        (got - expected).abs() < tol,
        "{label}: expected {expected}, got {got} (tol {tol})"
    );
}

// ── Zero field → zero loss (all five regularizers, both ranks) ───────────────

#[test]
fn zero_field_is_zero_planar() {
    let z = planar(&[0.0; 32], 2);
    assert_close(
        super::dispatch_bending_energy(&z, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "bending planar",
    );
    assert_close(
        super::dispatch_curvature(&z, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "curvature planar",
    );
    assert_close(
        super::dispatch_diffusion(&z, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "diffusion planar",
    );
    assert_close(
        super::dispatch_total_variation(&z, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "tv planar",
    );
    assert_close(
        super::dispatch_elastic(&z, 1.0, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "elastic planar",
    );
}

#[test]
fn zero_field_is_zero_volume() {
    let z = volume(&[0.0; 192], 3);
    assert_close(
        super::dispatch_bending_energy(&z, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "bending volume",
    );
    assert_close(
        super::dispatch_curvature(&z, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "curvature volume",
    );
    assert_close(
        super::dispatch_diffusion(&z, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "diffusion volume",
    );
    assert_close(
        super::dispatch_total_variation(&z, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "tv volume",
    );
    assert_close(
        super::dispatch_elastic(&z, 1.0, 1.0),
        0.0,
        ZERO_FIELD_LOSS_TOL,
        "elastic volume",
    );
}

// ── Diffusion: mean of |∇u|² ─────────────────────────────────────────────────

#[test]
fn diffusion_ramp_matches_closed_form_planar() {
    let f = planar(&width_ramp(32), 2);
    // mean(g_w²) = (3/4)·0.1² = 0.0075.
    assert_close(
        super::dispatch_diffusion(&f, 1.0),
        0.0075,
        FD_LOSS_TOL,
        "diffusion ramp planar",
    );
}

#[test]
fn diffusion_ramp_matches_closed_form_volume() {
    let f = volume(&width_ramp(192), 3);
    assert_close(
        super::dispatch_diffusion(&f, 1.0),
        0.0075,
        FD_LOSS_TOL,
        "diffusion ramp volume",
    );
}

#[test]
fn diffusion_weight_is_linear() {
    let f = planar(&width_ramp(32), 2);
    let base = super::dispatch_diffusion(&f, 1.0);
    let scaled = super::dispatch_diffusion(&f, 2.5);
    assert_close(
        scaled,
        2.5 * base,
        FD_LOSS_TOL,
        "diffusion weight linearity",
    );
}

// ── Total variation: mean of |∇u| ────────────────────────────────────────────

#[test]
fn total_variation_ramp_matches_closed_form_planar() {
    let f = planar(&width_ramp(32), 2);
    // mean(|g_w|) = (3/4)·0.1 = 0.075.
    assert_close(
        super::dispatch_total_variation(&f, 1.0),
        0.075,
        FD_LOSS_TOL,
        "tv ramp planar",
    );
}

#[test]
fn total_variation_ramp_matches_closed_form_volume() {
    let f = volume(&width_ramp(192), 3);
    assert_close(
        super::dispatch_total_variation(&f, 1.0),
        0.075,
        FD_LOSS_TOL,
        "tv ramp volume",
    );
}

// ── Elastic: α·mean(|∇u|²) + β·mean((div u)²) ────────────────────────────────

#[test]
fn elastic_ramp_matches_closed_form_planar() {
    let f = planar(&width_ramp(32), 2);
    // membrane mean = 0.0075; div u = ∂u_y/∂y on channel 1 (= g_w = 0.1 at the
    // three interior columns) → mean((div u)²) = (3/4)·0.1² = 0.0075.
    // α = β = 1 → 0.0075 + 0.0075 = 0.015.
    assert_close(
        super::dispatch_elastic(&f, 1.0, 1.0),
        0.015,
        FD_LOSS_TOL,
        "elastic ramp planar",
    );
}

#[test]
fn elastic_ramp_matches_closed_form_volume() {
    let f = volume(&width_ramp(192), 3);
    // membrane mean = 0.0075; div u = g_w on channel 2 → mean = 0.0075.
    assert_close(
        super::dispatch_elastic(&f, 1.0, 1.0),
        0.015,
        FD_LOSS_TOL,
        "elastic ramp volume",
    );
}

#[test]
fn elastic_alpha_beta_weight_terms_independently() {
    let f = planar(&width_ramp(32), 2);
    // membrane and divergence means are both 0.0075 here, so 2α + 3β form:
    assert_close(
        super::dispatch_elastic(&f, 2.0, 3.0),
        0.0075 * 5.0,
        FD_LOSS_TOL,
        "elastic weights",
    );
}

// ── Bending energy / curvature: mean of (∇²u)² ───────────────────────────────

#[test]
fn bending_energy_quadratic_matches_closed_form_planar() {
    let f = planar(&width_quadratic(32), 2);
    // Laplacian = 2 at the 4 interior voxels per plane, 0 border; over 16
    // voxels/plane mean((∇²u)²) = (4·2²)/16 = 1.0.
    assert_close(
        super::dispatch_bending_energy(&f, 1.0),
        1.0,
        FD_LOSS_TOL,
        "bending quad planar",
    );
}

#[test]
fn bending_energy_quadratic_matches_closed_form_volume() {
    let f = volume(&width_quadratic(192), 3);
    // Laplacian = 2 at the 8 interior voxels per volume, 0 border; over 64
    // voxels mean((∇²u)²) = (8·2²)/64 = 0.5.
    assert_close(
        super::dispatch_bending_energy(&f, 1.0),
        0.5,
        FD_LOSS_TOL,
        "bending quad volume",
    );
}

#[test]
fn bending_energy_equals_curvature_planar() {
    let vals: Vec<f32> = (0..32).map(|i| (i as f32 * 0.3).sin()).collect();
    let f1 = planar(&vals, 2);
    let f2 = planar(&vals, 2);
    let be = super::dispatch_bending_energy(&f1, 1.0);
    let cu = super::dispatch_curvature(&f2, 1.0);
    assert_close(be, cu, FD_LOSS_TOL, "bending ≡ curvature planar");
}

#[test]
fn bending_energy_equals_curvature_volume() {
    let vals: Vec<f32> = (0..192).map(|i| (i as f32 * 0.1).cos()).collect();
    let f1 = volume(&vals, 3);
    let f2 = volume(&vals, 3);
    let be = super::dispatch_bending_energy(&f1, 1.0);
    let cu = super::dispatch_curvature(&f2, 1.0);
    assert_close(be, cu, FD_LOSS_TOL, "bending ≡ curvature volume");
}
