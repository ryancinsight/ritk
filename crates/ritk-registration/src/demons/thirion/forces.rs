//! Optical-flow force computation and field clamping utilities.

use crate::deformable_field_ops::{warp_image};

/// Compute MSE = mean((F(p) − M_w(p))²) where M_w = warp(M, D).
pub(super) fn compute_mse(
    fixed: &[f32],
    moving: &[f32],
    dims: [usize; 3],
    dz: &[f32],
    dy: &[f32],
    dx: &[f32],
) -> f64 {
    let warped = warp_image(moving, dims, dz, dy, dx);
    fixed
        .iter()
        .zip(warped.iter())
        .map(|(&f, &m)| ((f - m) as f64).powi(2))
        .sum::<f64>()
        / fixed.len() as f64
}

/// Compute optical-flow Thirion forces given pre-computed fixed-image gradient.
///
/// Returns `(fz, fy, fx)` force components. Forces whose magnitude exceeds
/// `max_step_length` are rescaled.
pub fn thirion_forces(
    fixed: &[f32],
    m_warped: &[f32],
    grad_z: &[f32],
    grad_y: &[f32],
    grad_x: &[f32],
    max_step_length: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = fixed.len();
    let mut fz = vec![0.0_f32; n];
    let mut fy = vec![0.0_f32; n];
    let mut fx = vec![0.0_f32; n];

    thirion_forces_into(
        fixed,
        m_warped,
        grad_z,
        grad_y,
        grad_x,
        max_step_length,
        &mut fz,
        &mut fy,
        &mut fx,
    );

    (fz, fy, fx)
}

/// Compute optical-flow Thirion forces into caller-provided buffers.
pub(super) fn thirion_forces_into(
    fixed: &[f32],
    m_warped: &[f32],
    grad_z: &[f32],
    grad_y: &[f32],
    grad_x: &[f32],
    max_step_length: f32,
    fz: &mut [f32],
    fy: &mut [f32],
    fx: &mut [f32],
) {
    let sigma_x2 = max_step_length * max_step_length;

    for i in 0..fixed.len() {
        let diff = fixed[i] - m_warped[i];
        let gz = grad_z[i];
        let gy = grad_y[i];
        let gx = grad_x[i];
        let grad_sq = gz * gz + gy * gy + gx * gx;
        let denom = grad_sq + diff * diff / sigma_x2 + 1e-5;
        let scale = diff / denom;
        fz[i] = scale * gz;
        fy[i] = scale * gy;
        fx[i] = scale * gx;
    }

    clamp_field_magnitude(fz, fy, fx, max_step_length);
}

fn clamp_field_magnitude(fz: &mut [f32], fy: &mut [f32], fx: &mut [f32], max_length: f32) {
    let max2 = max_length * max_length;
    for i in 0..fz.len() {
        let mag2 = fz[i] * fz[i] + fy[i] * fy[i] + fx[i] * fx[i];
        if mag2 > max2 {
            let scale = max_length / mag2.sqrt();
            fz[i] *= scale;
            fy[i] *= scale;
            fx[i] *= scale;
        }
    }
}
