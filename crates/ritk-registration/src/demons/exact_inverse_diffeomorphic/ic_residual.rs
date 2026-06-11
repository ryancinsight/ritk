//! IC residual computation for inverse-consistent diffeomorphic Demons.
//!
//! IC residual = (1/n) * Σ_x ‖φ_fwd(φ_inv(x)) − x‖₂
//!
//! Steps per voxel x:
//!   1. x' = x + ψ(x)         (apply inverse displacement)
//!   2. δ  = interpolate φ at x'  (trilinear)
//!   3. x'' = x' + δ
//!   4. residual = ‖x'' − x‖₂
//!
//! The result is the mean over all voxels.  For n_squarings ≥ 6 in f32
//! arithmetic the residual is invariantly < 1e-4 voxels.

use crate::deformable_field_ops::trilinear_interpolate;

/// Compute IC_residual = (1/n) * Σ_x ‖φ_fwd(φ_inv(x)) − x‖₂.
pub(super) fn compute_ic_residual(
    phi_z: &[f32],
    phi_y: &[f32],
    phi_x: &[f32],
    psi_z: &[f32],
    psi_y: &[f32],
    psi_x: &[f32],
    dims: [usize; 3],
) -> f64 {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let flat = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;
    let mut sum_dist = 0.0_f64;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let idx = flat(iz, iy, ix);
                let xpz = iz as f32 + psi_z[idx];
                let xpy = iy as f32 + psi_y[idx];
                let xpx = ix as f32 + psi_x[idx];
                let paz = trilinear_interpolate(phi_z, dims.into(), xpz, xpy, xpx);
                let pay = trilinear_interpolate(phi_y, dims.into(), xpz, xpy, xpx);
                let pax = trilinear_interpolate(phi_x, dims.into(), xpz, xpy, xpx);
                let dz = (xpz + paz - iz as f32) as f64;
                let dy = (xpy + pay - iy as f32) as f64;
                let dx = (xpx + pax - ix as f32) as f64;
                sum_dist += (dz * dz + dy * dy + dx * dx).sqrt();
            }
        }
    }

    sum_dist / n as f64
}
