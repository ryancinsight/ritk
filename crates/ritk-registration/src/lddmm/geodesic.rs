//! Geodesic integration via forward Euler for LDDMM.

use crate::deformable_field_ops::{compose_fields_into, gaussian_smooth_with_scratch};

use super::adjoint::epdiff_adjoint_into;

/// Integrate the EPDiff equation forward from initial velocity `(v0z, v0y, v0x)`
/// for `num_steps` Euler steps and return the accumulated displacement field
/// at t = 1. Caller provides all scratch buffers — performs zero heap allocation.
///
/// At each step k ∈ \[0, num\_steps):
/// 1. m = K\_σ ∗ v  (momentum)
/// 2. a = K\_σ ∗ ad\*\_v(m)
/// 3. v ← v − dt · a
/// 4. φ ← (id + v·dt) ∘ φ   (compose incremental step)
///
/// # Scratch buffer requirements
/// - `smooth_tmp`: length `n`, reused for all Gaussian smoothing calls
/// - `mz/my/mx`: each length `n`, momentum storage (reused each step)
/// - `adz/ady/adx`: each length `n`, adjoint storage (reused each step)
/// - `dz/dy/dx`: displacement output (same buffers as return)
/// - `step_z/step_y/step_x`: each length `n`, step vectors (reused each step)
/// - `comp_z/comp_y/comp_x`: each length `n`, composition scratch (contents arbitrary on return)
#[allow(clippy::too_many_arguments)]
pub(super) fn integrate_geodesic_into(
    v0z: &[f32],
    v0y: &[f32],
    v0x: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    num_steps: usize,
    kernel_sigma: f64,
    dz: &mut [f32],
    dy: &mut [f32],
    dx: &mut [f32],
    // per-step scratch buffers (reused across all steps)
    smooth_tmp: &mut [f32],
    mz: &mut [f32],
    my: &mut [f32],
    mx: &mut [f32],
    adz: &mut [f32],
    ady: &mut [f32],
    adx: &mut [f32],
    step_z: &mut [f32],
    step_y: &mut [f32],
    step_x: &mut [f32],
    comp_z: &mut [f32],
    comp_y: &mut [f32],
    comp_x: &mut [f32],
) {
    let n = dims[0] * dims[1] * dims[2];
    let dt = 1.0 / num_steps as f32;

    let mut vz = v0z.to_vec();
    let mut vy = v0y.to_vec();
    let mut vx = v0x.to_vec();

    // Zero the displacement output (identity initialisation).
    dz.iter_mut().for_each(|v| *v = 0.0);
    dy.iter_mut().for_each(|v| *v = 0.0);
    dx.iter_mut().for_each(|v| *v = 0.0);

    for _ in 0..num_steps {
        // 1. Momentum: m = K_σ ∗ v  (clone v into m then smooth in-place)
        mz.copy_from_slice(&vz);
        my.copy_from_slice(&vy);
        mx.copy_from_slice(&vx);
        gaussian_smooth_with_scratch(mz, dims, kernel_sigma, smooth_tmp);
        gaussian_smooth_with_scratch(my, dims, kernel_sigma, smooth_tmp);
        gaussian_smooth_with_scratch(mx, dims, kernel_sigma, smooth_tmp);

        // 2. EPDiff adjoint ad*_v(m), then smooth.
        epdiff_adjoint_into(&vz, &vy, &vx, mz, my, mx, dims, spacing, adz, ady, adx);
        gaussian_smooth_with_scratch(adz, dims, kernel_sigma, smooth_tmp);
        gaussian_smooth_with_scratch(ady, dims, kernel_sigma, smooth_tmp);
        gaussian_smooth_with_scratch(adx, dims, kernel_sigma, smooth_tmp);

        // 3. Velocity update: v ← v − dt · K_σ ∗ ad*_v(m).
        for i in 0..n {
            vz[i] -= dt * adz[i];
            vy[i] -= dt * ady[i];
            vx[i] -= dt * adx[i];
        }

        // 4. Compose displacement: φ ← (v·dt) ∘ φ.
        for i in 0..n {
            step_z[i] = vz[i] * dt;
            step_y[i] = vy[i] * dt;
            step_x[i] = vx[i] * dt;
        }
        compose_fields_into(
            step_z, step_y, step_x, dz, dy, dx, dims, comp_z, comp_y, comp_x,
        );
        dz.swap_with_slice(comp_z);
        dy.swap_with_slice(comp_y);
        dx.swap_with_slice(comp_x);
    }
}

/// Integrate the EPDiff equation forward (allocating convenience wrapper).
#[cfg(test)]
pub(super) fn integrate_geodesic(
    v0z: &[f32],
    v0y: &[f32],
    v0x: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    num_steps: usize,
    kernel_sigma: f64,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = dims[0] * dims[1] * dims[2];
    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];
    let mut smooth_tmp = vec![0.0_f32; n];
    let mut mz = vec![0.0_f32; n];
    let mut my = vec![0.0_f32; n];
    let mut mx = vec![0.0_f32; n];
    let mut adz = vec![0.0_f32; n];
    let mut ady = vec![0.0_f32; n];
    let mut adx = vec![0.0_f32; n];
    let mut step_z = vec![0.0_f32; n];
    let mut step_y = vec![0.0_f32; n];
    let mut step_x = vec![0.0_f32; n];
    let mut comp_z = vec![0.0_f32; n];
    let mut comp_y = vec![0.0_f32; n];
    let mut comp_x = vec![0.0_f32; n];
    integrate_geodesic_into(
        v0z,
        v0y,
        v0x,
        dims,
        spacing,
        num_steps,
        kernel_sigma,
        &mut dz,
        &mut dy,
        &mut dx,
        &mut smooth_tmp,
        &mut mz,
        &mut my,
        &mut mx,
        &mut adz,
        &mut ady,
        &mut adx,
        &mut step_z,
        &mut step_y,
        &mut step_x,
        &mut comp_z,
        &mut comp_y,
        &mut comp_x,
    );
    (dz, dy, dx)
}
