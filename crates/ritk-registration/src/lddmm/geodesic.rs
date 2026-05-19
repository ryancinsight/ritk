//! Geodesic integration via forward Euler for LDDMM.

use crate::deformable_field_ops::{
    compose_fields_into, gaussian_smooth_with_scratch, VectorField3D, VectorFieldMut3D,
};
#[cfg(test)]
use crate::deformable_field_ops::{compose_fields, gaussian_smooth_inplace};

use super::adjoint::epdiff_adjoint_into;

/// Integrate the EPDiff equation forward from initial velocity `(v0z, v0y, v0x)`
/// for `num_steps` Euler steps and return the accumulated displacement field
/// at t = 1.
///
/// At each step k ∈ \[0, num\_steps):
/// 1. m = K\_σ ∗ v  (momentum)
/// 2. a = K\_σ ∗ ad\*\_v(m)
/// 3. v ← v − dt · a
/// 4. φ ← (id + v·dt) ∘ φ   (compose incremental step)
///
/// This is the allocating reference implementation used in tests.
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
    let dt = 1.0 / num_steps as f32;

    let mut vz = v0z.to_vec();
    let mut vy = v0y.to_vec();
    let mut vx = v0x.to_vec();

    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];

    for _ in 0..num_steps {
        // 1. Momentum: m = K_σ ∗ v.
        let mut mz = vz.clone();
        let mut my = vy.clone();
        let mut mx = vx.clone();
        gaussian_smooth_inplace(&mut mz, dims, kernel_sigma);
        gaussian_smooth_inplace(&mut my, dims, kernel_sigma);
        gaussian_smooth_inplace(&mut mx, dims, kernel_sigma);

        // 2. EPDiff adjoint ad*_v(m), then smooth.
        let mut adz = vec![0.0_f32; n];
        let mut ady = vec![0.0_f32; n];
        let mut adx = vec![0.0_f32; n];
        epdiff_adjoint_into(
            VectorField3D { z: &vz, y: &vy, x: &vx },
            VectorField3D { z: &mz, y: &my, x: &mx },
            dims,
            spacing,
            VectorFieldMut3D { z: &mut adz, y: &mut ady, x: &mut adx },
        );
        gaussian_smooth_inplace(&mut adz, dims, kernel_sigma);
        gaussian_smooth_inplace(&mut ady, dims, kernel_sigma);
        gaussian_smooth_inplace(&mut adx, dims, kernel_sigma);

        // 3. Velocity update: v ← v − dt · K_σ ∗ ad*_v(m).
        for i in 0..n {
            vz[i] -= dt * adz[i];
            vy[i] -= dt * ady[i];
            vx[i] -= dt * adx[i];
        }

        // 4. Compose displacement: φ ← (v·dt) ∘ φ.
        let step_z: Vec<f32> = vz.iter().map(|&v| v * dt).collect();
        let step_y: Vec<f32> = vy.iter().map(|&v| v * dt).collect();
        let step_x: Vec<f32> = vx.iter().map(|&v| v * dt).collect();

        let composed = compose_fields(&step_z, &step_y, &step_x, &dz, &dy, &dx, dims);
        dz = composed.0;
        dy = composed.1;
        dx = composed.2;
    }

    (dz, dy, dx)
}

/// Zero-allocation variant of [`integrate_geodesic`].
///
/// Integrates the EPDiff geodesic equation into caller-provided output buffers.
/// The loop body performs zero heap allocations; all intermediate values are
/// written into the caller-supplied scratch buffers.
///
/// # Buffer roles
/// - `out_z/y/x`: output displacement field φ (initialized to 0 on entry).
/// - `smooth_tmp`: single-component scratch for [`gaussian_smooth_with_scratch`].
/// - `vel_z/y/x`: working velocity copy (initialized from v0, updated each step).
/// - `mom_z/y/x`: momentum m = K_σ ∗ v (overwritten each step).
/// - `adj_z/y/x`: adjoint ad*_v(m) then step v·dt (overwritten each step).
/// - `comp_z/y/x`: compose output (v·dt) ∘ φ (overwritten each step).
///
/// All buffers must have length `dims[0] * dims[1] * dims[2]`.
#[allow(clippy::too_many_arguments)]
pub(super) fn integrate_geodesic_into(
    v0z: &[f32],
    v0y: &[f32],
    v0x: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    num_steps: usize,
    kernel_sigma: f64,
    // Output displacement
    out_z: &mut [f32],
    out_y: &mut [f32],
    out_x: &mut [f32],
    // Scratch: gaussian smooth
    smooth_tmp: &mut [f32],
    // Scratch: working velocity (copy of v0, updated in-place each step)
    vel_z: &mut [f32],
    vel_y: &mut [f32],
    vel_x: &mut [f32],
    // Scratch: momentum m = K_σ ∗ v (overwritten each step)
    mom_z: &mut [f32],
    mom_y: &mut [f32],
    mom_x: &mut [f32],
    // Scratch: adjoint output, then reused for step v·dt
    adj_z: &mut [f32],
    adj_y: &mut [f32],
    adj_x: &mut [f32],
    // Scratch: compose output (step ∘ φ)
    comp_z: &mut [f32],
    comp_y: &mut [f32],
    comp_x: &mut [f32],
) {
    let n = dims[0] * dims[1] * dims[2];
    let dt = 1.0 / num_steps as f32;

    // Initialize working velocity from v0 and output displacement to 0.
    vel_z.copy_from_slice(v0z);
    vel_y.copy_from_slice(v0y);
    vel_x.copy_from_slice(v0x);
    out_z.iter_mut().for_each(|v| *v = 0.0);
    out_y.iter_mut().for_each(|v| *v = 0.0);
    out_x.iter_mut().for_each(|v| *v = 0.0);

    for _ in 0..num_steps {
        // 1. Momentum: m ← K_σ ∗ v.
        //    Copy current velocity into mom buffers, then smooth in-place.
        mom_z.copy_from_slice(vel_z);
        mom_y.copy_from_slice(vel_y);
        mom_x.copy_from_slice(vel_x);
        gaussian_smooth_with_scratch(mom_z, dims, kernel_sigma, smooth_tmp);
        gaussian_smooth_with_scratch(mom_y, dims, kernel_sigma, smooth_tmp);
        gaussian_smooth_with_scratch(mom_x, dims, kernel_sigma, smooth_tmp);

        // 2. EPDiff adjoint ad*_v(m) → adj.
        epdiff_adjoint_into(
            VectorField3D { z: vel_z, y: vel_y, x: vel_x },
            VectorField3D { z: mom_z, y: mom_y, x: mom_x },
            dims,
            spacing,
            VectorFieldMut3D { z: adj_z, y: adj_y, x: adj_x },
        );
        // Smooth adjoint: K_σ ∗ ad*_v(m).
        gaussian_smooth_with_scratch(adj_z, dims, kernel_sigma, smooth_tmp);
        gaussian_smooth_with_scratch(adj_y, dims, kernel_sigma, smooth_tmp);
        gaussian_smooth_with_scratch(adj_x, dims, kernel_sigma, smooth_tmp);

        // 3. Velocity update: v ← v − dt · (K_σ ∗ ad*_v(m)).
        for i in 0..n {
            vel_z[i] -= dt * adj_z[i];
            vel_y[i] -= dt * adj_y[i];
            vel_x[i] -= dt * adj_x[i];
        }

        // 4. Step: adj ← v · dt (reuse adj buffers — adjoint no longer needed).
        for i in 0..n {
            adj_z[i] = vel_z[i] * dt;
            adj_y[i] = vel_y[i] * dt;
            adj_x[i] = vel_x[i] * dt;
        }

        // 5. Compose: φ ← step ∘ φ, written into comp buffers.
        {
            let step_z: &[f32] = adj_z;
            let step_y: &[f32] = adj_y;
            let step_x: &[f32] = adj_x;
            let d_z: &[f32] = out_z;
            let d_y: &[f32] = out_y;
            let d_x: &[f32] = out_x;
            compose_fields_into(
                VectorField3D { z: step_z, y: step_y, x: step_x },
                VectorField3D { z: d_z, y: d_y, x: d_x },
                dims,
                VectorFieldMut3D { z: comp_z, y: comp_y, x: comp_x },
            );
        }

        // 6. Update output displacement from compose result.
        out_z.copy_from_slice(comp_z);
        out_y.copy_from_slice(comp_y);
        out_x.copy_from_slice(comp_x);
    }
}
