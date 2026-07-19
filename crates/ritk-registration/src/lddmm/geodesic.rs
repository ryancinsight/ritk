//! Geodesic integration via forward Euler for LDDMM.

#[cfg(test)]
use crate::deformable_field_ops::gaussian_smooth_field_inplace;
#[cfg(test)]
use crate::deformable_field_ops::VelocityField;
use crate::deformable_field_ops::{
    compose_fields_into, FieldSmoother, VectorField, VectorFieldMut,
};

use super::adjoint::epdiff_adjoint_into;

/// Integrate the EPDiff equation forward from initial velocity `(v0z, v0y, v0x)`
/// for `num_steps` Euler steps and return the accumulated displacement field
/// at t = 1.
///
/// At each step k âˆˆ \[0, num\_steps):
/// 1. m = K\_Ïƒ âˆ— v  (momentum)
/// 2. a = K\_Ïƒ âˆ— ad\*\_v(m)
/// 3. v â† v âˆ’ dt Â· a
/// 4. Ï† â† (id + vÂ·dt) âˆ˜ Ï†   (compose incremental step)
///
/// All scratch buffers are pre-allocated once before the integration loop;
/// no per-step heap allocations occur. Returns owned displacement `Vec`s for
/// test convenience. For zero-allocation production paths use
/// [`integrate_geodesic_into_with_smoother`].
#[cfg(test)]
pub(super) fn integrate_geodesic(
    v0z: &[f32],
    v0y: &[f32],
    v0x: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    num_steps: usize,
    kernel_sigma: f64,
) -> VelocityField {
    let n = dims[0] * dims[1] * dims[2];
    let dt = 1.0 / num_steps as f32;

    let mut vz = v0z.to_vec();
    let mut vy = v0y.to_vec();
    let mut vx = v0x.to_vec();

    let mut dz = vec![0.0_f32; n];
    let mut dy = vec![0.0_f32; n];
    let mut dx = vec![0.0_f32; n];

    // Pre-allocated scratch buffers â€” no per-step heap allocations.
    let mut mz = vec![0.0_f32; n];
    let mut my = vec![0.0_f32; n];
    let mut mx = vec![0.0_f32; n];
    let mut adz = vec![0.0_f32; n];
    let mut ady = vec![0.0_f32; n];
    let mut adx = vec![0.0_f32; n];
    let mut comp_z = vec![0.0_f32; n];
    let mut comp_y = vec![0.0_f32; n];
    let mut comp_x = vec![0.0_f32; n];

    for _ in 0..num_steps {
        // 1. Momentum: m = K_Ïƒ âˆ— v.
        mz.copy_from_slice(&vz);
        my.copy_from_slice(&vy);
        mx.copy_from_slice(&vx);
        gaussian_smooth_field_inplace(&mut mz, &mut my, &mut mx, dims.into(), kernel_sigma);

        // 2. EPDiff adjoint ad*_v(m), then smooth.
        epdiff_adjoint_into(
            VectorField {
                z: &vz,
                y: &vy,
                x: &vx,
            },
            VectorField {
                z: &mz,
                y: &my,
                x: &mx,
            },
            dims,
            spacing,
            VectorFieldMut {
                z: &mut adz,
                y: &mut ady,
                x: &mut adx,
            },
        );
        gaussian_smooth_field_inplace(&mut adz, &mut ady, &mut adx, dims.into(), kernel_sigma);

        // 3. Velocity update: v â† v âˆ’ dt Â· K_Ïƒ âˆ— ad*_v(m).
        for i in 0..n {
            vz[i] -= dt * adz[i];
            vy[i] -= dt * ady[i];
            vx[i] -= dt * adx[i];
        }

        // 4. Step: adj â† v Â· dt (reuse adj buffers â€” adjoint no longer needed).
        for i in 0..n {
            adz[i] = vz[i] * dt;
            ady[i] = vy[i] * dt;
            adx[i] = vx[i] * dt;
        }

        // 5. Compose: Ï† â† (vÂ·dt) âˆ˜ Ï†.
        compose_fields_into(
            VectorField {
                z: &adz,
                y: &ady,
                x: &adx,
            },
            VectorField {
                z: &dz,
                y: &dy,
                x: &dx,
            },
            dims.into(),
            VectorFieldMut {
                z: &mut comp_z,
                y: &mut comp_y,
                x: &mut comp_x,
            },
        );
        dz.copy_from_slice(&comp_z);
        dy.copy_from_slice(&comp_y);
        dx.copy_from_slice(&comp_x);
    }

    VelocityField {
        z: dz,
        y: dy,
        x: dx,
    }
}

/// Zero-allocation EPDiff geodesic integration using a [`FieldSmoother`] for
/// Gaussian smoothing rather than a caller-provided `smooth_tmp` buffer.
///
/// When `smoother` is a [`crate::deformable_field_ops::GpuFieldSmoother`],
/// the per-step momentum and adjoint smoothing runs on the GPU â€” 10â€“50Ã—
/// faster than the CPU path for typical 256Â³ fields.
///
/// All other scratch buffers (`vel_*`, `mom_*`, `adj_*`, `comp_*`) are still
/// caller-provided; only the `smooth_tmp` buffer is replaced by the smoother.
#[allow(clippy::too_many_arguments)]
pub(super) fn integrate_geodesic_into_with_smoother(
    v0z: &[f32],
    v0y: &[f32],
    v0x: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    num_steps: usize,
    smoother: &mut impl FieldSmoother,
    // Output displacement
    out_z: &mut [f32],
    out_y: &mut [f32],
    out_x: &mut [f32],
    // Scratch: working velocity (copy of v0, updated in-place each step)
    vel_z: &mut [f32],
    vel_y: &mut [f32],
    vel_x: &mut [f32],
    // Scratch: momentum m = K_Ïƒ âˆ— v (overwritten each step)
    mom_z: &mut [f32],
    mom_y: &mut [f32],
    mom_x: &mut [f32],
    // Scratch: adjoint output, then reused for step vÂ·dt
    adj_z: &mut [f32],
    adj_y: &mut [f32],
    adj_x: &mut [f32],
    // Scratch: compose output (step âˆ˜ Ï†)
    comp_z: &mut [f32],
    comp_y: &mut [f32],
    comp_x: &mut [f32],
) {
    let n = dims[0] * dims[1] * dims[2];
    let dt = 1.0 / num_steps as f32;

    vel_z.copy_from_slice(v0z);
    vel_y.copy_from_slice(v0y);
    vel_x.copy_from_slice(v0x);
    out_z.iter_mut().for_each(|v| *v = 0.0);
    out_y.iter_mut().for_each(|v| *v = 0.0);
    out_x.iter_mut().for_each(|v| *v = 0.0);

    for _ in 0..num_steps {
        // 1. Momentum: m â† K_Ïƒ âˆ— v.
        mom_z.copy_from_slice(vel_z);
        mom_y.copy_from_slice(vel_y);
        mom_x.copy_from_slice(vel_x);
        smoother.smooth_field(mom_z, mom_y, mom_x);

        // 2. EPDiff adjoint ad*_v(m) â†’ adj.
        epdiff_adjoint_into(
            VectorField {
                z: vel_z,
                y: vel_y,
                x: vel_x,
            },
            VectorField {
                z: mom_z,
                y: mom_y,
                x: mom_x,
            },
            dims,
            spacing,
            VectorFieldMut {
                z: adj_z,
                y: adj_y,
                x: adj_x,
            },
        );
        smoother.smooth_field(adj_z, adj_y, adj_x);

        // 3. Velocity update: v â† v âˆ’ dt Â· (K_Ïƒ âˆ— ad*_v(m)).
        for i in 0..n {
            vel_z[i] -= dt * adj_z[i];
            vel_y[i] -= dt * adj_y[i];
            vel_x[i] -= dt * adj_x[i];
        }

        // 4. Step: adj â† v Â· dt (reuse adj buffers).
        for i in 0..n {
            adj_z[i] = vel_z[i] * dt;
            adj_y[i] = vel_y[i] * dt;
            adj_x[i] = vel_x[i] * dt;
        }

        // 5. Compose: Ï† â† step âˆ˜ Ï†.
        {
            let step_z: &[f32] = adj_z;
            let step_y: &[f32] = adj_y;
            let step_x: &[f32] = adj_x;
            let d_z: &[f32] = out_z;
            let d_y: &[f32] = out_y;
            let d_x: &[f32] = out_x;
            compose_fields_into(
                VectorField {
                    z: step_z,
                    y: step_y,
                    x: step_x,
                },
                VectorField {
                    z: d_z,
                    y: d_y,
                    x: d_x,
                },
                dims.into(),
                VectorFieldMut {
                    z: comp_z,
                    y: comp_y,
                    x: comp_x,
                },
            );
        }

        // 6. Update output displacement from compose result.
        out_z.copy_from_slice(comp_z);
        out_y.copy_from_slice(comp_y);
        out_x.copy_from_slice(comp_x);
    }
}
