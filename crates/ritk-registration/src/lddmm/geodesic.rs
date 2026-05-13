//! Geodesic integration via forward Euler for LDDMM.

use crate::deformable_field_ops::{compose_fields, gaussian_smooth_inplace};

use super::adjoint::epdiff_adjoint;

/// Integrate the EPDiff equation forward from initial velocity `(v0z, v0y, v0x)`
/// for `num_steps` Euler steps and return the accumulated displacement field
/// at t = 1.
///
/// At each step k ∈ \[0, num\_steps):
/// 1. m = K\_σ ∗ v  (momentum)
/// 2. a = K\_σ ∗ ad\*\_v(m)
/// 3. v ← v − dt · a
/// 4. φ ← (id + v·dt) ∘ φ   (compose incremental step)
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
        let (mut adz, mut ady, mut adx) =
            epdiff_adjoint(&vz, &vy, &vx, &mz, &my, &mx, dims, spacing);
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
