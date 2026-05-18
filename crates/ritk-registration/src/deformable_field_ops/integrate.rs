//! Scaling-and-squaring exponential map for stationary velocity fields.

use super::compose::compose_fields_into;

/// Compute the exponential map `exp(v)` into caller-provided output and scratch buffers.
///
/// Equivalent to [`scaling_and_squaring`] but performs zero heap allocation.
/// All 9 buffers must have length `dims[0] * dims[1] * dims[2]`.
///
/// # Buffer layout
/// - `out_z`, `out_y`, `out_x`: initially scaled velocity `v / 2^n_steps`;
///   on return, contain the final displacement field `exp(v)`.
/// - `scratch_z`, `scratch_y`, `scratch_x`: temporary composition buffers;
///   contents are arbitrary on return.
pub(crate) fn scaling_and_squaring_into(
    vz: &[f32],
    vy: &[f32],
    vx: &[f32],
    dims: [usize; 3],
    n_steps: usize,
    out_z: &mut [f32],
    out_y: &mut [f32],
    out_x: &mut [f32],
    scratch_z: &mut [f32],
    scratch_y: &mut [f32],
    scratch_x: &mut [f32],
) {
    let scale = 1.0_f32 / (1u32 << n_steps) as f32;
    for i in 0..out_z.len() {
        out_z[i] = vz[i] * scale;
        out_y[i] = vy[i] * scale;
        out_x[i] = vx[i] * scale;
    }
    for _ in 0..n_steps {
        compose_fields_into(
            out_z, out_y, out_x, out_z, out_y, out_x, dims, scratch_z, scratch_y, scratch_x,
        );
        out_z.swap_with_slice(scratch_z);
        out_y.swap_with_slice(scratch_y);
        out_x.swap_with_slice(scratch_x);
    }
}

/// Compute the exponential map `exp(v)` of a stationary velocity field `v`
/// via the scaling-and-squaring algorithm.
///
/// # Algorithm
/// 1. Scale: `φ ← v / 2^n_steps`
/// 2. Square n_steps times: `φ ← φ ∘ φ`
///
/// Using `n_steps = 6` corresponds to 64 integration steps and is the
/// standard choice for Diffeomorphic Demons (Vercauteren et al. 2009).
///
/// # Invariants
/// - For `v = 0` the result is the identity displacement `(0, 0, 0)`.
/// - For small `v`, `exp(v) ≈ v` (first-order approximation).
pub(crate) fn scaling_and_squaring(
    vz: &[f32],
    vy: &[f32],
    vx: &[f32],
    dims: [usize; 3],
    n_steps: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n = vz.len();
    let mut phiz = vec![0.0_f32; n];
    let mut phiy = vec![0.0_f32; n];
    let mut phix = vec![0.0_f32; n];
    let mut scratch_z = vec![0.0_f32; n];
    let mut scratch_y = vec![0.0_f32; n];
    let mut scratch_x = vec![0.0_f32; n];
    scaling_and_squaring_into(
        vz,
        vy,
        vx,
        dims,
        n_steps,
        &mut phiz,
        &mut phiy,
        &mut phix,
        &mut scratch_z,
        &mut scratch_y,
        &mut scratch_x,
    );
    (phiz, phiy, phix)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scaling-and-squaring of the zero field is the zero field.
    #[test]
    fn scaling_and_squaring_zero_field() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let vz = vec![0.0_f32; n];
        let vy = vec![0.0_f32; n];
        let vx = vec![0.0_f32; n];
        let (phiz, phiy, phix) = scaling_and_squaring(&vz, &vy, &vx, dims, 6);
        for i in 0..n {
            assert!(phiz[i].abs() < 1e-5, "phiz[{i}] = {} != 0", phiz[i]);
            assert!(phiy[i].abs() < 1e-5, "phiy[{i}] = {} != 0", phiy[i]);
            assert!(phix[i].abs() < 1e-5, "phix[{i}] = {} != 0", phix[i]);
        }
    }

    /// For a small constant velocity field, exp(v) ≈ v (first-order approximation).
    #[test]
    fn scaling_and_squaring_small_velocity_approx_identity() {
        let dims = [4usize, 4, 4];
        let n = 4 * 4 * 4;
        let vz = vec![0.0_f32; n];
        let vy = vec![0.0_f32; n];
        let vx = vec![0.01_f32; n];
        let (phiz, phiy, phix) = scaling_and_squaring(&vz, &vy, &vx, dims, 6);
        for i in 0..n {
            assert!(phiz[i].abs() < 1e-4, "phiz should be ~0, got {}", phiz[i]);
            assert!(phiy[i].abs() < 1e-4, "phiy should be ~0, got {}", phiy[i]);
            assert!(
                (phix[i] - 0.01).abs() < 0.002,
                "phix should be ~0.01, got {}",
                phix[i]
            );
        }
    }
}
