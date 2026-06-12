//! Scaling-and-squaring exponential map for stationary velocity fields.

use super::compose::compose_fields_into;
use super::{VectorField, VectorFieldMut, VelocityField};
use ritk_spatial::VolumeDims;

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
    dims: VolumeDims,
    n_steps: usize,
) -> VelocityField {
    let scale = 1.0_f32 / (1u32 << n_steps) as f32;

    let mut phiz: Vec<f32> = vz.iter().map(|&v| v * scale).collect();
    let mut phiy: Vec<f32> = vy.iter().map(|&v| v * scale).collect();
    let mut phix: Vec<f32> = vx.iter().map(|&v| v * scale).collect();

    let n = phiz.len();
    let mut next_z = vec![0.0_f32; n];
    let mut next_y = vec![0.0_f32; n];
    let mut next_x = vec![0.0_f32; n];

    for _ in 0..n_steps {
        compose_fields_into(
            VectorField {
                z: &phiz,
                y: &phiy,
                x: &phix,
            },
            VectorField {
                z: &phiz,
                y: &phiy,
                x: &phix,
            },
            dims,
            VectorFieldMut {
                z: &mut next_z,
                y: &mut next_y,
                x: &mut next_x,
            },
        );
        std::mem::swap(&mut phiz, &mut next_z);
        std::mem::swap(&mut phiy, &mut next_y);
        std::mem::swap(&mut phix, &mut next_x);
    }

    VelocityField {
        z: phiz,
        y: phiy,
        x: phix,
    }
}

/// Zero-allocation variant: computes `exp(v)` into caller-provided buffers.
///
/// Writes the exponential map of `v` into `out_z/y/x`.
/// `scratch_z/y/x` are internal ping-pong buffers owned by the caller to
/// avoid per-call allocation. All six mutable slices must have the same
/// length as `vz`.
///
/// # Algorithm
/// Identical to [`scaling_and_squaring`]: scale v by `1 / 2^n_steps`, then
/// compose with itself `n_steps` times.
///
/// # Invariants
/// Same correctness invariants as [`scaling_and_squaring`].
pub(crate) fn scaling_and_squaring_into(
    vz: &[f32],
    vy: &[f32],
    vx: &[f32],
    dims: VolumeDims,
    n_steps: usize,
    out_z: &mut [f32],
    out_y: &mut [f32],
    out_x: &mut [f32],
    scratch_z: &mut [f32],
    scratch_y: &mut [f32],
    scratch_x: &mut [f32],
) {
    let scale = 1.0_f32 / (1u32 << n_steps) as f32;
    let n = vz.len();
    for i in 0..n {
        out_z[i] = vz[i] * scale;
        out_y[i] = vy[i] * scale;
        out_x[i] = vx[i] * scale;
    }
    for _ in 0..n_steps {
        // Reborrow out as shared slices so compose_fields_into can read from
        // them while writing into the disjoint scratch buffers.
        let phi_z: &[f32] = out_z;
        let phi_y: &[f32] = out_y;
        let phi_x: &[f32] = out_x;
        compose_fields_into(
            VectorField {
                z: phi_z,
                y: phi_y,
                x: phi_x,
            },
            VectorField {
                z: phi_z,
                y: phi_y,
                x: phi_x,
            },
            dims,
            VectorFieldMut {
                z: scratch_z,
                y: scratch_y,
                x: scratch_x,
            },
        );
        out_z.copy_from_slice(scratch_z);
        out_y.copy_from_slice(scratch_y);
        out_x.copy_from_slice(scratch_x);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scaling-and-squaring of the zero field is the zero field.
    #[test]
    fn scaling_and_squaring_zero_field() {
        let dims = VolumeDims::new([4, 4, 4]);
        let n = 4 * 4 * 4;
        let vz = vec![0.0_f32; n];
        let vy = vec![0.0_f32; n];
        let vx = vec![0.0_f32; n];
        let phi = scaling_and_squaring(&vz, &vy, &vx, dims, 6);
        for i in 0..n {
            assert!(phi.z[i].abs() < 1e-5, "phiz[{i}] = {} != 0", phi.z[i]);
            assert!(phi.y[i].abs() < 1e-5, "phiy[{i}] = {} != 0", phi.y[i]);
            assert!(phi.x[i].abs() < 1e-5, "phix[{i}] = {} != 0", phi.x[i]);
        }
    }

    /// For a small constant velocity field, exp(v) ≈ v (first-order approximation).
    #[test]
    fn scaling_and_squaring_small_velocity_approx_identity() {
        let dims = VolumeDims::new([4, 4, 4]);
        let n = 4 * 4 * 4;
        let vz = vec![0.0_f32; n];
        let vy = vec![0.0_f32; n];
        let vx = vec![0.01_f32; n];
        let phi = scaling_and_squaring(&vz, &vy, &vx, dims, 6);
        for i in 0..n {
            assert!(phi.z[i].abs() < 1e-4, "phiz should be ~0, got {}", phi.z[i]);
            assert!(phi.y[i].abs() < 1e-4, "phiy should be ~0, got {}", phi.y[i]);
            assert!(
                (phi.x[i] - 0.01).abs() < 0.002,
                "phix should be ~0.01, got {}",
                phi.x[i]
            );
        }
    }

    /// `scaling_and_squaring_into` produces the same result as `scaling_and_squaring`.
    #[test]
    fn scaling_and_squaring_into_matches_allocating() {
        let dims = VolumeDims::new([4, 4, 4]);
        let n = 4 * 4 * 4;
        let vz: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001).collect();
        let vy: Vec<f32> = (0..n).map(|i| (i as f32) * -0.001).collect();
        let vx: Vec<f32> = (0..n).map(|i| ((i % 4) as f32) * 0.002).collect();

        let ref_phi = scaling_and_squaring(&vz, &vy, &vx, dims, 6);

        let mut out_z = vec![0.0_f32; n];
        let mut out_y = vec![0.0_f32; n];
        let mut out_x = vec![0.0_f32; n];
        let mut sc_z = vec![0.0_f32; n];
        let mut sc_y = vec![0.0_f32; n];
        let mut sc_x = vec![0.0_f32; n];
        scaling_and_squaring_into(
            &vz, &vy, &vx, dims, 6, &mut out_z, &mut out_y, &mut out_x, &mut sc_z, &mut sc_y,
            &mut sc_x,
        );

        for i in 0..n {
            assert!(
                (out_z[i] - ref_phi.z[i]).abs() < 1e-6,
                "z[{i}]: into={} ref={}",
                out_z[i],
                ref_phi.z[i]
            );
            assert!(
                (out_y[i] - ref_phi.y[i]).abs() < 1e-6,
                "y[{i}]: into={} ref={}",
                out_y[i],
                ref_phi.y[i]
            );
            assert!(
                (out_x[i] - ref_phi.x[i]).abs() < 1e-6,
                "x[{i}]: into={} ref={}",
                out_x[i],
                ref_phi.x[i]
            );
        }
    }
}
