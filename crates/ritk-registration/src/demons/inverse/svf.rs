//! Exact inverse of a stationary velocity field via negation.
//!
//! # Mathematical Basis
//!
//! For a stationary velocity field `v` the exponential map gives `φ = exp(v)`.
//! The exact inverse is `φ^{-1} = exp(−v)`, obtained by negating every
//! component.  This follows directly from the Baker-Campbell-Hausdorff identity:
//! `exp(v) ∘ exp(−v) = exp(v − v) = exp(0) = id`.
//!
//! This is a zero-cost O(n) operation — no integration, no iteration.

use crate::deformable_field_ops::VelocityField;

/// Write the exact inverse of a stationary velocity field into caller buffers.
///
/// Performs zero heap allocation: `inv_z[i] = -vel_z[i]` for each component.
pub fn invert_velocity_field_into(
    vel_z: &[f32],
    vel_y: &[f32],
    vel_x: &[f32],
    inv_z: &mut [f32],
    inv_y: &mut [f32],
    inv_x: &mut [f32],
) {
    for i in 0..vel_z.len() {
        inv_z[i] = -vel_z[i];
        inv_y[i] = -vel_y[i];
        inv_x[i] = -vel_x[i];
    }
}

/// Compute the exact inverse of a stationary velocity field.
///
/// Returns `VelocityField { z, y, x }` — negated velocity components as new `Vec<f32>`.
pub fn invert_velocity_field(vel_z: &[f32], vel_y: &[f32], vel_x: &[f32]) -> VelocityField {
    let n = vel_z.len();
    let mut inv_z = vec![0.0_f32; n];
    let mut inv_y = vec![0.0_f32; n];
    let mut inv_x = vec![0.0_f32; n];
    invert_velocity_field_into(vel_z, vel_y, vel_x, &mut inv_z, &mut inv_y, &mut inv_x);
    VelocityField {
        z: inv_z,
        y: inv_y,
        x: inv_x,
    }
}
