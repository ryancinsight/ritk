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

/// Compute the exact inverse of a stationary velocity field.
///
/// Returns `(inv_z, inv_y, inv_x)` — negated velocity components as new `Vec<f32>`.
pub fn invert_velocity_field(
    vel_z: &[f32],
    vel_y: &[f32],
    vel_x: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let inv_z: Vec<f32> = vel_z.iter().map(|&v| -v).collect();
    let inv_y: Vec<f32> = vel_y.iter().map(|&v| -v).collect();
    let inv_x: Vec<f32> = vel_x.iter().map(|&v| -v).collect();
    (inv_z, inv_y, inv_x)
}
