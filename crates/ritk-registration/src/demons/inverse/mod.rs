//! Exact and approximate inverse displacement field computation.
//!
//! Two inversion strategies are provided based on the field type.
//!
//! ## SVF Exact Inverse (Diffeomorphic Demons)
//!
//! For a stationary velocity field `v`, the forward diffeomorphism is `φ = exp(v)`.
//! The exact inverse is `φ^{-1} = exp(−v)` — negate all velocity components.
//!
//! ## Fixed-Point Iterative Inverse (Thirion / Symmetric Demons)
//!
//! For a general displacement field `u`, apply Christensen & Johnson (2001)
//! fixed-point iteration to converge on `u^{-1}`.
//!
//! # References
//! - Christensen, G. E. & Johnson, H. J. (2001). Consistent image registration.
//!   *IEEE Trans. Med. Imaging* 20(7):568–582.

mod displacement;
mod svf;

#[cfg(test)]
mod tests;

pub use displacement::{invert_displacement_field, InverseFieldConfig};
pub use svf::invert_velocity_field;
pub(crate) use svf::invert_velocity_field_into;
