//! Thirion Demons deformable image registration.
//!
//! # Mathematical Specification
//!
//! Given fixed image F and moving image M, the Demons algorithm computes a
//! displacement field D : ℤ³ → ℝ³ that warps M toward F.
//!
//! **Optical-flow force at voxel p** (Thirion 1998):
//!
//!   f(p) = (F(p) − M_w(p)) · ∇F(p) / (|∇F(p)|² + (F(p)−M_w(p))²/σₓ² + ε)
//!
//! where:
//! - M_w(p) = M(p + D(p))  — current warp of M
//! - ∇F(p)               — gradient of the fixed image
//! - σₓ                  — max_step_length parameter (intensity normalisation)
//! - ε = 1e-5             — numerical floor
//!
//! **Per-iteration update:**
//! 1. Warp M with current D → M_w
//! 2. Compute forces f from (F, M_w, ∇F)
//! 3. Clamp |f| ≤ max_step_length
//! 4. Optional fluid regularisation: smooth f with G_{σ_fluid}
//! 5. Accumulate: D ← D + f
//! 6. Diffusive regularisation: D ← G_{σ_diff} ∗ D
//! 7. Compute MSE = mean((F − warp(M, D))²) for convergence tracking
//!
//! # References
//! - Thirion, J.-P. (1998). Image matching as a diffusion process: an analogy
//!   with Maxwell's demons. *Medical Image Analysis* 2(3):243–260.

mod forces;
mod registration;

#[cfg(test)]
mod tests;

#[cfg(test)]
pub(super) use forces::thirion_forces;
pub use registration::ThirionDemonsRegistration;

/// Re-export `thirion_forces_into` for zero-allocation loop variants.
pub(crate) use forces::thirion_forces_into;
