//! Symmetric Demons deformable image registration.
//!
//! # Mathematical Specification
//!
//! The Symmetric Demons variant (Pennec et al. 1999, extended) uses gradient
//! information from **both** the fixed and warped moving images to compute
//! registration forces. This makes the algorithm approximately symmetric with
//! respect to swapping fixed and moving images.
//!
//! **Symmetric force at voxel p:**
//!
//! f(p) = (F(p) − M_w(p)) · [∇F(p) + ∇M_w(p)] /
//! (|∇F(p) + ∇M_w(p)|² / 4 + (F(p) − M_w(p))² / σâ‚“² + ε)
//!
//! where:
//! - M_w(p) = M(p + D(p)) — current warp of M
//! - ∇F(p) — gradient of the fixed image (constant)
//! - ∇M_w(p) — gradient of the warped moving image (recomputed each iteration)
//! - σâ‚“ — max_step_length parameter
//! - ε = 1e-5 — numerical floor
//!
//! The |∇F + ∇M_w|² / 4 denominator term (dividing by 4 instead of 1) comes
//! from the symmetric formulation where the combined gradient is the average of
//! the two individual gradients, so the effective gradient magnitude is halved.
//!
//! **Per-iteration update:**
//! 1. Warp M with current D → M_w
//! 2. Compute ∇F (fixed, cached) and ∇M_w (recomputed each iteration)
//! 3. Compute symmetric forces f
//! 4. Clamp |f| ≤ max_step_length
//! 5. Optional fluid regularisation: smooth f with G_{σ_fluid}
//! 6. Accumulate: D ← D + f
//! 7. Diffusive regularisation: D ← G_{σ_diff} ∗ D
//! 8. Compute MSE = mean((F − M_w)²) (reuses M_w from step 1)
//!
//! # Memory discipline
//! All scratch buffers are pre-allocated before the iteration loop.
//! The loop body performs **zero heap allocations**; `_into` variants of
//! image warp, gradient, force, and Gaussian smoothing write into
//! caller-provided buffers. Total pre-allocation: ~14n f32
//! (3 displacement + 1 warped + 3 fixed gradient + 3 moving gradient +
//! 3 forces + 1 smooth scratch = 14n).
//!
//! # Symmetry Property
//! When fixed and moving are swapped, the force direction reverses. More
//! precisely: for images F and M with displacement D_{FM}, and images M and F
//! with displacement D_{MF}, we expect D_{FM} ≈ −D_{MF} for small deformations.
//!
//! # References
//! - Pennec, X., Cachier, P. & Ayache, N. (1999). Understanding the
//!   "Demon's Algorithm": 3D Non-Rigid Registration by Gradient Descent.
//!   *MICCAI*, LNCS 1679:597–605.
//! - Cachier, P., Bardinet, E., Dormont, D., Pennec, X. & Ayache, N. (2003).
//!   Iconic feature based nonrigid registration: the PASHA algorithm.
//!   *CVIU* 89(2–3):272–298.

mod engine;

use super::config::DemonsConfig;

// ── Public types ──────────────────────────────────────────────────────────────

/// Symmetric Demons registration.
///
/// Extends the classic Thirion Demons by incorporating gradient information
/// from both the fixed and the warped moving images. The resulting forces
/// are approximately symmetric: swapping fixed and moving produces forces of
/// opposite sign.
#[derive(Debug, Clone)]
pub struct SymmetricDemonsRegistration {
    /// Algorithm configuration (shared with Thirion Demons).
    pub config: DemonsConfig,
}

#[cfg(test)]
mod tests;
