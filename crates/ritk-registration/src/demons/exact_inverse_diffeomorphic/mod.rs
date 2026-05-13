//! Inverse-consistent diffeomorphic Demons registration (GAP-R02b).
//!
//! # Mathematical Specification
//!
//! For SVF parametrization, inverse consistency is EXACT by construction:
//!   exp(v) ∘ exp(−v) = exp(v − v) = exp(0) = id
//!
//! ## Bilateral Objective
//!
//!   E(v) = (1−w)·‖F − M∘exp(v)‖² + w·‖M − F∘exp(−v)‖²
//!
//! ## Update Rule (first-order BCH)
//!
//!   v ← v + (1−w)·u_fwd − w·u_bwd
//!   v ← G_{σ_diff} ∗ v   (diffusive regularization)
//!
//! ## IC Residual
//!
//!   IC = (1/n) Σ_x ‖φ_fwd(φ_inv(x)) − x‖₂
//!
//! # References
//! - Vercauteren et al. (2009). Diffeomorphic Demons. NeuroImage 45(S1):S61–S72.
//! - Christensen & Johnson (2001). Consistent image registration. IEEE TMI 20(7).

mod ic_residual;
mod registration;

#[cfg(test)]
mod tests;

pub use registration::{
    InverseConsistentDemonsConfig, InverseConsistentDemonsResult,
    InverseConsistentDiffeomorphicDemonsRegistration,
};
