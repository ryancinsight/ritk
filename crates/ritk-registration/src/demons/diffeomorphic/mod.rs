//! Diffeomorphic Demons deformable image registration.
//!
//! # Mathematical Specification
//!
//! The Diffeomorphic Demons algorithm (Vercauteren et al. 2009) maintains a
//! **stationary velocity field** `v : ℤ³ → ℝ³` and produces the displacement
//! field as its exponential map `φ = exp(v)` via the scaling-and-squaring
//! algorithm.  This guarantees that `φ` is a diffeomorphism (invertible with
//! smooth inverse), unlike the classic Thirion formulation.
//!
//! **Per-iteration update:**
//! 1. Compute `φ = exp(v)` via scaling-and-squaring (`n_squarings` steps).
//! 2. Warp moving with `φ` → `M_w`.
//! 3. Compute Thirion forces `u` from `(F, M_w, ∇F)`.
//! 4. BCH velocity update (first-order): `v ← v + u`.
//! 5. Diffusive regularisation: `v ← G_{σ_diff} ∗ v`.
//! 6. Compute MSE = mean((F − M_w)²).
//!
//! # References
//! - Vercauteren, T., Pennec, X., Perchant, A. & Ayache, N. (2009).
//!   Diffeomorphic Demons: Efficient non-parametric image registration.
//!   *NeuroImage* 45(S1):S61–S72.

mod registration;

#[cfg(test)]
mod tests;

pub use registration::DiffeomorphicDemonsRegistration;
