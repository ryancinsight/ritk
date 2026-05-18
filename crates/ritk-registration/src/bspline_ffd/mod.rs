//! B-Spline Free-Form Deformation (FFD) registration.
//!
//! # Mathematical Specification
//!
//! Implements the FFD registration framework of Rueckert et al. (1999).
//! The transformation is parameterized by a uniform cubic B-spline control
//! lattice superimposed on the image domain.
//!
//! ## Transformation Model
//!
//! ```text
//! φ(x) = x + Σᵢ cᵢ · β₃((x − xᵢ) / δ)
//! ```
//!
//! ## Energy Functional
//!
//! ```text
//! E(c) = −D(F, M ∘ φ) + λ · R(φ)
//! ```
//!
//! where `D` is NCC (global), `R` is the bending energy, and `λ` controls
//! regularization strength.
//!
//! ## Multi-Resolution Strategy
//!
//! 1. Start with `initial_control_spacing`.
//! 2. Optimize to convergence at each level.
//! 3. Refine the control grid by halving spacing via B-spline subdivision.
//!
//! # References
//!
//! - Rueckert, D. et al. (1999). Nonrigid registration using free-form
//!   deformations. *IEEE TMI*, 18(8), 712–721.
//! - Lee, S. et al. (1997). Scattered data interpolation with multilevel
//!   B-splines. *IEEE TVCG*, 3(3), 228–244.

mod basis;
mod config;
mod metric;
mod pyramid;
mod registration;
mod regularization;
mod warp;

#[cfg(test)]
mod tests;

pub use config::{BSplineFFDConfig, BSplineFFDResult};
pub use registration::BSplineFFDRegistration;
pub use regularization::bending_energy;
pub use warp::warp_image_bspline;
