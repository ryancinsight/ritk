//! Selective State Space (S6) Module
//!
//! Implements the core State Space Model from Mamba (Gu & Dao, 2023) adapted for
//! 3D volumetric medical image registration. The key innovation is input-dependent
//! parameters (Δ, B, C) that enable content-based selective propagation of information.
//!
//! # Architecture
//!
//! The continuous state space model is:
//!   h'(t) = Ah(t) + Bx(t)
//!   y(t)  = Ch(t)
//!
//! Discretized with input-dependent step size Δ:
//!   h_k = Āh_{k-1} + B̄x_k
//!   y_k = Ch_k
//!
//! where Ā = exp(ΔA) and B̄ = (ΔA)^{-1}(exp(ΔA) - I)·ΔB

pub mod config;
pub mod model;
pub mod scan;

pub use config::{SelectiveStateSpaceConfig, StateSpaceParameters};
pub use model::SelectiveStateSpace;

#[cfg(test)]
mod tests;
