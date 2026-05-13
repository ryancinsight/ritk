//! Regular Step Gradient Descent (RSGD) optimizer.
//!
//! Implements ITK's `RegularStepGradientDescentOptimizerv4` algorithm for
//! image registration parameter optimization.
//!
//! # Algorithm
//!
//! ```text
//! θ₀ = initial parameters
//! Δ₀ = initial_step_length
//! δ_rel = relaxation_factor
//! Δ_min = minimum_step_length
//! Δ_max = maximum_step_length
//!
//! For iteration k = 0, 1, ..., max_iters-1:
//!   1. Compute ‖∇L(θₖ)‖₂; if < gradient_tolerance → STOP
//!   2. Clamp Δₖ to Δ_max
//!   3. Compute effective_lr = Δₖ / ‖g‖; apply θₖ₊₁ = θₖ − effective_lr · g
//!   4. If L(θₖ₊₁) > L(θₖ): revert θ, Δₖ₊₁ = Δₖ · δ_rel;
//!      if Δₖ₊₁ < Δ_min → STOP
//!   5. Else accept step; if steps ≥ max_iters → STOP
//! ```
//!
//! # References
//!
//! - ITK `RegularStepGradientDescentOptimizerv4`:
//!   <https://itk.org/Doxygen/html/classitk_1_1RegularStepGradientDescentOptimizerv4.html>

mod config;
mod convergence;
mod grad_norm;
mod optimizer;
mod step_mapper;

#[cfg(test)]
mod tests;

pub use config::RegularStepGdConfig;
pub use convergence::ConvergenceReason;
pub use optimizer::RegularStepGradientDescent;
