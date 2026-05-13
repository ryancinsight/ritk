//! Convergence reason enum for the RSGD optimizer.

/// Reason the optimizer stopped iterating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    /// Gradient L2-norm fell below `gradient_tolerance`.
    GradientConvergence,
    /// Step length fell below `minimum_step_length` after relaxation.
    StepConvergence,
    /// Number of accepted steps reached `maximum_iterations`.
    MaximumIterations,
}
