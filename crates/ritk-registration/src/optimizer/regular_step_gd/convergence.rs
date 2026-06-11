//! Shared convergence types for gradient-descent optimizers.

/// Internal convergence state for gradient-descent optimizers.
///
/// Tracks whether an optimizer should continue iterating or has met a
/// convergence criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConvergenceFlag {
    /// Optimizer is still iterating.
    #[default]
    Iterating,
    /// Convergence criterion was satisfied.
    Converged,
}

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
