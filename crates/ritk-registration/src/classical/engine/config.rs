//! Configuration for the classical registration engine.

/// Default convergence tolerance for classical optimizer similarity-metric improvement.
pub const DEFAULT_OPTIMIZER_TOLERANCE: f64 = 1e-6;

/// Configuration for classical registration algorithms.
#[derive(Debug, Clone)]
pub struct ClassicalConfig {
    /// Maximum number of iterations for optimization.
    pub max_iterations: usize,
    /// Convergence tolerance for similarity metric improvement.
    pub tolerance: f64,
    /// Step size multiplier for perturbations.
    pub step_multiplier: f64 }

impl Default for ClassicalConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: DEFAULT_OPTIMIZER_TOLERANCE,
            step_multiplier: 1.0 }
    }
}
