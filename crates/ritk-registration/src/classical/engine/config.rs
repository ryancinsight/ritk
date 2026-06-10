//! Configuration for the classical registration engine.

/// Configuration for classical registration algorithms.
#[derive(Debug, Clone)]
pub struct ClassicalConfig {
    /// Maximum number of iterations for optimization.
    pub max_iterations: usize,
    /// Convergence tolerance for similarity metric improvement.
    pub tolerance: f64,
    /// Step size multiplier for perturbations.
    pub step_multiplier: f64,
}

impl Default for ClassicalConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            step_multiplier: 1.0,
        }
    }
}
