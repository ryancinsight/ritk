//! Result type for global MI registration.

use crate::optimizer::ConvergenceReason;

/// Whether the multi-resolution MI registration converged at every level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// All levels converged before reaching the maximum iteration count.
    Converged,
    /// At least one level exhausted the maximum iteration count without converging.
    MaxIterationsReached,
}

/// Result of global MI registration.
#[derive(Debug, Clone)]
pub struct GlobalMiResult<const D: usize> {
    /// 4×4 homogeneous matrix representation of the final transform.
    pub matrix: [f64; 16],
    /// Final Mattes MI value (positive; negated from the loss).
    pub final_mi: f64,
    /// Per-level convergence reasons.
    pub convergence_history: Vec<ConvergenceReason>,
    /// Number of iterations per level.
    pub iterations_per_level: Vec<usize>,
    /// Loss history at the final level.
    pub loss_history: Vec<f64>,
    /// Whether overall registration converged at every level.
    pub convergence: ConvergenceStatus,
}
