//! Exact mathematical state definitions bounding CMA-ES configurations and convergence stops natively.

/// Exact convergence Stop Reason definitions tracking mathematical iteration faults strictly.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// Step-size fell strictly below the exact scalar tolerance threshold natively.
    StepSizeTooSmall,
    /// Exact generic hardbound iteration limit exceeded natively.
    MaxGenerations,
    /// Absolute numerical instability condition bounding convergence limits.
    ConditionTooLarge,
    /// Best target function scalar natively superseded explicitly.
    FunctionTolerance,
}

/// Rigid algorithmic bounds describing explicitly typed setup configurations for the solver.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Exact geometric boundary scale matching initial covariance generation step limits securely.
    pub sigma0: f64,
    /// Exact population size per loop explicitly mapped securely. (0 sets native log math).
    pub lambda: usize,
    /// Upper analytical cutoff iterations.
    pub max_generations: usize,
    /// Bottom step constraint.
    pub sigma_tol: f64,
    /// Minimal function tolerance difference strictly.
    pub ftol: f64,
    /// Deterministic explicitly injected seed.
    pub seed: u64,
    /// Deterministic vector flag.
    pub record_history: bool,
}

impl Default for CmaEsConfig {
    fn default() -> Self {
        Self {
            sigma0: 0.3,
            lambda: 0,
            max_generations: 10_000,
            sigma_tol: 1e-12,
            ftol: 1e-15,
            seed: 0xcafe_babe_dead_beef,
            record_history: false,
        }
    }
}

/// Bounded struct validating strictly defined optimization returns exactly natively via fields.
#[derive(Debug, Clone)]
pub struct CmaEsResult {
    /// Explicit target output geometric points tracking exactly mapping minima limits securely.
    pub best_x: Vec<f64>,
    /// Precise objective function scalar mapping best vector minima natively.
    pub best_f: f64,
    /// Overall generation loop count rigorously natively bound securely.
    pub generations: usize,
    /// Terminating structural fault or bounds matched securely natively.
    pub stop_reason: StopReason,
    /// Validated injection PRNG sequence hash securely matching reproducibility explicitly.
    pub seed_used: u64,
    /// Exiting standard deviation step constraint tracking the final boundary length natively.
    pub final_sigma: f64,
    /// The ending conditioning limits bound native Cholesky conditioning faults exactly.
    pub condition_estimate: f64,
    /// Iteration logging exactly native to generation loops selectively securely.
    pub best_history: Option<Vec<f64>>,
}
