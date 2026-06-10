//! CMA-ES algorithm state types: configuration, result, and convergence reasons.

/// Population evaluation strategy for CMA-ES.
///
/// Replaces the former `parallel_population: bool` field, eliminating boolean
/// blindness at call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PopulationEval {
    /// Evaluate candidates sequentially in the current thread.
    #[default]
    Sequential,
    /// Evaluate candidates in parallel across rayon threads.
    /// The objective function `f` must be `Sync`.
    Parallel,
}

/// Per-generation history recording policy for CMA-ES.
///
/// Replaces the former `record_history: bool` field, eliminating boolean
/// blindness at call sites.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HistoryPolicy {
    /// Do not record per-generation best-f values.
    #[default]
    Discard,
    /// Record the best function value at each generation in `CmaEsResult::best_history`.
    Record,
}

/// Reason the CMA-ES run terminated.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// Step-size σ fell below `CmaEsConfig::sigma_tol`.
    StepSizeTooSmall,
    /// Generation count reached `CmaEsConfig::max_generations`.
    MaxGenerations,
    /// Cholesky condition number estimate exceeded 10¹⁴ (numerical ill-conditioning).
    ConditionTooLarge,
    /// Best function value fell below `CmaEsConfig::ftol`.
    FunctionTolerance,
}

/// Configuration for a single CMA-ES run.
#[derive(Debug, Clone)]
pub struct CmaEsConfig {
    /// Initial global step-size σ₀. Calibrate to the expected search distance
    /// (e.g., 0.3 for normalised parameters).
    pub sigma0: f64,
    /// Population size λ (offspring per generation).
    /// 0 = use the default formula λ = 4 + ⌊3 ln n⌋.
    pub lambda: usize,
    /// Maximum number of generations before the run is declared converged by
    /// iteration limit.
    pub max_generations: usize,
    /// Stop when the step-size σ falls below this threshold (convergence by
    /// step-size shrinkage).
    pub sigma_tol: f64,
    /// Stop when the best function value falls below this threshold (solution
    /// quality gate).
    pub ftol: f64,
    /// LCG seed for the Box-Muller random normal generator. Different seeds
    /// give independent runs.
    pub seed: u64,
    /// Whether to evaluate the population in parallel using rayon.
    /// When [`PopulationEval::Parallel`], the λ candidates per generation are
    /// evaluated concurrently across CPU cores. The objective function `f`
    /// must be `Sync`.
    /// Default: [`PopulationEval::Sequential`] (backward-compatible).
    pub parallel_population: PopulationEval,
    /// Per-generation best-f recording policy.
    pub record_history: HistoryPolicy,
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
            parallel_population: PopulationEval::default(),
            record_history: HistoryPolicy::default(),
        }
    }
}

/// Result of a single CMA-ES run.
#[derive(Debug, Clone)]
pub struct CmaEsResult {
    /// Parameter vector achieving the lowest observed function value.
    pub best_x: Vec<f64>,
    /// Lowest observed function value f(best_x).
    pub best_f: f64,
    /// Number of generations completed before termination.
    pub generations: usize,
    /// Condition that triggered termination.
    pub stop_reason: StopReason,
    /// LCG seed actually used (equals `CmaEsConfig::seed`).
    pub seed_used: u64,
    /// Step-size σ at termination.
    pub final_sigma: f64,
    /// Cholesky-diagonal condition estimate (max dᵢ / min dᵢ)² at termination.
    pub condition_estimate: f64,
    /// Per-generation best function values, populated when
    /// `CmaEsConfig::record_history` is true.
    pub best_history: Option<Vec<f64>>,
}
