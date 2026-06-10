use crate::progress::ConvergenceChecker;
use crate::validation::ValidationConfig;

/// Whether early stopping is enabled during iterative optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EarlyStoppingPolicy {
    /// Early stopping is disabled.
    #[default]
    Disabled,
    /// Early stopping is enabled.
    Enabled,
}

/// Configuration for registration.
#[derive(Debug, Clone)]
pub struct RegistrationConfig {
    /// Validation configuration.
    pub validation: ValidationConfig,
    /// Early stopping policy.
    pub early_stopping: EarlyStoppingPolicy,
    /// Early stopping patience.
    pub early_stopping_patience: usize,
    /// Early stopping minimum improvement.
    pub early_stopping_min_improvement: f64,
    /// Log interval for progress.
    pub log_interval: usize,
    /// Convergence checker. When `Some`, convergence detection is enabled.
    pub convergence_checker: Option<ConvergenceChecker>,
}

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            validation: ValidationConfig::default(),
            early_stopping: EarlyStoppingPolicy::Disabled,
            early_stopping_patience: 50,
            early_stopping_min_improvement: 1e-6,
            log_interval: 50,
            convergence_checker: None,
        }
    }
}

impl RegistrationConfig {
    /// Create a new registration config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable early stopping.
    pub fn with_early_stopping(mut self, patience: usize, min_improvement: f64) -> Self {
        self.early_stopping = EarlyStoppingPolicy::Enabled;
        self.early_stopping_patience = patience;
        self.early_stopping_min_improvement = min_improvement;
        self
    }

    /// Disable early stopping.
    pub fn without_early_stopping(mut self) -> Self {
        self.early_stopping = EarlyStoppingPolicy::Disabled;
        self
    }

    /// Set log interval.
    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval;
        self
    }

    /// Enable convergence detection with the given checker.
    pub fn with_convergence_detection(mut self, checker: ConvergenceChecker) -> Self {
        self.convergence_checker = Some(checker);
        self
    }

    /// Disable convergence detection.
    pub fn without_convergence_detection(mut self) -> Self {
        self.convergence_checker = None;
        self
    }
}
