use crate::progress::ConvergenceChecker;
use crate::validation::ValidationConfig;

/// Whether early stopping is enabled during iterative optimization.
///
/// `Enabled` carries its parameters so invalid state
/// (`Disabled` + non-zero patience) is unrepresentable.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum EarlyStoppingPolicy {
    /// Early stopping is disabled.
    #[default]
    Disabled,
    /// Early stopping is enabled with the given patience and minimum improvement threshold.
    Enabled {
        /// Number of iterations without sufficient improvement before stopping.
        patience: usize,
        /// Minimum loss improvement required to reset the patience counter.
        min_improvement: f64,
    },
}

/// Configuration for registration.
#[derive(Debug, Clone)]
pub struct RegistrationConfig {
    /// Validation configuration.
    pub validation: ValidationConfig,
    /// Early stopping policy; parameters are co-located with the variant.
    pub early_stopping: EarlyStoppingPolicy,
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

    /// Enable early stopping with the given `patience` and `min_improvement` threshold.
    pub fn with_early_stopping(mut self, patience: usize, min_improvement: f64) -> Self {
        self.early_stopping = EarlyStoppingPolicy::Enabled {
            patience,
            min_improvement,
        };
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
