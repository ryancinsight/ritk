use crate::progress::ConvergenceChecker;
use crate::validation::ValidationConfig;

/// Configuration for registration.
#[derive(Debug, Clone)]
pub struct RegistrationConfig {
    /// Validation configuration.
    pub validation: ValidationConfig,
    /// Enable early stopping.
    pub enable_early_stopping: bool,
    /// Early stopping patience.
    pub early_stopping_patience: usize,
    /// Early stopping minimum improvement.
    pub early_stopping_min_improvement: f64,
    /// Log interval for progress.
    pub log_interval: usize,
    /// Enable convergence detection.
    pub enable_convergence_detection: bool,
    /// Convergence checker.
    pub convergence_checker: Option<ConvergenceChecker>,
}

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            validation: ValidationConfig::default(),
            enable_early_stopping: false,
            early_stopping_patience: 50,
            early_stopping_min_improvement: 1e-6,
            log_interval: 50,
            enable_convergence_detection: false,
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
        self.enable_early_stopping = true;
        self.early_stopping_patience = patience;
        self.early_stopping_min_improvement = min_improvement;
        self
    }

    /// Disable early stopping.
    pub fn without_early_stopping(mut self) -> Self {
        self.enable_early_stopping = false;
        self
    }

    /// Set log interval.
    pub fn with_log_interval(mut self, interval: usize) -> Self {
        self.log_interval = interval;
        self
    }

    /// Enable convergence detection.
    pub fn with_convergence_detection(mut self, checker: ConvergenceChecker) -> Self {
        self.enable_convergence_detection = true;
        self.convergence_checker = Some(checker);
        self
    }

    /// Disable convergence detection.
    pub fn without_convergence_detection(mut self) -> Self {
        self.enable_convergence_detection = false;
        self.convergence_checker = None;
        self
    }
}
