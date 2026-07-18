use crate::progress::ConvergenceChecker;
use crate::validation::ValidationConfig;

/// Output of [`RegistrationConfig::build_tracker`].
pub(crate) struct TrackerBuildResult {
    pub tracker: crate::progress::ProgressTracker,
    pub early_stopping: Option<std::sync::Arc<crate::progress::EarlyStoppingCallback>> }

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
        min_improvement: f64 } }

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
    pub convergence_checker: Option<ConvergenceChecker> }

impl Default for RegistrationConfig {
    fn default() -> Self {
        Self {
            validation: ValidationConfig::default(),
            early_stopping: EarlyStoppingPolicy::Disabled,
            log_interval: 50,
            convergence_checker: None }
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
            min_improvement };
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

    /// Construct a [`ProgressTracker`](crate::progress::ProgressTracker) pre-populated with
    /// callbacks derived from this config.
    ///
    /// Always adds [`ConsoleProgressCallback`](crate::progress::ConsoleProgressCallback).
    /// Adds [`EarlyStoppingCallback`](crate::progress::EarlyStoppingCallback) when
    /// `self.early_stopping == EarlyStoppingPolicy::Enabled`.
    pub(crate) fn build_tracker(&self) -> TrackerBuildResult {
        let mut tracker = crate::progress::ProgressTracker::new();
        let console = std::sync::Arc::new(crate::progress::ConsoleProgressCallback::new(
            self.log_interval,
        ));
        tracker.add_callback(console);
        let early_stopping = if let EarlyStoppingPolicy::Enabled {
            patience,
            min_improvement } = self.early_stopping
        {
            let es = std::sync::Arc::new(crate::progress::EarlyStoppingCallback::new(
                min_improvement,
                patience,
            ));
            tracker.add_callback(es.clone());
            Some(es)
        } else {
            None
        };
        TrackerBuildResult {
            tracker,
            early_stopping }
    }
}
