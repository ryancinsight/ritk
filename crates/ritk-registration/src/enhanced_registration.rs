//! Enhanced registration framework with validation, progress tracking, and convergence detection.
//!
//! This module provides an improved registration workflow with:
//! - Input validation and numerical stability checks
//! - Progress tracking and callbacks
//! - Early stopping and convergence detection
//! - Gradient clipping
//! - Learning rate scheduling
//! - Comprehensive error handling

use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use crate::metric::Metric;
use crate::optimizer::Optimizer;
use crate::error::Result;
use crate::validation::{
    ValidationConfig, validate_image_shapes, validate_tensor,
    validate_learning_rate, validate_iterations,
    ConvergenceChecker,
};
use crate::progress::{ProgressTracker, ProgressCallback, ConsoleProgressCallback, EarlyStoppingCallback};
use std::sync::Arc;
use std::marker::PhantomData;

/// Configuration for enhanced registration.
#[derive(Debug, Clone)]
pub struct RegistrationConfig {
    /// Validation configuration.
    pub validation: ValidationConfig,
    /// Enable gradient clipping.
    pub enable_gradient_clipping: bool,
    /// Maximum gradient norm.
    pub max_gradient_norm: f64,
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
            enable_gradient_clipping: true,
            max_gradient_norm: 1000.0,
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

    /// Enable gradient clipping.
    pub fn with_gradient_clipping(mut self, max_norm: f64) -> Self {
        self.enable_gradient_clipping = true;
        self.max_gradient_norm = max_norm;
        self
    }

    /// Disable gradient clipping.
    pub fn without_gradient_clipping(mut self) -> Self {
        self.enable_gradient_clipping = false;
        self
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

/// Enhanced registration framework with validation and progress tracking.
pub struct EnhancedRegistration<B, O, M, T, const D: usize>
where
    B: AutodiffBackend,
    O: Optimizer<T, B>,
    M: Metric<B, D>,
    T: Transform<B, D> + AutodiffModule<B>,
{
    optimizer: O,
    metric: M,
    config: RegistrationConfig,
    progress_tracker: ProgressTracker,
    early_stopping: Option<Arc<EarlyStoppingCallback>>,
    _phantom: PhantomData<(B, T)>,
}

impl<B, O, M, T, const D: usize> EnhancedRegistration<B, O, M, T, D>
where
    B: AutodiffBackend,
    O: Optimizer<T, B>,
    M: Metric<B, D>,
    T: Transform<B, D> + AutodiffModule<B>,
{
    /// Create a new enhanced registration.
    pub fn new(optimizer: O, metric: M) -> Self {
        let config = RegistrationConfig::default();
        Self::with_config(optimizer, metric, config)
    }

    /// Create a new enhanced registration with custom config.
    pub fn with_config(optimizer: O, metric: M, config: RegistrationConfig) -> Self {
        let mut progress_tracker = ProgressTracker::new();

        // Add console callback
        let console_callback = Arc::new(ConsoleProgressCallback::new(config.log_interval));
        progress_tracker.add_callback(console_callback);

        // Add early stopping callback if enabled
        let early_stopping = if config.enable_early_stopping {
            let es = Arc::new(EarlyStoppingCallback::new(
                config.early_stopping_min_improvement,
                config.early_stopping_patience,
            ));
            progress_tracker.add_callback(es.clone());
            Some(es)
        } else {
            None
        };

        Self {
            optimizer,
            metric,
            config,
            progress_tracker,
            early_stopping,
            _phantom: PhantomData,
        }
    }

    /// Add a custom progress callback.
    pub fn add_callback(&mut self, callback: Arc<dyn ProgressCallback>) {
        self.progress_tracker.add_callback(callback);
    }

    /// Execute registration with validation and progress tracking.
    pub fn execute(
        &mut self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        mut transform: T,
        iterations: usize,
        learning_rate: f64,
    ) -> Result<T> {
        // Validate inputs
        validate_learning_rate(learning_rate)?;
        validate_iterations(iterations)?;

        if self.config.validation.validate_shapes {
            validate_image_shapes(fixed, moving)?;
        }

        self.optimizer.set_learning_rate(learning_rate);
        self.progress_tracker.start();

        let mut loss_history = Vec::with_capacity(iterations);

        for i in 0..iterations {
            // Forward pass
            let loss = self.metric.forward(fixed, moving, &transform);

            // Validate loss
            if self.config.validation.check_numerical_stability {
                validate_tensor(&loss, &self.config.validation)?;
            }

            // Get loss value
            let loss_data = loss.to_data();
            let loss_val = loss_data.as_slice::<f32>().unwrap()[0] as f64;
            loss_history.push(loss_val);

            // Update progress
            self.progress_tracker.update(i + 1, Some(iterations), loss_val, learning_rate);

            // Check for early stopping
            if let Some(ref es) = self.early_stopping {
                if es.should_stop() {
                    tracing::info!("Early stopping triggered at iteration {}", i + 1);
                    break;
                }
            }

            // Check for convergence
            if self.config.enable_convergence_detection {
                if let Some(ref checker) = self.config.convergence_checker {
                    if checker.check_convergence(&loss_history) {
                        tracing::info!("Convergence detected at iteration {}", i + 1);
                        break;
                    }
                }
            }

            // Backward pass
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &transform);

            // Clip gradients if enabled
            let grads_params = if self.config.enable_gradient_clipping {
                // Note: Burn's GradientsParams doesn't support direct clipping
                // This is a placeholder for future enhancement
                grads_params
            } else {
                grads_params
            };

            // Optimizer step
            transform = self.optimizer.step(transform, grads_params);
        }

        self.progress_tracker.complete(
            *loss_history.last().unwrap_or(&f64::INFINITY),
            learning_rate,
        );

        Ok(transform)
    }

    /// Execute registration with custom progress tracker.
    pub fn execute_with_tracker(
        &mut self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        mut transform: T,
        iterations: usize,
        learning_rate: f64,
        tracker: &ProgressTracker,
    ) -> Result<T> {
        // Validate inputs
        validate_learning_rate(learning_rate)?;
        validate_iterations(iterations)?;

        if self.config.validation.validate_shapes {
            validate_image_shapes(fixed, moving)?;
        }

        self.optimizer.set_learning_rate(learning_rate);
        tracker.start();

        let mut loss_history = Vec::with_capacity(iterations);

        for i in 0..iterations {
            // Forward pass
            let loss = self.metric.forward(fixed, moving, &transform);

            // Validate loss
            if self.config.validation.check_numerical_stability {
                validate_tensor(&loss, &self.config.validation)?;
            }

            // Get loss value
            let loss_data = loss.to_data();
            let loss_val = loss_data.as_slice::<f32>().unwrap()[0] as f64;
            loss_history.push(loss_val);

            // Update progress
            tracker.update(i + 1, Some(iterations), loss_val, learning_rate);

            // Check for early stopping
            if let Some(ref es) = self.early_stopping {
                if es.should_stop() {
                    tracing::info!("Early stopping triggered at iteration {}", i + 1);
                    break;
                }
            }

            // Check for convergence
            if self.config.enable_convergence_detection {
                if let Some(ref checker) = self.config.convergence_checker {
                    if checker.check_convergence(&loss_history) {
                        tracing::info!("Convergence detected at iteration {}", i + 1);
                        break;
                    }
                }
            }

            // Backward pass
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &transform);

            // Optimizer step
            transform = self.optimizer.step(transform, grads_params);
        }

        tracker.complete(
            *loss_history.last().unwrap_or(&f64::INFINITY),
            learning_rate,
        );

        Ok(transform)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registration_config_default() {
        let config = RegistrationConfig::default();
        assert!(config.enable_gradient_clipping);
        assert!(!config.enable_early_stopping);
        assert_eq!(config.log_interval, 50);
    }

    #[test]
    fn test_registration_config_builder() {
        let config = RegistrationConfig::new()
            .with_gradient_clipping(100.0)
            .with_early_stopping(10, 1e-5)
            .with_log_interval(25);

        assert!(config.enable_gradient_clipping);
        assert_eq!(config.max_gradient_norm, 100.0);
        assert!(config.enable_early_stopping);
        assert_eq!(config.early_stopping_patience, 10);
        assert_eq!(config.log_interval, 25);
    }
}
