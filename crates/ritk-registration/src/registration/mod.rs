//! Registration framework with validation, progress tracking, and convergence detection.
//!
//! This module provides a registration workflow with:
//! - Input validation and numerical stability checks
//! - Progress tracking and callbacks
//! - Early stopping and convergence detection
//! - Gradient clipping
//! - Learning rate scheduling
//! - Comprehensive error handling

use crate::error::Result;
use crate::metric::Metric;
use crate::optimizer::Optimizer;
use crate::progress::{ConsoleProgressCallback, EarlyStoppingCallback, ProgressCallback, ProgressTracker};
use crate::validation::{validate_image_shapes, validate_iterations, validate_learning_rate, validate_tensor};
use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use std::marker::PhantomData;
use std::sync::Arc;

pub mod config;
pub mod summary;
pub mod dl_ssm_registration;
pub mod dl_registration_loss;

pub use config::RegistrationConfig;
pub use summary::RegistrationSummary;
pub use dl_registration_loss::{RegistrationLoss, RegistrationLossConfig, SimilarityMetric, RegularizationType};

/// Registration framework with validation and progress tracking.
pub struct Registration<B, O, M, T, const D: usize>
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

impl<B, O, M, T, const D: usize> Registration<B, O, M, T, D>
where
    B: AutodiffBackend,
    O: Optimizer<T, B>,
    M: Metric<B, D>,
    T: Transform<B, D> + AutodiffModule<B>,
{
    /// Create a new registration.
    pub fn new(optimizer: O, metric: M) -> Self {
        let config = RegistrationConfig::default();
        Self::with_config(optimizer, metric, config)
    }

    /// Create a new registration with custom config.
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
        transform: T,
        iterations: usize,
        learning_rate: f64,
    ) -> Result<T> {
        Ok(self
            .execute_with_summary(fixed, moving, transform, iterations, learning_rate)?
            .transform)
    }

    /// Execute registration and return execution diagnostics.
    pub fn execute_with_summary(
        &mut self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        mut transform: T,
        iterations: usize,
        learning_rate: f64,
    ) -> Result<RegistrationSummary<T>> {
        // Validate inputs
        validate_learning_rate(learning_rate)?;
        validate_iterations(iterations)?;

        if self.config.validation.validate_shapes {
            validate_image_shapes(fixed, moving)?;
        }

        self.optimizer.set_learning_rate(learning_rate);
        self.progress_tracker.start();

        let mut loss_history = Vec::with_capacity(iterations);
        let mut iterations_completed = 0usize;
        let mut stopped_early = false;

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
            self.progress_tracker
                .update(i + 1, Some(iterations), loss_val, learning_rate);
            iterations_completed = i + 1;

            // Check for early stopping
            if let Some(ref es) = self.early_stopping {
                if es.should_stop() {
                    tracing::info!("Early stopping triggered at iteration {}", i + 1);
                    stopped_early = true;
                    break;
                }
            }

            // Check for convergence
            if self.config.enable_convergence_detection {
                if let Some(ref checker) = self.config.convergence_checker {
                    if checker.check_convergence(&loss_history) {
                        tracing::info!("Convergence detected at iteration {}", i + 1);
                        stopped_early = true;
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

        self.progress_tracker.complete(
            *loss_history.last().unwrap_or(&f64::INFINITY),
            learning_rate,
        );

        let final_loss = *loss_history.last().unwrap_or(&f64::INFINITY);
        let optimizer_telemetry = self.optimizer.telemetry();

        Ok(RegistrationSummary {
            transform,
            loss_history,
            optimizer_telemetry,
            iterations_completed,
            final_loss,
            stopped_early,
        })
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
mod tests;
