//! Registration framework with validation, progress tracking, and convergence detection.

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
use crate::progress::{
    ConsoleProgressCallback, EarlyStoppingCallback, ProgressCallback, ProgressTracker,
};
use crate::validation::{
    validate_image_shapes, validate_iterations, validate_learning_rate, validate_tensor,
    NumericalCheck, ShapeValidation,
};
use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use ritk_core::image::Image;
use ritk_core::transform::Transform;
use std::marker::PhantomData;
use std::sync::Arc;

pub mod config;
pub mod dl_registration_loss;
pub mod dl_ssm_registration;
pub mod summary;

pub use config::{EarlyStoppingPolicy, RegistrationConfig};
pub use dl_registration_loss::{
    RegistrationLoss, RegistrationLossConfig, RegularizationType, SimilarityMetric,
};
pub use summary::{RegistrationSummary, StopReason};

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

/// Captures the loop outcome shared by both execution paths.
struct LoopOutcome<T> {
    transform: T,
    loss_history: Vec<f64>,
    iterations_completed: usize,
    stop_reason: StopReason,
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
        let early_stopping = if config.early_stopping == EarlyStoppingPolicy::Enabled {
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
        transform: T,
        iterations: usize,
        learning_rate: f64,
    ) -> Result<RegistrationSummary<T>> {
        // Validate inputs
        validate_learning_rate(learning_rate)?;
        validate_iterations(iterations)?;

        if self.config.validation.shape_validation == ShapeValidation::Enabled {
            validate_image_shapes(fixed, moving)?
        }

        self.optimizer.set_learning_rate(learning_rate);
        self.progress_tracker.start();

        let tracker = self.progress_tracker.clone();
        let outcome = self.run_loop(
            fixed,
            moving,
            transform,
            iterations,
            learning_rate,
            &tracker,
        )?;

        self.progress_tracker.complete(
            *outcome.loss_history.last().unwrap_or(&f64::INFINITY),
            learning_rate,
        );

        let final_loss = *outcome.loss_history.last().unwrap_or(&f64::INFINITY);
        let optimizer_telemetry = self.optimizer.telemetry();

        Ok(RegistrationSummary {
            transform: outcome.transform,
            loss_history: outcome.loss_history,
            optimizer_telemetry,
            iterations_completed: outcome.iterations_completed,
            final_loss,
            stop_reason: outcome.stop_reason,
        })
    }

    /// Execute registration with custom progress tracker.
    pub fn execute_with_tracker(
        &mut self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        transform: T,
        iterations: usize,
        learning_rate: f64,
        tracker: &ProgressTracker,
    ) -> Result<T> {
        // Validate inputs
        validate_learning_rate(learning_rate)?;
        validate_iterations(iterations)?;

        if self.config.validation.shape_validation == ShapeValidation::Enabled {
            validate_image_shapes(fixed, moving)?
        }

        self.optimizer.set_learning_rate(learning_rate);
        tracker.start();

        let outcome =
            self.run_loop(fixed, moving, transform, iterations, learning_rate, tracker)?;

        tracker.complete(
            *outcome.loss_history.last().unwrap_or(&f64::INFINITY),
            learning_rate,
        );

        Ok(outcome.transform)
    }

    /// Shared iteration loop used by both `execute_with_summary` and `execute_with_tracker`.
    ///
    /// Runs the forward/backward/optimizer cycle for up to `iterations` steps,
    /// tracking early stopping, convergence, and loss history. Returns the
    /// final transform, loss history, completed iteration count, and whether
    /// the loop exited before exhausting all iterations.
    fn run_loop(
        &mut self,
        fixed: &Image<B, D>,
        moving: &Image<B, D>,
        mut transform: T,
        iterations: usize,
        learning_rate: f64,
        tracker: &ProgressTracker,
    ) -> Result<LoopOutcome<T>> {
        let mut loss_history = Vec::with_capacity(iterations);
        let mut iterations_completed = 0usize;
        let mut stop_reason = StopReason::Completed;

        for i in 0..iterations {
            // Forward pass
            let loss = self.metric.forward(fixed, moving, &transform);

            // Validate loss
            if self.config.validation.numerical_check == NumericalCheck::Enabled {
                validate_tensor(&loss, &self.config.validation)?;
            }

            // Get loss value
            let loss_data = loss.to_data();
            let loss_val = loss_data
                .as_slice::<f32>()
                .expect("loss value tensor data must be contiguous f32")[0]
                as f64;
            loss_history.push(loss_val);

            // Update progress
            tracker.update(i + 1, Some(iterations), loss_val, learning_rate);
            iterations_completed = i + 1;

            // Check for early stopping
            if let Some(ref es) = self.early_stopping {
                if es.should_stop() {
                    tracing::info!("Early stopping triggered at iteration {}", i + 1);
                    stop_reason = StopReason::EarlyStopping;
                    break;
                }
            }

            // Check for convergence
            if let Some(ref checker) = self.config.convergence_checker {
                if checker.check_convergence(&loss_history) {
                    tracing::info!("Convergence detected at iteration {}", i + 1);
                    stop_reason = StopReason::EarlyStopping;
                    break;
                }
            }

            // Backward pass
            let grads = loss.backward();

            let grads_params = GradientsParams::from_grads(grads, &transform);

            // Optimizer step
            transform = self.optimizer.step(transform, grads_params);
        }

        Ok(LoopOutcome {
            transform,
            loss_history,
            iterations_completed,
            stop_reason,
        })
    }
}

#[cfg(test)]
mod tests;
