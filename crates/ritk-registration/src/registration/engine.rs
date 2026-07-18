use crate::error::Result;
use crate::metric::Metric;
use crate::optimizer::trait_::central_difference;
use crate::optimizer::Optimizer;
use crate::progress::ProgressTracker;
use crate::validation::{
    validate_image_shapes, validate_iterations, validate_learning_rate, validate_tensor,
    NumericalCheck, ShapeValidation };
use coeus_core::{CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_nn::Module;
use coeus_ops::BackendOps;
use ritk_image::Image;
use ritk_transform::Transform;

use super::config::{RegistrationConfig, TrackerBuildResult};
use super::summary::StopReason;
use super::Registration;

/// Captures the loop outcome shared by both execution paths.
struct LoopOutcome<T> {
    transform: T,
    loss_history: Vec<f64>,
    iterations_completed: usize,
    stop_reason: StopReason }

impl<B, O, M, T, const D: usize> Registration<B, O, M, T, D>
where
    B: ritk_image::tensor::Backend + BackendOps<f32> + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
    O: Optimizer<T, B>,
    M: Metric<B, D>,
    T: Transform<B, D> + Module<f32, B> + Clone,
{
    /// Create a new registration.
    pub fn new(optimizer: O, metric: M) -> Self {
        let config = RegistrationConfig::default();
        Self::with_config(optimizer, metric, config)
    }

    /// Create a new registration with custom config.
    pub fn with_config(optimizer: O, metric: M, config: RegistrationConfig) -> Self {
        let TrackerBuildResult {
            tracker: progress_tracker,
            early_stopping } = config.build_tracker();

        Self {
            optimizer,
            metric,
            config,
            progress_tracker,
            early_stopping,
            _phantom: std::marker::PhantomData }
    }

    /// Add a custom progress callback.
    pub fn add_callback(
        &mut self,
        callback: std::sync::Arc<dyn crate::progress::ProgressCallback>,
    ) {
        self.progress_tracker.add_callback(callback);
    }

    /// Execute registration with validation and progress tracking.
    pub fn execute(
        &mut self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
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
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
        transform: T,
        iterations: usize,
        learning_rate: f64,
    ) -> Result<super::summary::RegistrationSummary<T>> {
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

        Ok(super::summary::RegistrationSummary {
            transform: outcome.transform,
            loss_history: outcome.loss_history,
            optimizer_telemetry,
            iterations_completed: outcome.iterations_completed,
            final_loss,
            stop_reason: outcome.stop_reason })
    }

    /// Execute registration with custom progress tracker.
    pub fn execute_with_tracker(
        &mut self,
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
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
        fixed: &Image<f32, B, D>,
        moving: &Image<f32, B, D>,
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

            // Get loss value.
            // `clone()` is a cheap Arc-increment; `into_scalar()` consumes the clone
            // so the original `loss` remains live for `backward()` below.
            // `ElementConversion::elem()` converts B::FloatElem â†’ f64 for any backend type,
            // avoiding the `as_slice::<f32>()` hardcode that panics on non-f32 backends.
            let loss_val = f64::from(loss.as_slice()[0]);
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

            let gradients = central_difference(&transform, |candidate| {
                f64::from(self.metric.forward(fixed, moving, candidate).as_slice()[0])
            });
            transform = self.optimizer.step(transform, gradients);
        }

        Ok(LoopOutcome {
            transform,
            loss_history,
            iterations_completed,
            stop_reason })
    }
}
