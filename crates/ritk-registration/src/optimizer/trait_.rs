//! Optimizer trait for parameter optimization.
//!
//! This module defines the core Optimizer trait that all optimization algorithms
//! must implement for training transforms in image registration.

use burn::tensor::backend::AutodiffBackend;
use burn::module::AutodiffModule;
use burn::optim::GradientsParams;

/// Optimizer trait for training transforms.
///
/// Optimizers update transform parameters based on computed gradients
/// to minimize the registration metric (loss function).
///
/// # Type Parameters
/// * `M` - The module/transform type to optimize
/// * `B` - The backend for tensor operations (must support autodiff)
///
/// # Examples
///
/// ```rust,ignore
/// use ritk_registration::optimizer::{Optimizer, GradientDescent};
///
/// let optimizer = GradientDescent::new(0.01);
/// ```
pub trait Optimizer<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    /// Perform a single optimization step.
    ///
    /// Updates the module parameters based on the computed gradients.
    ///
    /// # Arguments
    /// * `module` - The module/transform to update
    /// * `gradients` - The gradients of the loss with respect to module parameters
    ///
    /// # Returns
    /// The updated module with new parameter values
    fn step(&mut self, module: M, gradients: GradientsParams) -> M;

    /// Get the current learning rate.
    ///
    /// # Returns
    /// The learning rate used for parameter updates
    fn learning_rate(&self) -> f64;

    /// Set the learning rate.
    ///
    /// # Arguments
    /// * `lr` - The new learning rate
    fn set_learning_rate(&mut self, lr: f64);
}

/// Learning rate scheduler trait.
///
/// Schedulers adjust the learning rate during training to improve convergence.
pub trait LearningRateScheduler: Send + Sync {
    /// Get the learning rate for the current step.
    ///
    /// # Arguments
    /// * `step` - The current optimization step
    /// * `initial_lr` - The initial learning rate
    ///
    /// # Returns
    /// The learning rate to use for this step
    fn get_lr(&self, step: usize, initial_lr: f64) -> f64;
}

/// Step decay learning rate scheduler.
///
/// Reduces the learning rate by a factor every `step_size` steps.
#[derive(Debug, Clone)]
pub struct StepDecay {
    step_size: usize,
    gamma: f64,
}

impl StepDecay {
    /// Create a new step decay scheduler.
    ///
    /// # Arguments
    /// * `step_size` - Number of steps between LR reductions
    /// * `gamma` - Multiplicative factor (typically 0.1 to 0.5)
    pub fn new(step_size: usize, gamma: f64) -> Self {
        assert!(gamma > 0.0 && gamma <= 1.0, "Gamma must be in (0, 1]");
        assert!(step_size > 0, "Step size must be positive");
        Self { step_size, gamma }
    }
}

impl LearningRateScheduler for StepDecay {
    fn get_lr(&self, step: usize, initial_lr: f64) -> f64 {
        let exponent = step / self.step_size;
        initial_lr * self.gamma.powi(exponent as i32)
    }
}
