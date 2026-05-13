//! Optimizer trait for parameter optimization.
//!
//! This module defines the core Optimizer trait that all optimization algorithms
//! must implement for training transforms in image registration.

use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;

/// Lightweight optimizer telemetry for registration workflows.
#[derive(Debug, Clone, PartialEq)]
pub struct OptimizerTelemetry {
    /// Human-readable optimizer name.
    pub algorithm: &'static str,
    /// Number of parameter-update steps taken.
    pub steps: usize,
    /// Current learning rate, if applicable.
    pub learning_rate: Option<f64>,
}

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

    /// Current optimizer telemetry.
    fn telemetry(&self) -> OptimizerTelemetry;
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── StepDecay ─────────────────────────────────────────────────────────────

    /// At step=0 the exponent is 0, so get_lr returns initial_lr * gamma^0 = initial_lr.
    #[test]
    fn step_decay_initial_step_returns_initial_lr() {
        let sched = StepDecay::new(10, 0.1);
        let lr = sched.get_lr(0, 1.0);
        assert!(
            (lr - 1.0).abs() < 1e-12,
            "step=0 must return initial_lr unchanged; got {lr}"
        );
    }

    /// After exactly `step_size` steps, one decay is applied: lr = initial * gamma.
    ///
    /// # Derivation
    /// step=10, step_size=10 → exponent=1 → lr = 0.5 * 0.5^1 = 0.25.
    #[test]
    fn step_decay_applies_one_decay_at_first_boundary() {
        let sched = StepDecay::new(10, 0.5);
        let lr = sched.get_lr(10, 0.5);
        let expected = 0.5 * 0.5_f64.powi(1);
        assert!(
            (lr - expected).abs() < 1e-12,
            "step=10 with step_size=10, initial=0.5, gamma=0.5: expected {expected}, got {lr}"
        );
    }

    /// Learning rate is monotone non-increasing with step count.
    #[test]
    fn step_decay_monotone_non_increasing() {
        let sched = StepDecay::new(5, 0.5);
        let initial_lr = 1.0;
        let lrs: Vec<f64> = (0..=25).map(|s| sched.get_lr(s, initial_lr)).collect();
        for window in lrs.windows(2) {
            assert!(
                window[0] >= window[1],
                "LR must be non-increasing: {:.6} >= {:.6}",
                window[0],
                window[1]
            );
        }
    }

    /// With gamma=1.0, the learning rate never decays.
    #[test]
    fn step_decay_gamma_one_constant_lr() {
        let sched = StepDecay::new(1, 1.0);
        let initial_lr = 0.01;
        for step in 0..100 {
            let lr = sched.get_lr(step, initial_lr);
            assert!(
                (lr - initial_lr).abs() < 1e-12,
                "gamma=1.0 must produce constant LR; step={step}, got {lr}"
            );
        }
    }

    /// Two boundary steps apart: exponent increments by 1 each time.
    ///
    /// # Derivation
    /// step_size=3, gamma=0.5:
    ///   step=0  → exp=0 → lr=initial
    ///   step=3  → exp=1 → lr=initial*0.5
    ///   step=6  → exp=2 → lr=initial*0.25
    #[test]
    fn step_decay_multiple_boundaries_correct() {
        let sched = StepDecay::new(3, 0.5);
        let lr0 = sched.get_lr(0, 1.0);
        let lr3 = sched.get_lr(3, 1.0);
        let lr6 = sched.get_lr(6, 1.0);
        assert!((lr0 - 1.0).abs() < 1e-12, "step=0 → 1.0; got {lr0}");
        assert!((lr3 - 0.5).abs() < 1e-12, "step=3 → 0.5; got {lr3}");
        assert!((lr6 - 0.25).abs() < 1e-12, "step=6 → 0.25; got {lr6}");
    }

    // ── OptimizerTelemetry ───────────────────────────────────────────────────

    #[test]
    fn optimizer_telemetry_debug_and_eq() {
        let t1 = OptimizerTelemetry {
            algorithm: "GradientDescent",
            steps: 42,
            learning_rate: Some(1e-3),
        };
        let t2 = t1.clone();
        assert_eq!(t1, t2);
        assert!(format!("{t1:?}").contains("GradientDescent"));
    }
}
