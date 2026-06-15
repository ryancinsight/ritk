//! Regular Step Gradient Descent optimizer struct and Optimizer impl.

use super::config::RegularStepGdConfig;
use super::convergence::{ConvergenceFlag, ConvergenceReason};
use super::grad_norm::GradientNormVisitor;
use super::step_mapper::RsgdStepMapper;
use crate::optimizer::{Optimizer, OptimizerAlgorithm, OptimizerTelemetry};
use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use std::marker::PhantomData;

/// Regular Step Gradient Descent optimizer (ITK `RegularStepGradientDescentOptimizerv4`).
///
/// Steps in the negative normalized-gradient direction with a step length
/// that shrinks by `relaxation_factor` whenever the loss does not improve.
/// Converges when either the gradient norm or the step length drops below
/// their respective tolerances, or the step count reaches the maximum.
///
/// # Type Parameters
/// - `M`: Module/transform type (must implement `AutodiffModule<B>`)
/// - `B`: Autodiff backend
pub struct RegularStepGradientDescent<M: AutodiffModule<B>, B: AutodiffBackend> {
    config: RegularStepGdConfig,
    /// Current step length Δₖ.
    current_step_length: f64,
    /// Loss value set via `set_loss()` before the current `step()` call.
    current_loss: Option<f64>,
    /// Loss from the previous accepted step, L(θₖ₋₁).
    prev_loss: Option<f64>,
    /// Number of accepted (non-reverted) optimization steps.
    steps: usize,
    /// Whether the optimizer has converged.
    convergence: ConvergenceFlag,
    /// Why the optimizer converged, if it has.
    convergence_reason: Option<ConvergenceReason>,
    _phantom: PhantomData<(M, B)>,
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> RegularStepGradientDescent<M, B> {
    /// Create a new RSGD optimizer from configuration.
    ///
    /// # Panics
    /// Panics if configuration validation fails.
    pub fn new(config: RegularStepGdConfig) -> Self {
        config.validate().expect("RSGD config validation failed");
        Self {
            current_step_length: config.initial_step_length,
            config,
            current_loss: None,
            prev_loss: None,
            steps: 0,
            convergence: ConvergenceFlag::default(),
            convergence_reason: None,
            _phantom: PhantomData,
        }
    }

    /// Create with default ITK-matching configuration.
    pub fn default_config() -> Self {
        Self::new(RegularStepGdConfig::default())
    }

    /// Communicate the current loss value before calling `step()`.
    ///
    /// The registration loop must call `set_loss(L(θₖ))` before each
    /// `step()` invocation so the optimizer can compare against the
    /// previous loss to decide whether to accept or revert the step.
    pub fn set_loss(&mut self, loss: f64) {
        self.current_loss = Some(loss);
    }

    /// Whether the optimizer has converged.
    pub fn converged(&self) -> bool {
        self.convergence == ConvergenceFlag::Converged
    }

    /// The convergence reason, if the optimizer has converged.
    pub fn convergence_reason(&self) -> Option<ConvergenceReason> {
        self.convergence_reason
    }

    /// Current step length Δₖ.
    pub fn current_step_length(&self) -> f64 {
        self.current_step_length
    }

    /// Number of accepted (non-reverted) optimization steps.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Compute the global gradient L2-norm ‖∇L‖₂ across all parameters.
    fn compute_gradient_norm(module: &M, grads: &GradientsParams) -> f64 {
        let mut visitor = GradientNormVisitor::<B>::new(grads);
        module.visit(&mut visitor);
        visitor.into_norm()
    }

    /// Apply the RSGD parameter update: θ_new = θ − (Δ / ‖g‖) · g.
    fn apply_rsgd_step(module: M, grads: &mut GradientsParams, effective_lr: f64) -> M {
        let mut mapper = RsgdStepMapper::<B>::new(grads, effective_lr);
        module.map(&mut mapper)
    }
}

impl<M, B> Optimizer<M, B> for RegularStepGradientDescent<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn step(&mut self, module: M, mut gradients: GradientsParams) -> M {
        if self.convergence == ConvergenceFlag::Converged {
            return module;
        }

        let grad_norm = Self::compute_gradient_norm(&module, &gradients);

        if grad_norm < self.config.gradient_tolerance {
            self.convergence = ConvergenceFlag::Converged;
            self.convergence_reason = Some(ConvergenceReason::GradientConvergence);
            tracing::info!(
                "RSGD: gradient convergence (‖g‖ = {:.2e} < tol = {:.2e})",
                grad_norm,
                self.config.gradient_tolerance
            );
            return module;
        }

        if self.current_step_length > self.config.maximum_step_length {
            self.current_step_length = self.config.maximum_step_length;
        }

        let old_module = module.clone();
        let effective_lr = self.current_step_length / grad_norm;
        let new_module = Self::apply_rsgd_step(module, &mut gradients, effective_lr);

        let improved = match (self.current_loss, self.prev_loss) {
            (Some(curr), Some(prev)) => curr <= prev,
            _ => true,
        };

        if !improved {
            self.current_step_length *= self.config.relaxation_factor;
            self.prev_loss = self.current_loss;

            tracing::debug!(
                "RSGD: step rejected (L={:.6e} > prev={:.6e}), shrinking Δ → {:.6e}",
                self.current_loss.unwrap_or(f64::NAN),
                self.prev_loss.unwrap_or(f64::NAN),
                self.current_step_length
            );

            if self.current_step_length < self.config.minimum_step_length {
                self.convergence = ConvergenceFlag::Converged;
                self.convergence_reason = Some(ConvergenceReason::StepConvergence);
                tracing::info!(
                    "RSGD: step convergence (Δ = {:.2e} < min = {:.2e})",
                    self.current_step_length,
                    self.config.minimum_step_length
                );
            }

            old_module
        } else {
            self.prev_loss = self.current_loss;
            self.steps += 1;

            tracing::debug!(
                "RSGD: step accepted (L={:.6e}), steps={}, Δ={:.6e}",
                self.current_loss.unwrap_or(f64::NAN),
                self.steps,
                self.current_step_length
            );

            if self.steps >= self.config.maximum_iterations {
                self.convergence = ConvergenceFlag::Converged;
                self.convergence_reason = Some(ConvergenceReason::MaximumIterations);
                tracing::info!(
                    "RSGD: maximum iterations reached ({})",
                    self.config.maximum_iterations
                );
            }

            // Apply Robbins-Monro decay on accepted steps.
            // Δ_{k+1} = Δ₀ / (1 + λ_decay · k)
            //
            // Note: this decay runs *after* the step-length check above, so
            // `self.current_step_length` reflects the (possibly already
            // relaxation-shrunk) value. When a step was previously rejected
            // and the step length was shrunk by `relaxation_factor`, the
            // decay formula further reduces it. This double-shrink is benign
            // because both mechanisms push Δ toward zero, but callers
            // should prefer `learning_rate_decay` OR small `relaxation_factor`,
            // not both, to avoid overly aggressive step reduction.
            // ITK resets Δ to `initial_step_length` after a failed line
            // search; that behaviour can be obtained by setting
            // `relaxation_factor = 1.0` and relying solely on the
            // Robbins-Monro schedule.
            if self.config.learning_rate_decay > 0.0 {
                let decayed = self.config.initial_step_length
                    / (1.0 + self.config.learning_rate_decay * self.steps as f64);
                self.current_step_length = decayed.max(self.config.minimum_step_length);
            }

            new_module
        }
    }

    fn learning_rate(&self) -> f64 {
        self.current_step_length
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.current_step_length = lr;
    }

    fn telemetry(&self) -> OptimizerTelemetry {
        OptimizerTelemetry {
            algorithm: OptimizerAlgorithm::RegularStepGradientDescent,
            steps: self.steps,
            learning_rate: Some(self.current_step_length),
        }
    }
}
