//! Regular Step Gradient Descent (RSGD) optimizer.
//!
//! Implements ITK's `RegularStepGradientDescentOptimizerv4` algorithm for
//! image registration parameter optimization.
//!
//! # Algorithm
//!
//! ```text
//! θ₀ = initial parameters
//! Δ₀ = initial_step_length (e.g., 1.0)
//! δ_rel = relaxation_factor (e.g., 0.5)
//! Δ_min = minimum_step_length (e.g., 1e-6)
//! Δ_max = maximum_step_length (e.g., 10.0)
//! t_g = gradient_tolerance (e.g., 1e-6)
//! max_iters = maximum_iterations
//!
//! For iteration k = 0, 1, ..., max_iters-1:
//!   1. Compute gradient: g = ∇θ L(θₖ)
//!   2. Compute gradient magnitude: |g|
//!   3. If |g| < t_g: STOP (gradient convergence)
//!   4. If Δₖ > Δ_max: Δₖ = Δ_max
//!   5. Compute unit gradient: ĝ = g / |g|
//!   6. θₖ₊₁ = θₖ − Δₖ · ĝ  (negative gradient for minimization)
//!   7. Evaluate L(θₖ₊₁)
//!   8. If L(θₖ₊₁) > L(θₖ) (step did not improve):
//!      a. Δₖ₊₁ = Δₖ · δ_rel  (shrink step)
//!      b. Revert: θₖ₊₁ = θₖ  (undo the step)
//!   9. Else (step improved):
//!      a. Δₖ₊₁ = Δₖ  (keep step size)
//!  10. If Δₖ₊₁ < Δ_min: STOP (step convergence)
//! ```
//!
//! # Trait Integration
//!
//! The `Optimizer<M, B>` trait's `step()` method does not receive the loss
//! value, which the RSGD algorithm requires for step-accept/reject decisions.
//! Two mechanisms bridge this gap:
//!
//! 1. **`set_loss()`** — called before each `step()` to communicate the
//!    current loss value. The optimizer compares it against the previous
//!    iteration's loss to decide whether to shrink the step.
//!
//! 2. **Module clone for revert** — when the step does not improve the loss,
//!    the optimizer returns the pre-step module clone, effectively reverting
//!    the parameter update (ITK-faithful behavior).
//!
//! # Gradient Normalization
//!
//! RSGD normalizes the gradient to unit length and steps by `current_step_length`
//! in the negative gradient direction. This is achieved by setting the effective
//! learning rate to `step_length / gradient_norm` when delegating to Burn's
//! SGD optimizer.
//!
//! # References
//!
//! - ITK `RegularStepGradientDescentOptimizerv4`:
//!   <https://itk.org/Doxygen/html/classitk_1_1RegularStepGradientDescentOptimizerv4.html>

use crate::optimizer::{Optimizer, OptimizerTelemetry};
use burn::module::{AutodiffModule, Param};
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};
use std::marker::PhantomData;

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for [`RegularStepGradientDescent`].
///
/// Default values match ITK's `RegularStepGradientDescentOptimizerv4`.
#[derive(Debug, Clone)]
pub struct RegularStepGdConfig {
    /// Initial step length in parameter space.
    pub initial_step_length: f64,
    /// Multiplicative factor applied to shrink the step when a step
    /// fails to decrease the loss. Must be in (0, 1).
    pub relaxation_factor: f64,
    /// Step length threshold below which the optimizer reports step convergence.
    pub minimum_step_length: f64,
    /// Upper bound on step length (clamped each iteration).
    pub maximum_step_length: f64,
    /// Gradient L2-norm threshold below which the optimizer reports
    /// gradient convergence.
    pub gradient_tolerance: f64,
    /// Maximum number of accepted optimization steps.
    pub maximum_iterations: usize,
}

impl Default for RegularStepGdConfig {
    fn default() -> Self {
        Self {
            initial_step_length: 1.0,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-6,
            maximum_step_length: 10.0,
            gradient_tolerance: 1e-6,
            maximum_iterations: 200,
        }
    }
}

impl RegularStepGdConfig {
    /// Validate configuration invariants.
    ///
    /// # Invariants
    /// - `initial_step_length > 0`
    /// - `0 < relaxation_factor < 1`
    /// - `0 < minimum_step_length < initial_step_length`
    /// - `minimum_step_length < maximum_step_length`
    /// - `gradient_tolerance > 0`
    /// - `maximum_iterations > 0`
    pub fn validate(&self) -> Result<(), String> {
        if self.initial_step_length <= 0.0 {
            return Err(format!(
                "initial_step_length must be > 0, got {}",
                self.initial_step_length
            ));
        }
        if self.relaxation_factor <= 0.0 || self.relaxation_factor >= 1.0 {
            return Err(format!(
                "relaxation_factor must be in (0, 1), got {}",
                self.relaxation_factor
            ));
        }
        if self.minimum_step_length <= 0.0 {
            return Err(format!(
                "minimum_step_length must be > 0, got {}",
                self.minimum_step_length
            ));
        }
        if self.minimum_step_length >= self.initial_step_length {
            return Err(format!(
                "minimum_step_length ({}) must be < initial_step_length ({})",
                self.minimum_step_length, self.initial_step_length
            ));
        }
        if self.minimum_step_length >= self.maximum_step_length {
            return Err(format!(
                "minimum_step_length ({}) must be < maximum_step_length ({})",
                self.minimum_step_length, self.maximum_step_length
            ));
        }
        if self.gradient_tolerance <= 0.0 {
            return Err(format!(
                "gradient_tolerance must be > 0, got {}",
                self.gradient_tolerance
            ));
        }
        if self.maximum_iterations == 0 {
            return Err("maximum_iterations must be > 0".to_string());
        }
        Ok(())
    }
}

// ─── Convergence ─────────────────────────────────────────────────────────────

/// Reason the optimizer stopped iterating.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceReason {
    /// Gradient L2-norm fell below `gradient_tolerance`.
    GradientConvergence,
    /// Step length fell below `minimum_step_length` after relaxation.
    StepConvergence,
    /// Number of accepted steps reached `maximum_iterations`.
    MaximumIterations,
}

// ─── Gradient Norm Visitor ───────────────────────────────────────────────────

/// `ModuleVisitor` that accumulates the squared-L2-norm of all gradient tensors
/// stored in a [`GradientsParams`], producing the global gradient norm
/// ‖∇L‖₂ = √(Σᵢ ‖gᵢ‖₂²) across all parameter tensors.
///
/// Type parameter `B` is `AutodiffBackend`. Gradients are retrieved on
/// `B::InnerBackend` (the non-autodiff backend) since `GradientsParams`
/// stores inner-backend tensors.
struct GradientNormVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    norm_sq: f64,
    _phantom: PhantomData<B>,
}

impl<'a, B: AutodiffBackend> GradientNormVisitor<'a, B> {
    fn new(grads: &'a GradientsParams) -> Self {
        Self {
            grads,
            norm_sq: 0.0,
            _phantom: PhantomData,
        }
    }

    fn into_norm(self) -> f64 {
        self.norm_sq.sqrt()
    }
}

impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientNormVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
            let data = grad.to_data();
            if let Ok(slice) = data.as_slice::<f32>() {
                for &v in slice {
                    let vf: f64 = v.elem();
                    self.norm_sq += vf * vf;
                }
            } else if let Ok(slice) = data.as_slice::<f64>() {
                for &v in slice {
                    self.norm_sq += v * v;
                }
            }
        }
    }
}

// ─── Normalized Gradient Mapper ──────────────────────────────────────────────

/// `ModuleMapper` that applies the RSGD update rule on an AutodiffModule:
///
/// ```text
/// θ_new = θ_old − (step_length / grad_norm) · g
/// ```
///
/// This is mathematically equivalent to:
///
/// ```text
/// θ_new = θ_old − step_length · ĝ
/// ```
///
/// where ĝ = g / ‖g‖₂ is the unit gradient direction.
///
/// The mapper reads gradients from `GradientsParams` (consuming them via
/// `remove`), scales by `effective_lr = step_length / grad_norm`, and
/// subtracts from the parameter tensor. Operates on the inner (non-autodiff)
/// backend tensors, then wraps the result back into the autodiff tensor.
struct RsgdStepMapper<'a, B: AutodiffBackend> {
    grads: &'a mut GradientsParams,
    effective_lr: f64,
    _phantom: PhantomData<B>,
}

impl<'a, B: AutodiffBackend> RsgdStepMapper<'a, B> {
    fn new(grads: &'a mut GradientsParams, effective_lr: f64) -> Self {
        Self {
            grads,
            effective_lr,
            _phantom: PhantomData,
        }
    }
}

impl<B: AutodiffBackend> burn::module::ModuleMapper<B> for RsgdStepMapper<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let is_require_grad = tensor.is_require_grad();

        let tensor = if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
            // Operate on inner-backend tensors (no autodiff tracking)
            let inner_tensor = tensor.inner();

            // θ_new = θ − lr · g,  where lr = step_length / grad_norm
            let delta = grad.mul_scalar(self.effective_lr);
            let updated = inner_tensor.sub(delta);

            // Wrap back into autodiff tensor
            let mut updated = Tensor::<B, D>::from_inner(updated);
            if is_require_grad {
                updated = updated.require_grad();
            }
            updated
        } else {
            tensor
        };

        Param::from_mapped_value(id, tensor, mapper)
    }
}

// ─── Optimizer Struct ────────────────────────────────────────────────────────

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
    /// Represents L(θₖ) computed at the current parameter values.
    current_loss: Option<f64>,
    /// Loss from the previous accepted step, L(θₖ₋₁).
    prev_loss: Option<f64>,
    /// Number of accepted (non-reverted) optimization steps.
    steps: usize,
    /// Whether the optimizer has converged.
    converged: bool,
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
            converged: false,
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
        self.converged
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
    ///
    /// Uses the `ModuleVisitor` pattern to visit each parameter tensor,
    /// extract its gradient from `GradientsParams`, and accumulate the
    /// squared-norm. This avoids hardcoding tensor dimensions.
    fn compute_gradient_norm(module: &M, grads: &GradientsParams) -> f64 {
        let mut visitor = GradientNormVisitor::<B>::new(grads);
        module.visit(&mut visitor);
        visitor.into_norm()
    }

    /// Apply the RSGD parameter update: θ_new = θ − (Δ / ‖g‖) · g.
    ///
    /// Uses the `ModuleMapper` pattern to visit each parameter tensor,
    /// extract its gradient, scale by `effective_lr`, and subtract from
    /// the parameter value.
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
        // ── Already converged: return module unchanged ──────────────────
        if self.converged {
            return module;
        }

        // ── 1. Compute gradient magnitude ‖g‖₂ ────────────────────────
        let grad_norm = Self::compute_gradient_norm(&module, &gradients);

        // ── 2. Gradient convergence check ──────────────────────────────
        if grad_norm < self.config.gradient_tolerance {
            self.converged = true;
            self.convergence_reason = Some(ConvergenceReason::GradientConvergence);
            tracing::info!(
                "RSGD: gradient convergence (‖g‖ = {:.2e} < tol = {:.2e})",
                grad_norm,
                self.config.gradient_tolerance
            );
            return module;
        }

        // ── 3. Clamp step length to maximum ────────────────────────────
        if self.current_step_length > self.config.maximum_step_length {
            self.current_step_length = self.config.maximum_step_length;
        }

        // ── 4. Save module clone for potential revert ───────────────────
        let old_module = module.clone();

        // ── 5. Compute effective learning rate and apply step ───────────
        //    effective_lr = Δₖ / ‖g‖₂
        //    so the update θ_new = θ − effective_lr · g = θ − Δₖ · ĝ
        let effective_lr = self.current_step_length / grad_norm;

        let new_module = Self::apply_rsgd_step(module, &mut gradients, effective_lr);

        // ── 6. Check loss improvement ──────────────────────────────────
        //    current_loss = L(θₖ), set via set_loss() before this call
        //    prev_loss = L(θₖ₋₁), stored from the previous accepted step
        let improved = match (self.current_loss, self.prev_loss) {
            (Some(curr), Some(prev)) => curr <= prev,
            _ => true, // First step: no previous loss → always accept
        };

        if !improved {
            // ── Step did not improve: revert and shrink ─────────────────
            self.current_step_length *= self.config.relaxation_factor;

            // Update loss tracking: the current loss becomes prev for next iteration
            // even though we reverted, the loss at the current point is unchanged
            self.prev_loss = self.current_loss;

            tracing::debug!(
                "RSGD: step rejected (L={:.6e} > prev={:.6e}), shrinking Δ → {:.6e}",
                self.current_loss.unwrap_or(f64::NAN),
                self.prev_loss.unwrap_or(f64::NAN),
                self.current_step_length
            );

            // Check step convergence after shrinking
            if self.current_step_length < self.config.minimum_step_length {
                self.converged = true;
                self.convergence_reason = Some(ConvergenceReason::StepConvergence);
                tracing::info!(
                    "RSGD: step convergence (Δ = {:.2e} < min = {:.2e})",
                    self.current_step_length,
                    self.config.minimum_step_length
                );
            }

            old_module
        } else {
            // ── Step improved: accept and increment counter ─────────────
            self.prev_loss = self.current_loss;
            self.steps += 1;

            tracing::debug!(
                "RSGD: step accepted (L={:.6e}), steps={}, Δ={:.6e}",
                self.current_loss.unwrap_or(f64::NAN),
                self.steps,
                self.current_step_length
            );

            // Check maximum iterations
            if self.steps >= self.config.maximum_iterations {
                self.converged = true;
                self.convergence_reason = Some(ConvergenceReason::MaximumIterations);
                tracing::info!(
                    "RSGD: maximum iterations reached ({})",
                    self.config.maximum_iterations
                );
            }

            new_module
        }
    }

    fn learning_rate(&self) -> f64 {
        self.current_step_length
    }

    fn set_learning_rate(&mut self, lr: f64) {
        // Map learning rate to step length for compatibility with the
        // Optimizer trait. The Registration framework calls this method
        // with the user-provided learning_rate, which RSGD interprets
        // as the initial step length.
        self.current_step_length = lr;
    }

    fn telemetry(&self) -> OptimizerTelemetry {
        OptimizerTelemetry {
            algorithm: "RegularStepGradientDescent",
            steps: self.steps,
            learning_rate: Some(self.current_step_length),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Autodiff;
    use burn::module::Module;
    use burn::nn::Linear;
    use burn::tensor::Tensor;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestModule = Linear<TestBackend>;

    // ── Shared test module: f(θ) = Σᵢ θᵢ² ─────────────────────────────
    //
    // A minimal 1-D parameter module used to test RSGD step mechanics
    // with a known analytical gradient ∇f = 2θ.
    // The type parameter B is `Backend` (matching Burn conventions for Module derive).
    // When used with `Autodiff<NdArray<f32>>`, the autodiff backend automatically
    // provides the `AutodiffModule` impl via the `Module` derive macro.
    #[derive(Module, Debug)]
    struct Quadratic<B: burn::tensor::backend::Backend> {
        x: Param<Tensor<B, 1>>,
    }

    impl<B: burn::tensor::backend::Backend> Quadratic<B> {
        fn new(x0: &[f32], device: &B::Device) -> Self {
            let x = Tensor::<B, 1>::from_data(burn::tensor::TensorData::from(x0), device);
            Self {
                x: Param::from_tensor(x),
            }
        }

        /// f(θ) = Σᵢ θᵢ² → element-wise square (autodiff-tracked)
        fn forward(&self) -> Tensor<B, 1> {
            let x = self.x.val();
            x.clone() * x
        }

        /// Scalar loss value L = Σᵢ θᵢ² (computed without autodiff)
        fn loss_value(&self) -> f64 {
            let x = self.x.val();
            let data = x.to_data();
            let slice = data.as_slice::<f32>().unwrap();
            slice.iter().map(|&v| (v as f64) * (v as f64)).sum()
        }

        /// First element of the parameter vector.
        fn param_value(&self) -> f64 {
            let x = self.x.val();
            let data = x.to_data();
            let slice = data.as_slice::<f32>().unwrap();
            slice[0] as f64
        }
    }

    // ── Config validation ─────────────────────────────────────────────────

    #[test]
    fn default_config_validates() {
        assert!(RegularStepGdConfig::default().validate().is_ok());
    }

    #[test]
    fn config_rejects_zero_initial_step() {
        let mut cfg = RegularStepGdConfig::default();
        cfg.initial_step_length = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_negative_initial_step() {
        let mut cfg = RegularStepGdConfig::default();
        cfg.initial_step_length = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_relaxation_zero() {
        let mut cfg = RegularStepGdConfig::default();
        cfg.relaxation_factor = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_relaxation_one() {
        let mut cfg = RegularStepGdConfig::default();
        cfg.relaxation_factor = 1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_min_step_ge_initial() {
        let mut cfg = RegularStepGdConfig::default();
        cfg.minimum_step_length = 1.0; // equal to initial
        assert!(cfg.validate().is_err());

        cfg.minimum_step_length = 2.0; // greater than initial
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_min_step_ge_max() {
        let mut cfg = RegularStepGdConfig::default();
        cfg.minimum_step_length = 10.0;
        cfg.maximum_step_length = 10.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_zero_gradient_tolerance() {
        let mut cfg = RegularStepGdConfig::default();
        cfg.gradient_tolerance = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_zero_max_iterations() {
        let mut cfg = RegularStepGdConfig::default();
        cfg.maximum_iterations = 0;
        assert!(cfg.validate().is_err());
    }

    // ── ConvergenceReason equality ────────────────────────────────────────

    #[test]
    fn convergence_reason_equality() {
        assert_eq!(
            ConvergenceReason::GradientConvergence,
            ConvergenceReason::GradientConvergence
        );
        assert_ne!(
            ConvergenceReason::GradientConvergence,
            ConvergenceReason::StepConvergence
        );
        assert_ne!(
            ConvergenceReason::StepConvergence,
            ConvergenceReason::MaximumIterations
        );
    }

    // ── Structural invariants ─────────────────────────────────────────────

    #[test]
    fn rsgd_default_initial_state() {
        let rsgd: RegularStepGradientDescent<TestModule, TestBackend> =
            RegularStepGradientDescent::new(RegularStepGdConfig::default());

        assert!(!rsgd.converged());
        assert_eq!(rsgd.convergence_reason(), None);
        assert_eq!(rsgd.steps(), 0);
        assert!((rsgd.current_step_length() - 1.0).abs() < 1e-12);
        assert!((rsgd.learning_rate() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rsgd_set_learning_rate_updates_step_length() {
        let mut rsgd: RegularStepGradientDescent<TestModule, TestBackend> =
            RegularStepGradientDescent::new(RegularStepGdConfig::default());

        rsgd.set_learning_rate(0.5);
        assert!((rsgd.learning_rate() - 0.5).abs() < 1e-12);
        assert!((rsgd.current_step_length() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn rsgd_telemetry_reports_algorithm_name() {
        let rsgd: RegularStepGradientDescent<TestModule, TestBackend> =
            RegularStepGradientDescent::new(RegularStepGdConfig::default());

        let telemetry = rsgd.telemetry();
        assert_eq!(telemetry.algorithm, "RegularStepGradientDescent");
        assert_eq!(telemetry.steps, 0);
        assert!(telemetry.learning_rate.is_some());
    }

    #[test]
    fn rsgd_set_loss_stores_current_loss() {
        let mut rsgd: RegularStepGradientDescent<TestModule, TestBackend> =
            RegularStepGradientDescent::new(RegularStepGdConfig::default());

        rsgd.set_loss(1.5);
        assert!(!rsgd.converged());
    }

    // ── Functional test: quadratic minimization ───────────────────────────
    //
    // Minimize f(x) = xᵀx using RSGD starting from x₀ = [5.0, -3.0].
    // The gradient is ∇f = 2x, so the unit gradient direction is x/‖x‖.
    // Each RSGD step moves in the direction -Δ · x/‖x‖ = -Δ · x̂.

    #[test]
    fn rsgd_minimizes_quadratic_function() {
        let device = Default::default();
        let mut module = Quadratic::<TestBackend>::new(&[5.0, -3.0], &device);

        let config = RegularStepGdConfig {
            initial_step_length: 0.5,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-10,
            maximum_step_length: 10.0,
            gradient_tolerance: 1e-8,
            maximum_iterations: 500,
        };

        let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
            RegularStepGradientDescent::new(config);

        let initial_loss = module.loss_value();
        assert!(
            initial_loss > 1.0,
            "Initial loss must be significantly > 0; got {initial_loss}"
        );

        for _ in 0..1000 {
            if optimizer.converged() {
                break;
            }
            let loss = module.forward();
            let loss_val = module.loss_value();
            optimizer.set_loss(loss_val);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &module);
            module = optimizer.step(module, grads_params);
        }

        let final_loss = module.loss_value();

        assert!(
            final_loss < initial_loss * 0.5,
            "RSGD must reduce loss by at least 50%; initial={initial_loss:.6e}, final={final_loss:.6e}, steps={}, converged={}, reason={:?}",
            optimizer.steps(),
            optimizer.converged(),
            optimizer.convergence_reason()
        );

        assert!(
            optimizer.converged(),
            "RSGD should have converged; steps={}, Δ={:.6e}",
            optimizer.steps(),
            optimizer.current_step_length()
        );
    }

    // ── Functional test: gradient convergence ─────────────────────────────
    //
    // Starting near the minimum, the gradient norm should fall below
    // gradient_tolerance, triggering gradient convergence.

    #[test]
    fn rsgd_detects_gradient_convergence() {
        let device = Default::default();
        let mut module = Quadratic::<TestBackend>::new(&[1e-8, -1e-8], &device);

        let config = RegularStepGdConfig {
            initial_step_length: 0.1,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-20,
            maximum_step_length: 10.0,
            gradient_tolerance: 1e-5,
            maximum_iterations: 100,
        };

        let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
            RegularStepGradientDescent::new(config);

        for _ in 0..10 {
            if optimizer.converged() {
                break;
            }
            let loss = module.forward();
            let loss_val = module.loss_value();
            optimizer.set_loss(loss_val);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &module);
            module = optimizer.step(module, grads_params);
        }

        assert!(
            optimizer.converged(),
            "RSGD should converge near the minimum"
        );
        assert_eq!(
            optimizer.convergence_reason(),
            Some(ConvergenceReason::GradientConvergence),
            "Should report gradient convergence when starting near minimum"
        );
    }

    // ── Functional test: step convergence from overshooting ──────────────
    //
    // With a very large initial step, every step overshoots,
    // causing repeated shrinkage until the step falls below minimum_step_length.

    #[test]
    fn rsgd_detects_step_convergence() {
        let device = Default::default();
        let mut module = Quadratic::<TestBackend>::new(&[1.0, -1.0], &device);

        let config = RegularStepGdConfig {
            initial_step_length: 100.0,
            relaxation_factor: 0.5,
            minimum_step_length: 0.1,
            maximum_step_length: 1000.0,
            gradient_tolerance: 1e-20,
            maximum_iterations: 10000,
        };

        let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
            RegularStepGradientDescent::new(config);

        for _ in 0..1000 {
            if optimizer.converged() {
                break;
            }
            let loss = module.forward();
            let loss_val = module.loss_value();
            optimizer.set_loss(loss_val);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &module);
            module = optimizer.step(module, grads_params);
        }

        assert!(
            optimizer.converged(),
            "RSGD should converge; steps={}, Δ={:.6e}",
            optimizer.steps(),
            optimizer.current_step_length()
        );
        assert_eq!(
            optimizer.convergence_reason(),
            Some(ConvergenceReason::StepConvergence),
            "Should report step convergence; got {:?}",
            optimizer.convergence_reason()
        );
    }

    // ── Functional test: maximum iterations ───────────────────────────────

    #[test]
    fn rsgd_detects_maximum_iterations() {
        let device = Default::default();
        let mut module = Quadratic::<TestBackend>::new(&[10.0, -10.0], &device);

        let config = RegularStepGdConfig {
            initial_step_length: 0.01,
            relaxation_factor: 0.9,
            minimum_step_length: 1e-30,
            maximum_step_length: 10.0,
            gradient_tolerance: 1e-30,
            maximum_iterations: 5,
        };

        let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
            RegularStepGradientDescent::new(config);

        for _ in 0..100 {
            if optimizer.converged() {
                break;
            }
            let loss = module.forward();
            let loss_val = module.loss_value();
            optimizer.set_loss(loss_val);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &module);
            module = optimizer.step(module, grads_params);
        }

        assert!(optimizer.converged(), "RSGD should converge");
        assert_eq!(
            optimizer.convergence_reason(),
            Some(ConvergenceReason::MaximumIterations),
            "Should report maximum iterations; got {:?}",
            optimizer.convergence_reason()
        );
        assert_eq!(optimizer.steps(), 5);
    }

    // ── Functional test: revert on loss increase ─────────────────────────
    //
    // Verify that when a step increases the loss, the parameters are
    // reverted to their pre-step values.
    //
    // Starting at x = [1.0] with step_length = 3.0:
    //   grad = 2·1.0 = 2.0, grad_norm = 2.0
    //   effective_lr = 3.0 / 2.0 = 1.5
    //   x_new = 1.0 − 1.5 · 2.0 = 1.0 − 3.0 = −2.0
    //   First step is always accepted (no prev_loss).
    //   After first step, prev_loss = 1.0 (L(θ₀)).
    //
    //   On second step, L(θ₁) = 4.0 > 1.0 = L(θ₀) → reject, revert, shrink.

    #[test]
    fn rsgd_reverts_on_loss_increase() {
        let device = Default::default();
        let mut module = Quadratic::<TestBackend>::new(&[1.0], &device);

        let config = RegularStepGdConfig {
            initial_step_length: 3.0,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-20,
            maximum_step_length: 100.0,
            gradient_tolerance: 1e-30,
            maximum_iterations: 100,
        };

        let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
            RegularStepGradientDescent::new(config);

        // First step: L(θ₀) = 1.0
        let loss = module.forward();
        let loss_val = module.loss_value();
        assert!((loss_val - 1.0).abs() < 1e-6, "Initial loss should be 1.0");
        optimizer.set_loss(loss_val);

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);

        // After first step (accepted by default): x ≈ −2.0
        let first_step_x = module.param_value();
        assert!(
            (first_step_x - (-2.0)).abs() < 0.01,
            "First step should move x to ≈ −2.0; got {first_step_x:.4}"
        );

        // Second step: L(θ₁) = 4.0 > prev_loss = 1.0 → reject
        let loss = module.forward();
        let loss_val = module.loss_value();
        optimizer.set_loss(loss_val);

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);
        module = optimizer.step(module, grads_params);

        // After rejection, x should be reverted to −2.0
        let reverted_x = module.param_value();
        assert!(
            (reverted_x - first_step_x).abs() < 0.01,
            "After rejection, x should revert to {first_step_x:.4}; got {reverted_x:.4}"
        );

        // Step length should have shrunk: 3.0 × 0.5 = 1.5
        assert!(
            (optimizer.current_step_length() - 1.5).abs() < 1e-10,
            "Step length should be 1.5; got {}",
            optimizer.current_step_length()
        );
    }

    // ── Functional test: converged optimizer returns module unchanged ─────

    #[test]
    fn rsgd_returns_module_unchanged_after_convergence() {
        let device = Default::default();
        let mut module = Quadratic::<TestBackend>::new(&[1e-8], &device);

        let config = RegularStepGdConfig {
            initial_step_length: 0.1,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-20,
            maximum_step_length: 10.0,
            gradient_tolerance: 1e-5,
            maximum_iterations: 100,
        };

        let mut optimizer: RegularStepGradientDescent<Quadratic<TestBackend>, TestBackend> =
            RegularStepGradientDescent::new(config);

        for _ in 0..20 {
            if optimizer.converged() {
                break;
            }
            let loss = module.forward();
            let loss_val = module.loss_value();
            optimizer.set_loss(loss_val);
            let grads = loss.backward();
            let grads_params = GradientsParams::from_grads(grads, &module);
            module = optimizer.step(module, grads_params);
        }

        assert!(optimizer.converged());

        let pre_step_x = module.x.val().to_data();
        let pre_slice = pre_step_x.as_slice::<f32>().unwrap().to_vec();

        let loss = module.forward();
        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &module);

        optimizer.set_loss(1.0);
        module = optimizer.step(module, grads_params);

        let post_step_x = module.x.val().to_data();
        let post_slice = post_step_x.as_slice::<f32>().unwrap();

        for (i, (pre, post)) in pre_slice.iter().zip(post_slice.iter()).enumerate() {
            assert!(
                (pre - post).abs() < 1e-10,
                "Parameter [{i}] unchanged after convergence: pre={pre}, post={post}"
            );
        }
    }
}
