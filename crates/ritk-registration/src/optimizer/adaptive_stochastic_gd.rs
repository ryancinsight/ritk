//! Adaptive Stochastic Gradient Descent (ASGD) optimizer.
//!
//! Implements the Adaptive Stochastic Gradient Descent algorithm for
//! image registration parameter optimization, based on Klein et al. (2009)
//! and the elastix implementation.
//!
//! # Algorithm
//!
//! The ASGD optimizer uses an adaptive step size mechanism based on the inner
//! product of the current and previous gradients.
//!
//! ```text
//! θ₀ = initial parameters
//! t₀ = 0
//!
//! For iteration k = 0, 1, ..., max_iters-1:
//!   1. Compute gradient: g_k = ∇θ L(θₖ)
//!   2. Compute gradient magnitude: |g_k|
//!   3. If |g_k| < t_g: STOP (gradient convergence)
//!   4. Compute step size: a(t_k) = a / (A + t_k + 1)^alpha
//!   5. θₖ₊₁ = θₖ − a(t_k) · g_k
//!   6. Compute inner product: dot = g_k · g_{k-1}
//!   7. Update adaptive time:
//!      f(dot) = f_min + (f_max - f_min) / (1 + exp(dot / omega))
//!      t_{k+1} = max(0, t_k + f(dot))
//! ```
//!
//! If the gradients point in the same direction (`dot > 0`), the step size decays more
//! slowly (or increases). If they point in opposite directions (`dot < 0`), the step size
//! decays more rapidly to stabilize the optimization.
//!
//! # References
//!
//! - Klein, S., Pluim, J. P. W., Staring, M., & Viergever, M. A. (2009).
//!   "Adaptive stochastic gradient descent optimisation for image registration."
//!   International Journal of Computer Vision, 81(3), 227-239.

use crate::optimizer::{ConvergenceReason, Optimizer, OptimizerTelemetry};
use burn::module::{AutodiffModule, Param};
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{ElementConversion, Tensor};
use std::marker::PhantomData;

// ─── Convergence State ───────────────────────────────────────────────────────

/// Internal convergence state for the adaptive step-size gradient descent optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum ConvergenceFlag {
    /// Optimizer is still iterating.
    #[default]
    Iterating,
    /// Convergence criterion was satisfied.
    Converged,
}

// ─── Configuration ───────────────────────────────────────────────────────────

/// Configuration for [`AdaptiveStochasticGradientDescent`].
#[derive(Debug, Clone)]
pub struct AdaptiveStochasticGdConfig {
    /// Gain parameter `a` in `a / (a_damping + t + 1)^alpha`.
    pub a: f64,
    /// Damping parameter (Klein 2009: `A`) in `a / (a_damping + t + 1)^alpha`.
    pub a_damping: f64,
    /// Decay parameter `alpha` in `a / (a_damping + t + 1)^alpha`.
    pub alpha: f64,
    /// Maximum value of the sigmoid function `f_max`.
    pub sigmoid_max: f64,
    /// Minimum value of the sigmoid function `f_min`.
    pub sigmoid_min: f64,
    /// Scale of the sigmoid function `omega`.
    pub sigmoid_scale: f64,
    /// Gradient L2-norm threshold below which the optimizer reports convergence.
    pub gradient_tolerance: f64,
    /// Maximum number of optimization steps.
    pub maximum_iterations: usize,
}

impl Default for AdaptiveStochasticGdConfig {
    fn default() -> Self {
        Self {
            a: 1.0,
            a_damping: 20.0,
            alpha: 1.0,
            sigmoid_max: 1.0,
            sigmoid_min: -0.5,
            sigmoid_scale: 1e-8,
            gradient_tolerance: 1e-6,
            maximum_iterations: 500,
        }
    }
}

impl AdaptiveStochasticGdConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<(), String> {
        if self.a <= 0.0 {
            return Err(format!("a must be > 0, got {}", self.a));
        }
        if self.a_damping < 0.0 {
            return Err(format!("a_damping must be >= 0, got {}", self.a_damping));
        }
        if self.alpha < 0.0 {
            return Err(format!("alpha must be >= 0, got {}", self.alpha));
        }
        if self.sigmoid_max <= self.sigmoid_min {
            return Err(format!(
                "sigmoid_max ({}) must be > sigmoid_min ({})",
                self.sigmoid_max, self.sigmoid_min
            ));
        }
        if self.sigmoid_scale <= 0.0 {
            return Err(format!(
                "sigmoid_scale must be > 0, got {}",
                self.sigmoid_scale
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

// ─── Gradient Extraction Visitor ─────────────────────────────────────────────

/// `ModuleVisitor` that extracts all gradients into a flattened vector
/// and computes the squared-L2-norm simultaneously.
struct GradientExtractVisitor<'a, B: AutodiffBackend> {
    grads: &'a GradientsParams,
    flat_grads: Vec<f64>,
    norm_sq: f64,
    _phantom: PhantomData<fn() -> B>,
}

impl<'a, B: AutodiffBackend> GradientExtractVisitor<'a, B> {
    fn new(grads: &'a GradientsParams) -> Self {
        Self {
            grads,
            // Capacity: exact count unknown before traversal; 256 is a typical
            // parameter count for 3-D affine (12) to B-spline transforms
            flat_grads: Vec::with_capacity(256),
            norm_sq: 0.0,
            _phantom: PhantomData,
        }
    }

    fn into_result(self) -> (Vec<f64>, f64) {
        (self.flat_grads, self.norm_sq.sqrt())
    }
}

impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientExtractVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(grad) = self.grads.get::<B::InnerBackend, D>(param.id) {
            let data = grad.to_data();
            if let Ok(slice) = data.as_slice::<f32>() {
                for &v in slice {
                    let vf: f64 = v.elem();
                    self.flat_grads.push(vf);
                    self.norm_sq += vf * vf;
                }
            } else if let Ok(slice) = data.as_slice::<f64>() {
                for &v in slice {
                    self.flat_grads.push(v);
                    self.norm_sq += v * v;
                }
            }
        }
    }
}

// ─── Mapper ──────────────────────────────────────────────────────────────────

/// `ModuleMapper` that applies the standard SGD update rule:
///
/// `θ_new = θ_old − learning_rate · g`
struct AsgdStepMapper<'a, B: AutodiffBackend> {
    grads: &'a mut GradientsParams,
    learning_rate: f64,
    _phantom: PhantomData<fn() -> B>,
}

impl<'a, B: AutodiffBackend> AsgdStepMapper<'a, B> {
    fn new(grads: &'a mut GradientsParams, learning_rate: f64) -> Self {
        Self {
            grads,
            learning_rate,
            _phantom: PhantomData,
        }
    }
}

impl<B: AutodiffBackend> burn::module::ModuleMapper<B> for AsgdStepMapper<'_, B> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let is_require_grad = tensor.is_require_grad();

        let tensor = if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(id) {
            let inner_tensor = tensor.inner();
            let delta = grad.mul_scalar(self.learning_rate);
            let updated = inner_tensor.sub(delta);
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

/// Adaptive Stochastic Gradient Descent (ASGD) optimizer.
pub struct AdaptiveStochasticGradientDescent<M: AutodiffModule<B>, B: AutodiffBackend> {
    config: AdaptiveStochasticGdConfig,
    /// Current adaptive time `t_k`.
    t_k: f64,
    /// Flattened gradient from the previous step.
    prev_grad: Option<Vec<f64>>,
    /// Number of steps taken.
    steps: usize,
    /// Whether the optimizer has converged.
    convergence: ConvergenceFlag,
    /// Why the optimizer converged, if it has.
    convergence_reason: Option<ConvergenceReason>,
    _phantom: PhantomData<fn() -> (M, B)>,
}

impl<M: AutodiffModule<B>, B: AutodiffBackend> AdaptiveStochasticGradientDescent<M, B> {
    /// Create a new ASGD optimizer from configuration.
    pub fn new(config: AdaptiveStochasticGdConfig) -> Self {
        config.validate().expect("ASGD config validation failed");

        Self {
            config,
            t_k: 0.0,
            prev_grad: None,
            steps: 0,
            convergence: ConvergenceFlag::default(),
            convergence_reason: None,
            _phantom: PhantomData,
        }
    }

    /// Create with default configuration.
    pub fn default_config() -> Self {
        Self::new(AdaptiveStochasticGdConfig::default())
    }

    /// Whether the optimizer has converged.
    pub fn converged(&self) -> bool {
        self.convergence == ConvergenceFlag::Converged
    }

    /// The convergence reason, if the optimizer has converged.
    pub fn convergence_reason(&self) -> Option<ConvergenceReason> {
        self.convergence_reason
    }

    /// Number of accepted optimization steps.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Current adaptive time `t_k`.
    pub fn t_k(&self) -> f64 {
        self.t_k
    }
}

impl<M, B> Optimizer<M, B> for AdaptiveStochasticGradientDescent<M, B>
where
    M: AutodiffModule<B>,
    B: AutodiffBackend,
{
    fn step(&mut self, module: M, mut gradients: GradientsParams) -> M {
        if self.convergence == ConvergenceFlag::Converged {
            return module;
        }

        // ── 1. Extract gradient and compute norm ──────────────────────
        let mut visitor = GradientExtractVisitor::<B>::new(&gradients);
        module.visit(&mut visitor);
        let (current_grad, grad_norm) = visitor.into_result();

        // ── 2. Gradient convergence check ──────────────────────────────
        if grad_norm < self.config.gradient_tolerance {
            self.convergence = ConvergenceFlag::Converged;
            self.convergence_reason = Some(ConvergenceReason::GradientConvergence);
            tracing::info!(
                "ASGD: gradient convergence (‖g‖ = {:.2e} < tol = {:.2e})",
                grad_norm,
                self.config.gradient_tolerance
            );
            return module;
        }

        // ── 3. Compute step size a(t_k) ───────────────────────────────
        let a_tk = self.config.a / (self.config.a_damping + self.t_k + 1.0).powf(self.config.alpha);

        // ── 4. Apply step ──────────────────────────────────────────────
        let new_module = {
            let mut mapper = AsgdStepMapper::<B>::new(&mut gradients, a_tk);
            module.map(&mut mapper)
        };

        // ── 5. Compute dot product and update t_k ──────────────────────
        if let Some(prev) = &self.prev_grad {
            let mut dot = 0.0;
            for (c, p) in current_grad.iter().zip(prev.iter()) {
                dot += c * p;
            }

            // f(dot) = f_min + (f_max - f_min) / (1 + exp(dot / omega))
            let f_val = self.config.sigmoid_min
                + (self.config.sigmoid_max - self.config.sigmoid_min)
                    / (1.0 + (dot / self.config.sigmoid_scale).exp());

            self.t_k = (self.t_k + f_val).max(0.0);

            tracing::debug!(
                "ASGD: step {}, a(t_k)={:.6e}, dot={:.6e}, f_val={:.6e}, t_k={:.6e}",
                self.steps,
                a_tk,
                dot,
                f_val,
                self.t_k
            );
        } else {
            tracing::debug!(
                "ASGD: step {}, a(t_k)={:.6e}, t_k={:.6e} (first step)",
                self.steps,
                a_tk,
                self.t_k
            );
        }

        // ── 6. Update state ────────────────────────────────────────────
        self.prev_grad = Some(current_grad);
        self.steps += 1;

        if self.steps >= self.config.maximum_iterations {
            self.convergence = ConvergenceFlag::Converged;
            self.convergence_reason = Some(ConvergenceReason::MaximumIterations);
            tracing::info!(
                "ASGD: maximum iterations reached ({})",
                self.config.maximum_iterations
            );
        }

        new_module
    }

    fn learning_rate(&self) -> f64 {
        self.config.a / (self.config.a_damping + self.t_k + 1.0).powf(self.config.alpha)
    }

    fn set_learning_rate(&mut self, lr: f64) {
        // In ASGD, setting learning rate typically means setting the gain `a`.
        self.config.a = lr;
    }

    fn telemetry(&self) -> OptimizerTelemetry {
        OptimizerTelemetry {
            algorithm: "AdaptiveStochasticGradientDescent",
            steps: self.steps,
            learning_rate: Some(self.learning_rate()),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_adaptive_stochastic_gd.rs"]
mod tests;
