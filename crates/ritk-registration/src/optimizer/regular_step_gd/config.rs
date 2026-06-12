//! Configuration for the Regular Step Gradient Descent optimizer.

/// Configuration for [`super::RegularStepGradientDescent`].
///
/// Default values match ITK's `RegularStepGradientDescentOptimizerv4`.
///
/// # Adaptive Learning Rate (Robbins-Monro Schedule)
///
/// When `learning_rate_decay` is set (> 0), the step length follows the
/// Robbins-Monro stochastic approximation schedule:
///
/// ```text
/// Δ_k = Δ₀ / (1 + λ_decay · (k − 1))
/// ```
///
/// This guarantees almost-sure convergence in the convex limit by satisfying
/// Σ Δ_k = ∞ and Σ Δ_k² < ∞ (Robbins & Monro, 1951). In practice it
/// stabilises the late-stage registration by reducing overshoot when the
/// gradient norm has already collapsed to the noise floor.
///
/// Set `learning_rate_decay = 0.0` to disable and use the classic fixed-step
/// RSGD behaviour (ITK-compatible).
#[derive(Debug, Clone, Copy)]
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
    /// Robbins-Monro decay factor for adaptive step-length scheduling.
    ///
    /// Step length at iteration k is:
    ///   `Δ_k = initial_step_length / (1 + learning_rate_decay * (k - 1))`
    ///
    /// Typical values: 1e-4–1e-3.  Set to 0.0 to disable (classic RSGD).
    pub learning_rate_decay: f64,
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
            learning_rate_decay: 0.0,
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
