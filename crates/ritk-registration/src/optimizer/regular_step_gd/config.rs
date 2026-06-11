//! Configuration for the Regular Step Gradient Descent optimizer.

/// Configuration for [`super::RegularStepGradientDescent`].
///
/// Default values match ITK's `RegularStepGradientDescentOptimizerv4`.
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
