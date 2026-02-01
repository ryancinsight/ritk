//! Validation utilities for registration operations.
//!
//! This module provides validation functions for ensuring numerical stability,
//! input validity, and bounds checking in registration workflows.

use burn::tensor::{Tensor, backend::Backend, ElementConversion};
use ritk_core::image::Image;
use crate::error::{RegistrationError, Result};

/// Validation configuration.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Maximum allowed gradient norm before clipping.
    pub max_gradient_norm: Option<f64>,
    /// Minimum allowed value for tensor elements.
    pub min_value: Option<f32>,
    /// Maximum allowed value for tensor elements.
    pub max_value: Option<f32>,
    /// Tolerance for NaN/Inf detection.
    pub nan_inf_tolerance: f32,
    /// Enable shape validation.
    pub validate_shapes: bool,
    /// Enable numerical stability checks.
    pub check_numerical_stability: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            max_gradient_norm: Some(1000.0),
            min_value: Some(-1e6),
            max_value: Some(1e6),
            nan_inf_tolerance: 1e-6,
            validate_shapes: true,
            check_numerical_stability: true,
        }
    }
}

impl ValidationConfig {
    /// Create a new validation config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum gradient norm.
    pub fn with_max_gradient_norm(mut self, norm: f64) -> Self {
        self.max_gradient_norm = Some(norm);
        self
    }

    /// Disable gradient clipping.
    pub fn without_gradient_clipping(mut self) -> Self {
        self.max_gradient_norm = None;
        self
    }

    /// Set value bounds.
    pub fn with_value_bounds(mut self, min: f32, max: f32) -> Self {
        self.min_value = Some(min);
        self.max_value = Some(max);
        self
    }

    /// Disable shape validation.
    pub fn without_shape_validation(mut self) -> Self {
        self.validate_shapes = false;
        self
    }

    /// Disable numerical stability checks.
    pub fn without_numerical_checks(mut self) -> Self {
        self.check_numerical_stability = false;
        self
    }
}

/// Validate that two images have compatible shapes.
pub fn validate_image_shapes<B: Backend, const D: usize>(
    fixed: &Image<B, D>,
    moving: &Image<B, D>,
) -> Result<()> {
    let fixed_shape = fixed.shape();
    let moving_shape = moving.shape();

    if fixed_shape != moving_shape {
        return Err(RegistrationError::ShapeMismatch {
            expected: fixed_shape.to_vec(),
            actual: moving_shape.to_vec(),
        });
    }

    Ok(())
}

/// Validate tensor for numerical stability (NaN, Inf, extreme values).
pub fn validate_tensor<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    config: &ValidationConfig,
) -> Result<()> {
    if !config.check_numerical_stability {
        return Ok(());
    }

    // Check value bounds if specified
    if let (Some(min), Some(max)) = (config.min_value, config.max_value) {
        let min_val = tensor.clone().min().into_scalar();
        let max_val = tensor.clone().max().into_scalar();

        // Convert to f64 for comparison using ElementConversion
        let min_val_f64 = min_val.elem::<f64>();
        let max_val_f64 = max_val.elem::<f64>();

        if min_val_f64 < min as f64 || max_val_f64 > max as f64 {
            return Err(RegistrationError::numerical_instability(format!(
                "Tensor values out of bounds: [{:.6}, {:.6}] vs expected [{}, {}]",
                min_val_f64, max_val_f64, min, max
            )));
        }
    }

    Ok(())
}

/// Clip gradients to prevent exploding gradients.
pub fn clip_gradients<B: Backend, const D: usize>(
    gradients: Tensor<B, D>,
    max_norm: f64,
) -> Tensor<B, D> {
    let norm = gradients.clone().powf_scalar(2.0).sum().sqrt();
    let norm_scalar = norm.clone().into_scalar().elem::<f64>();

    if norm_scalar > max_norm {
        let scale = max_norm / norm_scalar;
        gradients.mul_scalar(scale as f32)
    } else {
        gradients
    }
}

/// Validate learning rate.
pub fn validate_learning_rate(lr: f64) -> Result<()> {
    if lr <= 0.0 {
        return Err(RegistrationError::invalid_configuration(
            format!("Learning rate must be positive, got {}", lr),
        ));
    }

    if lr > 10.0 {
        return Err(RegistrationError::invalid_configuration(
            format!("Learning rate too large: {}", lr),
        ));
    }

    if lr < 1e-10 {
        return Err(RegistrationError::invalid_configuration(
            format!("Learning rate too small: {}", lr),
        ));
    }

    Ok(())
}

/// Validate iteration count.
pub fn validate_iterations(iterations: usize) -> Result<()> {
    if iterations == 0 {
        return Err(RegistrationError::invalid_configuration(
            "Iterations must be positive",
        ));
    }

    if iterations > 1_000_000 {
        return Err(RegistrationError::invalid_configuration(
            format!("Iterations too large: {}", iterations),
        ));
    }

    Ok(())
}

/// Validate histogram parameters for mutual information.
pub fn validate_histogram_params(num_bins: usize, min_intensity: f32, max_intensity: f32) -> Result<()> {
    if num_bins < 2 {
        return Err(RegistrationError::invalid_configuration(
            format!("Number of bins must be at least 2, got {}", num_bins),
        ));
    }

    if num_bins > 1024 {
        return Err(RegistrationError::invalid_configuration(
            format!("Number of bins too large: {}", num_bins),
        ));
    }

    if min_intensity >= max_intensity {
        return Err(RegistrationError::invalid_configuration(format!(
            "min_intensity ({}) must be less than max_intensity ({})",
            min_intensity, max_intensity
        )));
    }

    Ok(())
}

/// Validate L-BFGS history size.
pub fn validate_lbfgs_history_size(history_size: usize) -> Result<()> {
    if history_size < 1 {
        return Err(RegistrationError::invalid_configuration(
            "L-BFGS history size must be at least 1",
        ));
    }

    if history_size > 100 {
        return Err(RegistrationError::invalid_configuration(
            format!("L-BFGS history size too large: {}", history_size),
        ));
    }

    Ok(())
}

/// Check for convergence based on loss history.
#[derive(Debug, Clone)]
pub struct ConvergenceChecker {
    /// Minimum relative improvement to consider converged.
    pub min_improvement: f64,
    /// Number of iterations to check for improvement.
    pub patience: usize,
    /// Minimum absolute loss to consider converged.
    pub min_loss: Option<f64>,
}

impl Default for ConvergenceChecker {
    fn default() -> Self {
        Self {
            min_improvement: 1e-6,
            patience: 50,
            min_loss: None,
        }
    }
}

impl ConvergenceChecker {
    /// Create a new convergence checker.
    pub fn new(min_improvement: f64, patience: usize) -> Self {
        Self {
            min_improvement,
            patience,
            min_loss: None,
        }
    }

    /// Set minimum loss threshold.
    pub fn with_min_loss(mut self, min_loss: f64) -> Self {
        self.min_loss = Some(min_loss);
        self
    }

    /// Check if converged based on loss history.
    ///
    /// Returns true if converged, false otherwise.
    pub fn check_convergence(&self, loss_history: &[f64]) -> bool {
        if loss_history.is_empty() {
            return false;
        }

        // Check minimum loss threshold
        if let Some(min_loss) = self.min_loss {
            if loss_history.last().unwrap() < &min_loss {
                return true;
            }
        }

        // Need at least patience + 1 samples
        if loss_history.len() < self.patience + 1 {
            return false;
        }

        // Check improvement over last patience iterations
        let recent_start = loss_history.len() - self.patience - 1;
        let recent_losses = &loss_history[recent_start..];

        let best_loss = recent_losses.iter().cloned().fold(f64::INFINITY, f64::min);
        let current_loss = *loss_history.last().unwrap();

        let relative_improvement = (best_loss - current_loss) / (best_loss.abs() + 1e-10);

        relative_improvement < self.min_improvement
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn test_validate_learning_rate() {
        assert!(validate_learning_rate(0.01).is_ok());
        assert!(validate_learning_rate(1.0).is_ok());
        assert!(validate_learning_rate(0.0).is_err());
        assert!(validate_learning_rate(-0.01).is_err());
        assert!(validate_learning_rate(100.0).is_err());
        assert!(validate_learning_rate(1e-12).is_err());
    }

    #[test]
    fn test_validate_iterations() {
        assert!(validate_iterations(100).is_ok());
        assert!(validate_iterations(1000).is_ok());
        assert!(validate_iterations(0).is_err());
        assert!(validate_iterations(2_000_000).is_err());
    }

    #[test]
    fn test_validate_histogram_params() {
        assert!(validate_histogram_params(32, 0.0, 255.0).is_ok());
        assert!(validate_histogram_params(1, 0.0, 255.0).is_err());
        assert!(validate_histogram_params(2000, 0.0, 255.0).is_err());
        assert!(validate_histogram_params(32, 255.0, 0.0).is_err());
    }

    #[test]
    fn test_validate_lbfgs_history_size() {
        assert!(validate_lbfgs_history_size(10).is_ok());
        assert!(validate_lbfgs_history_size(0).is_err());
        assert!(validate_lbfgs_history_size(200).is_err());
    }

    #[test]
    fn test_clip_gradients() {
        let device = Default::default();
        let gradients = Tensor::<B, 1>::from_floats([10.0, 10.0, 10.0], &device);
        let clipped = clip_gradients(gradients, 15.0);
        let norm = clipped.clone().powf_scalar(2.0).sum().sqrt().into_scalar();
        assert!((norm - 15.0).abs() < 1e-5);
    }

    #[test]
    fn test_validate_tensor_bounds() {
        let device = Default::default();
        let tensor = Tensor::<B, 1>::from_floats([1.0, 2.0, 3.0], &device);
        let config = ValidationConfig::new().with_value_bounds(0.0, 10.0);
        assert!(validate_tensor(&tensor, &config).is_ok());

        let tensor = Tensor::<B, 1>::from_floats([1.0, 2.0, 20.0], &device);
        assert!(validate_tensor(&tensor, &config).is_err());
    }
}
