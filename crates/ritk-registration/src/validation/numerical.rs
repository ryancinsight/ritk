//! Numerical constraint evaluation bounding configurations for optimization solvers.

use super::config::{NumericalCheck, ValidationConfig};
use crate::error::{RegistrationError, Result};
use burn::tensor::{backend::Backend, ElementConversion, Tensor};

pub fn validate_tensor<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    config: &ValidationConfig,
) -> Result<()> {
    if config.numerical_check == NumericalCheck::Disabled {
        return Ok(());
    }

    if let (Some(min), Some(max)) = (config.min_value, config.max_value) {
        let min_val = tensor.clone().min().into_scalar();
        let max_val = tensor.clone().max().into_scalar();

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

pub fn validate_learning_rate(lr: f64) -> Result<()> {
    if lr <= 0.0 {
        return Err(RegistrationError::invalid_configuration(format!(
            "Learning rate must be positive, got {}",
            lr
        )));
    }
    if lr > 10.0 {
        return Err(RegistrationError::invalid_configuration(format!(
            "Learning rate too large: {}",
            lr
        )));
    }
    if lr < 1e-10 {
        return Err(RegistrationError::invalid_configuration(format!(
            "Learning rate too small: {}",
            lr
        )));
    }
    Ok(())
}

pub fn validate_iterations(iterations: usize) -> Result<()> {
    if iterations == 0 {
        return Err(RegistrationError::invalid_configuration(
            "Iterations must be positive",
        ));
    }
    if iterations > 1_000_000 {
        return Err(RegistrationError::invalid_configuration(format!(
            "Iterations too large: {}",
            iterations
        )));
    }
    Ok(())
}

pub fn validate_histogram_params(
    num_bins: usize,
    min_intensity: f32,
    max_intensity: f32,
) -> Result<()> {
    if num_bins < 2 {
        return Err(RegistrationError::invalid_configuration(format!(
            "Number of bins must be at least 2, got {}",
            num_bins
        )));
    }
    if num_bins > 1024 {
        return Err(RegistrationError::invalid_configuration(format!(
            "Number of bins too large: {}",
            num_bins
        )));
    }
    if min_intensity >= max_intensity {
        return Err(RegistrationError::invalid_configuration(format!(
            "min_intensity ({}) must be less than max_intensity ({})",
            min_intensity, max_intensity
        )));
    }
    Ok(())
}

pub fn validate_lbfgs_history_size(history_size: usize) -> Result<()> {
    if history_size < 1 {
        return Err(RegistrationError::invalid_configuration(
            "L-BFGS history size must be at least 1",
        ));
    }
    if history_size > 100 {
        return Err(RegistrationError::invalid_configuration(format!(
            "L-BFGS history size too large: {}",
            history_size
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    fn make_tensor_1d(vals: Vec<f32>) -> Tensor<B, 1> {
        let n = vals.len();
        let device = Default::default();
        Tensor::<B, 1>::from_data(TensorData::new(vals, Shape::new([n])), &device)
    }

    // ── validate_tensor ────────────────────────────────────────────────────

    #[test]
    fn validate_tensor_within_bounds_passes() {
        let t = make_tensor_1d(vec![0.0, 500.0, -500.0]);
        let config = ValidationConfig::default().with_value_bounds(-1e6, 1e6);
        assert!(validate_tensor(&t, &config).is_ok());
    }

    #[test]
    fn validate_tensor_above_max_fails() {
        let t = make_tensor_1d(vec![0.0, 2e6]);
        let config = ValidationConfig::default().with_value_bounds(-1e6, 1e6);
        assert!(
            validate_tensor(&t, &config).is_err(),
            "tensor with value > max_value must fail validation"
        );
    }

    #[test]
    fn validate_tensor_skipped_when_checks_disabled() {
        // Values well outside bounds, but check is disabled.
        let t = make_tensor_1d(vec![1e10]);
        let config = ValidationConfig::default().without_numerical_checks();
        assert!(
            validate_tensor(&t, &config).is_ok(),
            "validation must be skipped when check_numerical_stability = false"
        );
    }

    // ── validate_learning_rate ─────────────────────────────────────────────

    #[test]
    fn validate_lr_typical_passes() {
        assert!(validate_learning_rate(1e-3).is_ok());
        assert!(validate_learning_rate(0.1).is_ok());
        assert!(validate_learning_rate(1.0).is_ok());
    }

    #[test]
    fn validate_lr_zero_fails() {
        assert!(validate_learning_rate(0.0).is_err(), "lr = 0 must fail");
    }

    #[test]
    fn validate_lr_negative_fails() {
        assert!(
            validate_learning_rate(-1e-4).is_err(),
            "negative lr must fail"
        );
    }

    #[test]
    fn validate_lr_too_large_fails() {
        assert!(validate_learning_rate(11.0).is_err(), "lr > 10 must fail");
    }

    #[test]
    fn validate_lr_too_small_fails() {
        assert!(
            validate_learning_rate(1e-11).is_err(),
            "lr < 1e-10 must fail"
        );
    }

    // ── validate_iterations ──────────────────────────────────────────────

    #[test]
    fn validate_iterations_typical_passes() {
        assert!(validate_iterations(1).is_ok());
        assert!(validate_iterations(100).is_ok());
        assert!(validate_iterations(1_000_000).is_ok());
    }

    #[test]
    fn validate_iterations_zero_fails() {
        assert!(validate_iterations(0).is_err(), "iterations = 0 must fail");
    }

    #[test]
    fn validate_iterations_overflow_fails() {
        assert!(
            validate_iterations(1_000_001).is_err(),
            "iterations > 1_000_000 must fail"
        );
    }

    // ── validate_histogram_params ─────────────────────────────────────────

    #[test]
    fn validate_histogram_typical_passes() {
        assert!(validate_histogram_params(32, 0.0, 255.0).is_ok());
    }

    #[test]
    fn validate_histogram_one_bin_fails() {
        assert!(
            validate_histogram_params(1, 0.0, 1.0).is_err(),
            "num_bins = 1 must fail"
        );
    }

    #[test]
    fn validate_histogram_too_many_bins_fails() {
        assert!(
            validate_histogram_params(1025, 0.0, 1.0).is_err(),
            "num_bins > 1024 must fail"
        );
    }

    #[test]
    fn validate_histogram_inverted_range_fails() {
        assert!(
            validate_histogram_params(32, 1.0, 0.0).is_err(),
            "min >= max must fail"
        );
    }

    #[test]
    fn validate_histogram_equal_bounds_fails() {
        assert!(
            validate_histogram_params(32, 5.0, 5.0).is_err(),
            "min == max must fail"
        );
    }

    // ── validate_lbfgs_history_size ─────────────────────────────────────────

    #[test]
    fn validate_lbfgs_typical_passes() {
        assert!(validate_lbfgs_history_size(5).is_ok());
        assert!(validate_lbfgs_history_size(100).is_ok());
    }

    #[test]
    fn validate_lbfgs_zero_fails() {
        assert!(
            validate_lbfgs_history_size(0).is_err(),
            "history_size = 0 must fail"
        );
    }

    #[test]
    fn validate_lbfgs_too_large_fails() {
        assert!(
            validate_lbfgs_history_size(101).is_err(),
            "history_size > 100 must fail"
        );
    }
}
