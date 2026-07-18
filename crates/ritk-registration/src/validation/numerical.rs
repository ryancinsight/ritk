//! Numerical constraint evaluation bounding configurations for optimization solvers.

use super::config::{NumericalCheck, ValidationConfig};
use crate::error::{RegistrationError, Result};
use coeus_core::CpuAddressableStorage;
use ritk_image::tensor::{Backend, Tensor};

pub fn validate_tensor<B: Backend>(
    tensor: &Tensor<f32, B>,
    config: &ValidationConfig,
) -> Result<()>
where
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    if config.numerical_check == NumericalCheck::Disabled {
        return Ok(());
    }

    if let (Some(min), Some(max)) = (config.min_value, config.max_value) {
        let values = tensor.as_slice();
        let min_val_f64 = values.iter().copied().fold(f32::INFINITY, f32::min) as f64;
        let max_val_f64 = values.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;

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
#[path = "tests_numerical.rs"]
mod tests;
