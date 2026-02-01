//! Error types for registration operations.
//!
//! This module provides structured error types for registration workflows,
//! enabling better error handling and debugging.

use thiserror::Error;

/// Main error type for registration operations.
#[derive(Error, Debug)]
pub enum RegistrationError {
    /// Error in metric computation.
    #[error("Metric error: {0}")]
    MetricError(String),

    /// Error in optimizer operation.
    #[error("Optimizer error: {0}")]
    OptimizerError(String),

    /// Error in transform operation.
    #[error("Transform error: {0}")]
    TransformError(String),

    /// Error in image validation.
    #[error("Image validation error: {0}")]
    ImageValidationError(String),

    /// Error in interpolation.
    #[error("Interpolation error: {0}")]
    InterpolationError(String),

    /// Numerical instability detected.
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// Convergence failure.
    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Dimension mismatch.
    #[error("Dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Shape mismatch.
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

/// Result type for registration operations.
pub type Result<T> = std::result::Result<T, RegistrationError>;

impl RegistrationError {
    /// Create a metric error.
    pub fn metric(msg: impl Into<String>) -> Self {
        Self::MetricError(msg.into())
    }

    /// Create an optimizer error.
    pub fn optimizer(msg: impl Into<String>) -> Self {
        Self::OptimizerError(msg.into())
    }

    /// Create a transform error.
    pub fn transform(msg: impl Into<String>) -> Self {
        Self::TransformError(msg.into())
    }

    /// Create an image validation error.
    pub fn image_validation(msg: impl Into<String>) -> Self {
        Self::ImageValidationError(msg.into())
    }

    /// Create an interpolation error.
    pub fn interpolation(msg: impl Into<String>) -> Self {
        Self::InterpolationError(msg.into())
    }

    /// Create a numerical instability error.
    pub fn numerical_instability(msg: impl Into<String>) -> Self {
        Self::NumericalInstability(msg.into())
    }

    /// Create a convergence failure error.
    pub fn convergence_failure(msg: impl Into<String>) -> Self {
        Self::ConvergenceFailure(msg.into())
    }

    /// Create an invalid configuration error.
    pub fn invalid_configuration(msg: impl Into<String>) -> Self {
        Self::InvalidConfiguration(msg.into())
    }

    /// Create a dimension mismatch error.
    pub fn dimension_mismatch(msg: impl Into<String>) -> Self {
        Self::DimensionMismatch(msg.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = RegistrationError::metric("test error");
        assert!(matches!(err, RegistrationError::MetricError(_)));
    }

    #[test]
    fn test_error_display() {
        let err = RegistrationError::metric("test error");
        assert_eq!(err.to_string(), "Metric error: test error");
    }

    #[test]
    fn test_shape_mismatch() {
        let err = RegistrationError::ShapeMismatch {
            expected: vec![10, 10],
            actual: vec![5, 5],
        };
        let err_str = err.to_string();
        assert!(err_str.contains("expected"));
        assert!(err_str.contains("got"));
    }
}
