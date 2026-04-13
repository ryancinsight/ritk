//! Error types for classical registration operations.
use thiserror::Error;

use super::spatial::SpatialError;

/// Error type for classical registration operations.
#[derive(Error, Debug)]
pub enum RegistrationError {
    /// Input validation failure (empty point sets, mismatched dimensions, etc.)
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    /// SVD or other numerical algorithm failed to converge
    #[error("Numerical failure: {0}")]
    NumericalFailure(String),
    /// Optimisation did not converge within the iteration budget
    #[error("Convergence failure: {0}")]
    ConvergenceFailure(String),
}

impl From<SpatialError> for RegistrationError {
    fn from(err: SpatialError) -> Self {
        match err {
            SpatialError::InvalidPointSet(msg) => RegistrationError::InvalidInput(msg),
            SpatialError::SvdConvergence(msg) => RegistrationError::NumericalFailure(msg),
            SpatialError::InvalidTransform(msg) => RegistrationError::InvalidInput(msg),
        }
    }
}

/// Result type for classical registration operations.
pub type Result<T> = std::result::Result<T, RegistrationError>;
