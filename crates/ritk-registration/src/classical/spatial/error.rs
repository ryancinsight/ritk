//! `SpatialError` â€” error variants for classical spatial transform operations.

use thiserror::Error;

/// Errors for classical spatial transform operations.
#[derive(Error, Debug)]
pub enum SpatialError {
    #[error("Invalid point set: {0}")]
    InvalidPointSet(String),
    #[error("SVD did not converge: {0}")]
    SvdConvergence(String),
    #[error("Invalid transform matrix: {0}")]
    InvalidTransform(String) }
