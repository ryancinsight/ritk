/// Classical (non-ML) image registration algorithms.
///
/// This module provides deterministic, CPU-based registration algorithms
/// that do not require a deep-learning backend:
///
/// - **Rigid body**: Kabsch SVD landmark registration + mutual-information hill-climb
/// - **Affine**: 9-DOF MI optimisation (rotation + translation + anisotropic scale)
/// - **Temporal sync**: Cross-correlation phase estimation for multi-modal acquisitions
///
/// All types in this module are re-exported from authoritative locations:
/// - `RegistrationQualityMetrics` from [`crate::validation`]
/// - `TemporalQualityMetrics` from [`crate::validation`]
pub mod engine;
pub mod error;
pub mod spatial;
pub mod temporal;

// Re-export core types for convenience
pub use engine::{ImageRegistration, RegistrationResult};
pub use error::{RegistrationError, Result};
pub use spatial::SpatialTransform;
pub use temporal::TemporalSync;

// Re-export quality metrics from validation (SSOT)
pub use crate::validation::{RegistrationQualityMetrics, TemporalQualityMetrics};
