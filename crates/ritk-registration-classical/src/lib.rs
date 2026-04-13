//! Classical (non-ML) image registration for the ritk toolkit.
//!
//! This crate provides deterministic, CPU-based registration algorithms
//! that do not require a deep-learning backend:
//!
//! - **Rigid body**: Kabsch SVD landmark registration + mutual-information hill-climb
//! - **Affine**: 9-DOF MI optimisation (rotation + translation + anisotropic scale)
//! - **Temporal sync**: Cross-correlation phase estimation for multi-modal acquisitions
//!
//! ## Quick start
//!
//! ```no_run
//! use ritk_registration_classical::ImageRegistration;
//! use ndarray::Array2;
//!
//! let reg = ImageRegistration::default();
//!
//! // Landmark-based rigid registration
//! let fixed  = Array2::from_shape_vec((3,3), vec![0.,0.,0., 1.,0.,0., 0.,1.,0.]).unwrap();
//! let moving = Array2::from_shape_vec((3,3), vec![1.,2.,3., 2.,2.,3., 1.,3.,3.]).unwrap();
//! let result = reg.rigid_registration_landmarks(&fixed, &moving).unwrap();
//! ```

pub mod engine;
pub mod error;
pub mod intensity;
pub mod metrics;
pub mod spatial;
pub mod temporal;

#[cfg(test)]
mod tests;

// Re-export core types
pub use engine::{ImageRegistration, RegistrationResult};
pub use error::{RegistrationError, Result};
pub use metrics::{RegistrationQualityMetrics, TemporalQualityMetrics};
pub use spatial::SpatialTransform;
pub use temporal::TemporalSync;
