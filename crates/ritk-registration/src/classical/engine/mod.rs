//! Classical image registration engine.
//!
//! Orchestrates non-ML registration algorithms using pure ndarray primitives:
//! - Kabsch SVD for landmark-based rigid registration
//! - Mutual-information hill-climb for intensity-based rigid/affine registration
//!
//! All operations are deterministic and CPU-based with no deep-learning dependency.

pub mod config;
pub mod metric;
mod registration;
pub mod result;

pub use config::ClassicalConfig;
pub use metric::MutualInformationMetric;
pub use registration::ImageRegistration;
pub use result::RegistrationResult;

#[cfg(test)]
mod tests;
