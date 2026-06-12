//! Deep-learning registration loss modules.
//!
//! Separated by concern:
//! - [`lncc`]: Local Normalized Cross Correlation loss
//! - [`ncc`]: Global Normalized Cross Correlation loss
//! - [`grad`]: First-order gradient regularization loss
//! - [`combined`]: Orchestrator combining similarity + regularization

pub mod combined;
pub mod grad;
pub mod lncc;
pub mod ncc;

pub use combined::{
    RegistrationLoss, RegistrationLossConfig, RegularizationType, SimilarityMetric,
};
pub use grad::{GradLoss, GradientPenalty};
pub use lncc::LocalNCCLoss;
pub use ncc::GlobalNCCLoss;
