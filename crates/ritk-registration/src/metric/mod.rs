//! Metric implementations.
//!
//! This module contains various similarity metrics used for image registration.

pub mod trait_;
pub mod mse;
pub mod mutual_information;
pub mod ncc;
pub mod lncc;
pub mod advanced_correlation_ratio;
pub mod advanced_mutual_information;

pub use trait_::Metric;
pub use mse::MeanSquaredError;
pub use mutual_information::MutualInformation;
pub use ncc::NormalizedCrossCorrelation;
pub use lncc::LocalNormalizedCrossCorrelation;
pub use advanced_correlation_ratio::{AdvancedCorrelationRatio, CorrelationDirection};
pub use advanced_mutual_information::AdvancedMutualInformation;
