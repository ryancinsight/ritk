//! Metric implementations.
//!
//! This module contains various similarity metrics used for image registration.

pub mod trait_;
pub mod histogram;
pub mod mse;
pub mod mutual_information;
pub mod ncc;
pub mod lncc;
pub mod correlation_ratio;
pub mod normalized_mutual_information;

pub use trait_::Metric;
pub use mse::MeanSquaredError;
pub use mutual_information::MutualInformation;
pub use ncc::NormalizedCrossCorrelation;
pub use lncc::LocalNormalizedCrossCorrelation;
pub use correlation_ratio::{CorrelationRatio, CorrelationDirection};
pub use normalized_mutual_information::{NormalizedMutualInformation, NormalizationMethod};
