//! Metric implementations.
//!
//! This module contains various similarity metrics used for image registration.

pub mod correlation_ratio;
pub mod histogram;
pub mod lncc;
pub mod mse;
pub mod mutual_information;
pub mod ncc;
pub mod trait_;

pub use correlation_ratio::{CorrelationDirection, CorrelationRatio};
pub use lncc::LocalNormalizedCrossCorrelation;
pub use mse::MeanSquaredError;
pub use mutual_information::{MutualInformation, MutualInformationVariant, NormalizationMethod};
pub use ncc::NormalizedCrossCorrelation;
pub use trait_::Metric;
