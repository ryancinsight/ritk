//! Metric implementations.
//!
//! This module contains various similarity metrics used for image registration.

pub mod autodiff;
pub(crate) mod cache_slot;
pub mod correlation_ratio;
pub mod dl_losses;
pub mod entropy;
pub mod histogram;
pub mod lncc;
pub mod mse;
pub mod mutual_information;
pub mod ncc;
pub mod ngf;
pub mod sampling;
pub mod trait_;

pub use correlation_ratio::{CorrelationDirection, CorrelationRatio};
pub use entropy::{entropy, entropy_with_eps, DEFAULT_ENTROPY_EPS};
pub use lncc::LocalNormalizedCrossCorrelation;
pub use mse::MeanSquaredError;
pub use mutual_information::{MutualInformation, MutualInformationVariant, NormalizationMethod};
pub use ncc::NormalizedCrossCorrelation;
pub use ngf::NormalizedGradientField;
pub use sampling::{resolve_n_points, SamplingConfig, SamplingMode};
pub use trait_::Metric;
