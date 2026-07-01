//! Metric implementations.
//!
//! This module contains various similarity metrics used for image registration.

pub(crate) mod cache_slot;
#[cfg(feature = "coeus")]
pub mod coeus_autograd;
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

#[cfg(feature = "coeus")]
pub use coeus_autograd::{
    affine_mse_coeus, affine_transform_coeus, mean_squared_error_coeus, sample_linear_1d_coeus,
    sample_trilinear_coeus, sgd_step_var, translate_axis_coeus, translation_mse_coeus,
};
pub use correlation_ratio::{CorrelationDirection, CorrelationRatio};
pub use entropy::{entropy, entropy_with_eps, DEFAULT_ENTROPY_EPS};
pub use lncc::LocalNormalizedCrossCorrelation;
pub use mse::MeanSquaredError;
pub use mutual_information::{MutualInformation, MutualInformationVariant, NormalizationMethod};
pub use ncc::NormalizedCrossCorrelation;
pub use ngf::NormalizedGradientField;
pub use sampling::{resolve_n_points, SamplingConfig, SamplingMode};
pub use trait_::Metric;
