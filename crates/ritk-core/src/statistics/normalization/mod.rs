//! Intensity normalization algorithms.
//!
//! Provides five normalization strategies:
//! - [`ZScoreNormalizer`]: standardizes to zero mean, unit variance.
//! - [`MinMaxNormalizer`]: rescales intensities to \[0, 1\] or an arbitrary range.
//! - [`HistogramMatcher`]: matches the intensity histogram of a source image to a
//!   reference image via CDF-based lookup.
//! - [`NyulUdupaNormalizer`]: Nyúl-Udupa piecewise-linear histogram standardization
//!   via learned landmark percentiles.
//! - [`WhiteStripeNormalizer`]: white stripe normalization for brain MRI
//!   (Shinohara et al. 2014).

pub mod histogram_matching;
pub mod minmax;
pub mod nyul_udupa;
pub mod white_stripe;
pub mod zscore;

pub use histogram_matching::HistogramMatcher;
pub use minmax::MinMaxNormalizer;
pub use nyul_udupa::NyulUdupaNormalizer;
pub use white_stripe::{MriContrast, WhiteStripeConfig, WhiteStripeNormalizer, WhiteStripeResult};
pub use zscore::ZScoreNormalizer;
