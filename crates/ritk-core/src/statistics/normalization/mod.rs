//! Intensity normalization algorithms.
//!
//! Provides four normalization strategies:
//! - [`ZScoreNormalizer`]: standardizes to zero mean, unit variance.
//! - [`MinMaxNormalizer`]: rescales intensities to \[0, 1\] or an arbitrary range.
//! - [`HistogramMatcher`]: matches the intensity histogram of a source image to a
//!   reference image via CDF-based lookup.
//! - [`NyulUdupaNormalizer`]: Nyúl-Udupa piecewise-linear histogram standardization
//!   via learned landmark percentiles.

pub mod histogram_matching;
pub mod minmax;
pub mod nyul_udupa;
pub mod zscore;

pub use histogram_matching::HistogramMatcher;
pub use minmax::MinMaxNormalizer;
pub use nyul_udupa::NyulUdupaNormalizer;
pub use zscore::ZScoreNormalizer;
