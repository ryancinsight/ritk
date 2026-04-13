//! Intensity normalization algorithms.
//!
//! Provides three normalization strategies:
//! - [`ZScoreNormalizer`]: standardizes to zero mean, unit variance.
//! - [`MinMaxNormalizer`]: rescales intensities to \[0, 1\] or an arbitrary range.
//! - [`HistogramMatcher`]: matches the intensity histogram of a source image to a
//!   reference image via CDF-based lookup.

pub mod histogram_matching;
pub mod minmax;
pub mod zscore;

pub use histogram_matching::HistogramMatcher;
pub use minmax::MinMaxNormalizer;
pub use zscore::ZScoreNormalizer;
