//! Intensity thresholding algorithms.
//!
//! Provides two threshold-selection strategies:
//! - [`OtsuThreshold`]: single-threshold Otsu method (maximises between-class variance).
//! - [`MultiOtsuThreshold`]: multi-class extension (K classes, K-1 thresholds).

pub mod multi_otsu;
pub mod otsu;

pub use multi_otsu::{multi_otsu_threshold, MultiOtsuThreshold};
pub use otsu::{otsu_threshold, OtsuThreshold};
