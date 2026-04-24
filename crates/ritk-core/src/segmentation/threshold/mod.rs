//! Intensity thresholding algorithms.
//!
//! Provides threshold-selection strategies:
//! - [`OtsuThreshold`]: single-threshold Otsu method (maximises between-class variance).
//! - [`MultiOtsuThreshold`]: multi-class extension (K classes, K-1 thresholds).
//! - [`LiThreshold`]: Li's minimum cross-entropy iterative thresholding.
//! - [`YenThreshold`]: Yen's maximum correlation thresholding.
//! - [`KapurThreshold`]: Kapur's maximum entropy thresholding.
//! - [`TriangleThreshold`]: Triangle (Zack) geometric thresholding.

pub mod kapur;
pub mod li;
pub mod multi_otsu;
pub mod otsu;
pub mod triangle;
pub mod yen;

pub use kapur::{kapur_threshold, KapurThreshold};
pub use li::{li_threshold, LiThreshold};
pub use multi_otsu::{multi_otsu_threshold, MultiOtsuThreshold};
pub use otsu::{otsu_threshold, OtsuThreshold};
pub use triangle::{triangle_threshold, TriangleThreshold};
pub use yen::{yen_threshold, YenThreshold};
pub use kapur::compute_kapur_threshold_from_slice;
pub use li::compute_li_threshold_from_slice;
pub use multi_otsu::compute_multi_otsu_thresholds_from_slice;
pub use triangle::compute_triangle_threshold_from_slice;
pub use yen::compute_yen_threshold_from_slice;

