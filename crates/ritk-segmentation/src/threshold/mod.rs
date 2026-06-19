//! Intensity thresholding algorithms.
//!
//! Provides threshold-selection strategies:
//! - [`BinaryThreshold`]: user-specified intensity band thresholding.
//! - [`OtsuThreshold`]: single-threshold Otsu method (maximises between-class variance).
//! - [`MultiOtsuThreshold`]: multi-class extension (K classes, K-1 thresholds).
//! - [`LiThreshold`]: Li's minimum cross-entropy iterative thresholding.
//! - [`YenThreshold`]: Yen's maximum correlation thresholding.
//! - [`KapurThreshold`]: Kapur's maximum entropy thresholding.
//! - [`TriangleThreshold`]: Triangle (Zack) geometric thresholding.
//!
//! The five auto-selection algorithms share a common scaffold via the sealed
//! [`AutoThreshold`] trait; see [`auto_threshold`] for details.

pub mod auto_threshold;
pub mod binary;
pub mod huang;
pub mod intermodes;
pub mod isodata;
pub mod kapur;
pub mod kittler;
pub mod li;
pub mod moments;
pub mod multi_otsu;
pub mod otsu;
pub mod renyi;
pub mod shanbhag;
pub mod triangle;
pub mod yen;

pub use auto_threshold::AutoThreshold;
pub use binary::{apply_binary_threshold_to_slice, binary_threshold, BinaryThreshold};
pub use huang::{compute_huang_threshold_from_slice, huang_threshold, HuangThreshold};
pub use intermodes::{
    compute_intermodes_threshold_from_slice, intermodes_threshold, IntermodesThreshold,
};
pub use isodata::{compute_isodata_threshold_from_slice, isodata_threshold, IsoDataThreshold};
pub use kapur::compute_kapur_threshold_from_slice;
pub use kapur::{kapur_threshold, KapurThreshold};
pub use kittler::{
    compute_kittler_illingworth_threshold_from_slice, kittler_illingworth_threshold,
    KittlerIllingworthThreshold,
};
pub use li::compute_li_threshold_from_slice;
pub use li::{li_threshold, LiThreshold};
pub use moments::{compute_moments_threshold_from_slice, moments_threshold, MomentsThreshold};
pub use multi_otsu::compute_multi_otsu_thresholds_from_slice;
pub use multi_otsu::{multi_otsu_threshold, MultiOtsuThreshold};
pub use otsu::{otsu_threshold, OtsuThreshold};
pub use renyi::{
    compute_renyi_entropy_threshold_from_slice, renyi_entropy_threshold, RenyiEntropyThreshold,
};
pub use shanbhag::{compute_shanbhag_threshold_from_slice, shanbhag_threshold, ShanbhagThreshold};
pub use triangle::compute_triangle_threshold_from_slice;
pub use triangle::{triangle_threshold, TriangleThreshold};
pub use yen::compute_yen_threshold_from_slice;
pub use yen::{yen_threshold, YenThreshold};

/// Near-zero class probability guard: thresholds with essentially empty classes
/// (probability < PROB_ZERO_GUARD) are skipped to avoid log(0) and division by zero.
pub(super) const PROB_ZERO_GUARD: f64 = 1e-12;
