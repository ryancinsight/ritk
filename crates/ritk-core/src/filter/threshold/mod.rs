//! Threshold-selection algorithms (ITK / SimpleITK / ImageJ parity).
//!
//! Re-exports all threshold-selection strategies from `crate::segmentation::threshold`,
//! making them accessible under the `filter::` path alongside all other ritk-core filters.
//!
//! | Type | ITK equivalent | Method |
//! |------|---------------|--------|
//! | [`MultiOtsuThreshold`] | `OtsuMultipleThresholdsImageFilter` | Maximise between-class variance, K classes |
//! | [`OtsuThreshold`] | `OtsuThresholdImageFilter` | Standard 2-class Otsu (K=2 degenerate case) |
//! | [`LiThreshold`] | `LiThresholdImageFilter` | Minimum cross-entropy iterative method |
//! | [`YenThreshold`] | `YenThresholdImageFilter` | Maximum correlation (Yen 1995) |
//! | [`KapurThreshold`] | `KapurThresholdImageFilter` | Maximum entropy (Kapur 1985) |
//! | [`TriangleThreshold`] | `TriangleThresholdImageFilter` | Geometric triangle / Zack 1977 |
//! | [`BinaryThreshold`] | — | User-specified intensity band |

pub use crate::segmentation::threshold::{
    apply_binary_threshold_to_slice,
    binary_threshold,
    BinaryThreshold,
    compute_kapur_threshold_from_slice,
    compute_li_threshold_from_slice,
    compute_multi_otsu_thresholds_from_slice,
    compute_triangle_threshold_from_slice,
    compute_yen_threshold_from_slice,
    kapur_threshold,
    KapurThreshold,
    li_threshold,
    LiThreshold,
    multi_otsu_threshold,
    MultiOtsuThreshold,
    otsu_threshold,
    OtsuThreshold,
    triangle_threshold,
    TriangleThreshold,
    yen_threshold,
    YenThreshold,
};
