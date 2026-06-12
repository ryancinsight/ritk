//! Image statistics — re-exported from the [`ritk_statistics`] crate.
//!
//! The concrete types and functions ([`ImageStatistics`], [`Histogram`],
//! [`compute_statistics`], [`dice_coefficient`], [`hausdorff_distance`],
//! [`psnr`], [`ssim`], normalization utilities, information-theoretic metrics,
//! and noise estimators) live in `ritk_statistics`.
//!
//! This module is a thin compatibility shim so existing
//! `ritk_core::statistics::*` import paths continue to resolve.

pub use ritk_statistics::*;
