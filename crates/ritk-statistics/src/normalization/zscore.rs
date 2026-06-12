//! Z-score intensity normalization.
//!
//! # Mathematical Specification
//! Given an image X with population mean μ and population standard deviation σ:
//!
//!   Z(x) = (x − μ) / (σ + ε),   ε = 1e-8
//!
//! After normalization: E\[Z\] ≈ 0, Var\[Z\] ≈ 1.
//! The ε term prevents division by zero on constant images.
//!
//! # Invariants
//! - Output image carries the same spatial metadata (origin, spacing, direction)
//!   as the input.
//! - μ and σ are computed from the full image population (not a sample).

use crate::image_statistics::{compute_statistics, masked_statistics};
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

/// Z-score normalizer.
///
/// Transforms image intensities to zero mean and unit variance using
/// population statistics derived from the image itself.
pub struct ZScoreNormalizer;

impl ZScoreNormalizer {
    /// Create a new `ZScoreNormalizer`.
    pub fn new() -> Self {
        Self
    }

    /// Normalize `image` to zero mean, unit variance.
    ///
    /// # Formula
    /// `output = (input − mean) / (std + 1e-8)`
    ///
    /// Spatial metadata is preserved exactly.
    pub fn normalize<B: Backend, const D: usize>(&self, image: &Image<B, D>) -> Image<B, D> {
        let stats = compute_statistics(image);
        let mean = stats.mean;
        let std = stats.std;
        let denom = std + super::NORMALIZER_EPSILON;

        let normalized = image.data().clone().sub_scalar(mean).div_scalar(denom);

        Image::new(
            normalized,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }

    /// Normalize `image` to zero mean, unit variance using statistics derived
    /// from masked foreground voxels only.
    ///
    /// # Formula
    /// `output = (input − μ_mask) / (σ_mask + 1e-8)`
    ///
    /// μ_mask and σ_mask are population statistics computed from voxels where
    /// `mask > 0.5`. If the mask contains no foreground voxels the method falls
    /// back to full-image population statistics (identical to \[`normalize`\]).
    ///
    /// All voxels — including background voxels — are transformed using the
    /// same μ_mask and σ_mask parameters. Spatial metadata is preserved exactly.
    ///
    /// # Arguments
    /// * `image` — Input image.
    /// * `mask`  — Binary mask (foreground > 0.5). Must have the same element
    ///   count as `image`; a shape mismatch propagates as a panic from
    ///   [`masked_statistics`].
    pub fn normalize_masked<B: Backend, const D: usize>(
        &self,
        image: &Image<B, D>,
        mask: &Image<B, D>,
    ) -> Image<B, D> {
        // Extract mask slice once to check for foreground before calling
        // masked_statistics, which panics on an empty foreground set.
        let (mask_vals, _) = extract_vec_infallible(mask);
        let mask_slice: &[f32] = &mask_vals;
        let has_foreground = mask_slice.iter().any(|&m| m > crate::FOREGROUND_THRESHOLD);

        let stats = if has_foreground {
            masked_statistics(image, mask)
        } else {
            compute_statistics(image)
        };

        let mean = stats.mean;
        let std = stats.std;
        let denom = std + super::NORMALIZER_EPSILON;

        let normalized = image.data().clone().sub_scalar(mean).div_scalar(denom);

        Image::new(
            normalized,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }
}

impl Default for ZScoreNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests_zscore.rs"]
mod tests;
