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
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::tensor::Backend;
use ritk_image::Image as NativeImage;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;
use ritk_tensor_ops::native as tensor_ops;

/// Apply the z-score transform `(v − mean) / (std + ε)` to a host buffer.
///
/// Single host realization of the z-score formula shared by the Coeus-native
/// paths; the Burn paths express the identical arithmetic through on-device
/// tensor scalar ops so GPU backends stay lazy.
#[inline]
fn zscore_values(values: &mut [f32], mean: f32, std: f32) {
    let denom = std + super::NORMALIZER_EPSILON;
    for v in values.iter_mut() {
        *v = (*v - mean) / denom;
    }
}

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
    pub fn normalize<B: Backend, const D: usize>(
        &self,
        image: &Image<f32, B, D>,
    ) -> Image<f32, B, D>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let stats = compute_statistics(image);
        let (mut values, dims) = extract_vec_infallible(image);
        zscore_values(&mut values, stats.mean, stats.std);
        ritk_tensor_ops::rebuild(values, dims, image)
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
        image: &Image<f32, B, D>,
        mask: &Image<f32, B, D>,
    ) -> Image<f32, B, D>
    where
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
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

        let (mut values, dims) = extract_vec_infallible(image);
        zscore_values(&mut values, stats.mean, stats.std);
        ritk_tensor_ops::rebuild(values, dims, image)
    }
}

/// Coeus-native z-score paths (host-resident `ComputeBackend`).
impl ZScoreNormalizer {
    /// Normalize a Coeus-backed `image` to zero mean, unit variance.
    ///
    /// Coeus-native sister of [`ZScoreNormalizer::normalize`]; identical formula
    /// `output = (input − mean) / (std + 1e-8)` on population statistics, with
    /// spatial metadata preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt tensor fails shape validation.
    pub fn normalize_native<B, const D: usize>(
        &self,
        image: &NativeImage<f32, B, D>,
    ) -> anyhow::Result<NativeImage<f32, B, D>>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let stats = crate::image_statistics::native::compute_statistics(image)?;
        let (mut values, dims) = tensor_ops::extract_image_vec(image)?;
        zscore_values(&mut values, stats.mean, stats.std);
        tensor_ops::rebuild_image(values, dims, image, &B::default())
    }

    /// Normalize a Coeus-backed `image` using statistics from masked foreground
    /// voxels only (`mask > 0.5`); falls back to full-image statistics when the
    /// mask has no foreground.
    ///
    /// Coeus-native sister of [`ZScoreNormalizer::normalize_masked`]. All voxels
    /// are transformed with the same `μ_mask`, `σ_mask`.
    ///
    /// # Errors
    /// Returns an error when either tensor is not host-addressable/contiguous,
    /// the image and mask element counts differ, or rebuild validation fails.
    pub fn normalize_masked_native<B, const D: usize>(
        &self,
        image: &NativeImage<f32, B, D>,
        mask: &NativeImage<f32, B, D>,
    ) -> anyhow::Result<NativeImage<f32, B, D>>
    where
        B: ComputeBackend + Default,
        B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
    {
        let (mask_slice, _) = tensor_ops::extract_image_slice(mask)?;
        let has_foreground = mask_slice.iter().any(|&m| m > crate::FOREGROUND_THRESHOLD);

        let stats = if has_foreground {
            crate::image_statistics::native::masked_statistics(image, mask)?
        } else {
            crate::image_statistics::native::compute_statistics(image)?
        };

        let (mut values, dims) = tensor_ops::extract_image_vec(image)?;
        zscore_values(&mut values, stats.mean, stats.std);
        tensor_ops::rebuild_image(values, dims, image, &B::default())
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
