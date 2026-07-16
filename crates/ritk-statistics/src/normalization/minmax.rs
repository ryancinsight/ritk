//! Min-max intensity normalization.
//!
//! # Mathematical Specification
//! Given an image X with intensity range [xₘᵢₙ, xₘₐₓ]:
//!
//!   N(x) = (x − xₘᵢₙ) / (xₘₐₓ − xₘᵢₙ + ε),   ε = 1e-8
//!
//! This maps intensities to [0, 1].  An optional affine remap then applies:
//!
//!   R(x) = N(x) · range.span() + range.min()
//!
//! which maps to `[range.min(), range.max()]`.
//!
//! # Invariants
//! - Output image carries the same spatial metadata (origin, spacing, direction)
//!   as the input.
//! - ε prevents division by zero on constant images.
//! - Default target range is [0.0, 1.0].

use super::intensity_range::IntensityRange;
use crate::image_statistics::compute_statistics;
use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::native::Image as NativeImage;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::native as tensor_ops;

/// Min-max intensity normalizer.
///
/// Rescales image intensities to [0, 1] (default) or an arbitrary range
/// encoded as an [`IntensityRange<f32>`].
pub struct MinMaxNormalizer {
    /// Target output intensity range `[min, max]`.
    pub range: IntensityRange<f32>,
}

impl MinMaxNormalizer {
    /// Create a normalizer that maps intensities to [0, 1].
    pub fn new() -> Self {
        Self {
            range: IntensityRange::new_unchecked(0.0_f32, 1.0_f32),
        }
    }

    /// Create a normalizer that maps intensities to `[target_min, target_max]`.
    ///
    /// # Panics
    /// Panics if `target_min > target_max` (invariant: must satisfy `target_min ≤ target_max`).
    pub fn with_range(target_min: f32, target_max: f32) -> Self {
        Self {
            range: IntensityRange::new(target_min, target_max)
                .expect("invariant: target_min <= target_max"),
        }
    }

    /// Normalize `image`.
    ///
    /// # Formula
    /// ```text
    /// normalized = (input − min) / (max − min + 1e-8)
    /// output     = normalized · range.span() + range.min()
    /// ```
    ///
    /// Spatial metadata is preserved exactly.
    pub fn normalize<B: Backend, const D: usize>(&self, image: &Image<f32, B, D>) -> Image<f32, B, D> {
        let stats = compute_statistics(image);
        let min = stats.min;
        let max = stats.max;
        let input_range = (max - min) + super::NORMALIZER_EPSILON;

        // N(x) = (x − min) / (max − min + ε)
        let normalized = image.data().clone().sub_scalar(min).div_scalar(input_range);

        // R(x) = N(x) · range.span() + range.min()
        let output_span = self.range.span();
        let remapped = if (output_span - 1.0).abs() < super::UNIT_RANGE_EPSILON
            && self.range.min().abs() < super::UNIT_RANGE_EPSILON
        {
            // Default [0,1] case: skip the remap arithmetic entirely.
            normalized
        } else {
            normalized
                .mul_scalar(output_span)
                .add_scalar(self.range.min())
        };

        Image::new(
            remapped,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        )
    }
}

impl MinMaxNormalizer {
    /// Apply the min-max map `N(x) = (x − min)/(max − min + ε)` then the optional
    /// affine remap `R(x) = N(x)·span + range.min()` to a host buffer.
    ///
    /// Single host realization shared by [`MinMaxNormalizer::normalize_native`];
    /// the Burn path expresses the identical arithmetic through on-device tensor
    /// scalar ops.
    fn remap_values(&self, values: &mut [f32], min: f32, max: f32) {
        let input_range = (max - min) + super::NORMALIZER_EPSILON;
        let output_span = self.range.span();
        let output_min = self.range.min();
        let is_unit_range = (output_span - 1.0).abs() < super::UNIT_RANGE_EPSILON
            && output_min.abs() < super::UNIT_RANGE_EPSILON;

        for v in values.iter_mut() {
            let normalized = (*v - min) / input_range;
            *v = if is_unit_range {
                normalized
            } else {
                normalized * output_span + output_min
            };
        }
    }

    /// Normalize a Coeus-backed `image` to the configured range.
    ///
    /// Coeus-native sister of [`MinMaxNormalizer::normalize`]; identical formula
    /// and spatial-metadata preservation.
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
        self.remap_values(&mut values, stats.min, stats.max);
        tensor_ops::rebuild_image(values, dims, image, &B::default())
    }
}

impl Default for MinMaxNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests_minmax.rs"]
mod tests;
