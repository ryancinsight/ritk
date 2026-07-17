//! Intensity clamp filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! output(x) = clamp(I(x), lower, upper)
//!           = max(lower, min(upper, I(x)))
//!
//! This is the pointwise projection of I onto the interval [lower, upper].
//!
//! ## Proof of idempotence
//!
//! Applying the clamp twice yields identical output because every voxel
//! already lies in [lower, upper] after the first application:
//!
//!   clamp(clamp(v, lo, hi), lo, hi) = clamp(v, lo, hi)
//!
//! ## Invariants
//!
//! - All output values satisfy `lower â‰¤ out(x) â‰¤ upper`.
//! - If all input values already lie in `[lower, upper]`, the output equals
//!   the input exactly.
//! - `lower > upper` is a logic error; the constructor panics.
//!
//! # ITK / SimpleITK Parity
//!
//! Corresponds to `itk::ClampImageFilter<TInputImage, TOutputImage>` with
//! `SetBounds(lower, upper)`.
//!
//! # Complexity
//!
//! O(N) time, O(N) space (output allocation), O(1) auxiliary.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// â”€â”€ Filter struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Clamp image intensity to the closed interval `[lower, upper]`.
///
/// Every voxel whose value lies below `lower` is raised to `lower`; every
/// voxel whose value exceeds `upper` is lowered to `upper`; all other voxels
/// are preserved unchanged.
///
/// # Invariants
///
/// - `lower â‰¤ upper` (enforced by constructor panic).
/// - All output values lie in `[lower, upper]`.
#[derive(Debug, Clone)]
pub struct ClampImageFilter {
    /// Inclusive lower bound for output intensity.
    pub lower: f32,
    /// Inclusive upper bound for output intensity.
    pub upper: f32,
}

impl ClampImageFilter {
    /// Create a `ClampImageFilter` with the given bounds.
    ///
    /// # Panics
    ///
    /// Panics if `lower > upper`.
    pub fn new(lower: f32, upper: f32) -> Self {
        assert!(
            lower <= upper,
            "ClampImageFilter: lower bound {lower} must be â‰¤ upper bound {upper}"
        );
        Self { lower, upper }
    }

    /// Apply the clamp to `image`.
    ///
    /// Returns a new image with identical shape and spatial metadata.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let out = clamp_vec(&vals, self.lower, self.upper);
        Ok(rebuild(out, dims, image))
    }

    /// Coeus-native sister of [`ClampImageFilter::apply`].
    ///
    /// Runs the identical pointwise clamp via the shared `clamp_vec` host core
    /// on the image's contiguous host buffer, so the result is bitwise-identical
    /// to the Burn path. No Burn tensor is constructed. Spatial metadata
    /// (origin, spacing, direction) is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B::default()) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, _dims| {
            clamp_vec(vals, self.lower, self.upper)
        })
    }
}

/// Substrate-agnostic host core for [`ClampImageFilter`].
///
/// Projects every voxel onto the closed interval `[lower, upper]` via
/// `f32::clamp` (NaN-propagating, matching ITK). Requires `lower <= upper`
/// (guaranteed by the constructor).
pub(crate) fn clamp_vec(vals: &[f32], lower: f32, upper: f32) -> Vec<f32> {
    vals.iter().map(|&v| v.clamp(lower, upper)).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_clamp.rs"]
mod tests;
