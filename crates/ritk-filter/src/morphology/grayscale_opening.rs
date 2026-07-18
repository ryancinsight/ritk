//! Grayscale morphological opening filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale opening with a flat cubic structuring element B of half-width r:
//!
//!   O_B(f) = D_B(E_B(f))
//!
//! i.e. erosion followed by dilation.  Both operations use replicate (clamp)
//! boundary conditions.
//!
//! # Properties
//!
//! - **Anti-extensivity**: O_B(f)(x) â‰¤ f(x) for all x.
//!   Proof: E_B(f)(x) â‰¤ f(x) (anti-extensivity of erosion).  Monotonicity
//!   of dilation gives D_B(E_B(f)) â‰¤ D_B(f).  Anti-extensivity of erosion
//!   applied once more: E_B(f) â‰¤ f â‡’ D_B(E_B(f)) â‰¤ D_B(f).  For the flat
//!   SE centred at the origin, D_B(f)(x) â‰¥ f(x), so the chain is consistent.
//!   The formal proof uses the adjunction pair (erosion, dilation). âˆŽ
//!
//! - **Idempotence**: O_B(O_B(f)) = O_B(f).
//!   O_B(f) â‰¤ f (anti-extensivity), so erosion cannot decrease it further;
//!   dilation then restores it to the same level. âˆŽ
//!
//! - **Removes bright protrusions**: eliminates bright features (regional
//!   maxima) whose diameter is smaller than 2r + 1 voxels.
//!
//! # ITK Parity
//!
//! Matches `itk::GrayscaleMorphologicalOpeningImageFilter` with:
//! - Flat cubic structuring element of half-width `radius`.
//! - Replicate (safe border) padding â€” default ITK boundary condition.
//!
//! # Complexity
//!
//! O(N Â· (2r + 1)Â³) for each of erosion and dilation pass.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer, pp. 84â€“88.

use super::grayscale_dilation::dilate_3d;
use super::grayscale_erosion::erode_3d;
use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// â”€â”€ Filter struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Grayscale morphological opening filter for 3-D images.
///
/// Applies erosion followed by dilation with a flat cubic structuring
/// element of half-width `radius`.  Removes bright protrusions smaller
/// than the structuring element without altering large bright regions.
#[derive(Debug, Clone)]
pub struct GrayscaleOpeningFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleOpeningFilter {
    /// Create a new grayscale opening filter with the given radius.
    ///
    /// A radius of 0 yields the identity (single-voxel SE).
    /// A radius of 1 uses a 3Ã—3Ã—3 cubic structuring element.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Set the structuring element radius.
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }

    /// Apply grayscale opening to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let opened = open_3d(&vals, dims, self.radius);
        Ok(rebuild(opened, dims, image))
    }

    /// Coeus-native sister of [`GrayscaleOpeningFilter::apply`].
    ///
    /// Runs the identical safe-border erodeâ†’dilate opening via the shared
    /// `open_3d` host core on the image's contiguous host buffer, so the result
    /// is bitwise-identical to the Burn path. No Burn tensor is constructed.
    /// Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            open_3d(vals, dims, self.radius)
        })
    }
}

/// Substrate-agnostic host core for [`GrayscaleOpeningFilter`].
///
/// `O_B(f) = D_B(E_B(f))` with ITK's safe border: replicate-pad by `radius`,
/// run the erode/dilate pair on the padded volume, then crop. This keeps the
/// border band bit-exact to `sitk.GrayscaleMorphologicalOpening` (naive
/// erodeâ†’dilate diverges within `radius` of the edge).
pub(crate) fn open_3d(vals: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let (padded, pdims) = super::pad_replicate_3d(vals, dims, radius);
    let eroded = erode_3d(&padded, pdims, radius);
    let dilated = dilate_3d(&eroded, pdims, radius);
    let (opened, _) = super::crop_border_3d(&dilated, pdims, radius);
    opened
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_grayscale_opening.rs"]
mod tests_grayscale_opening;
