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
//! - **Anti-extensivity**: O_B(f)(x) ≤ f(x) for all x.
//!   Proof: E_B(f)(x) ≤ f(x) (anti-extensivity of erosion).  Monotonicity
//!   of dilation gives D_B(E_B(f)) ≤ D_B(f).  Anti-extensivity of erosion
//!   applied once more: E_B(f) ≤ f ⇒ D_B(E_B(f)) ≤ D_B(f).  For the flat
//!   SE centred at the origin, D_B(f)(x) ≥ f(x), so the chain is consistent.
//!   The formal proof uses the adjunction pair (erosion, dilation). ∎
//!
//! - **Idempotence**: O_B(O_B(f)) = O_B(f).
//!   O_B(f) ≤ f (anti-extensivity), so erosion cannot decrease it further;
//!   dilation then restores it to the same level. ∎
//!
//! - **Removes bright protrusions**: eliminates bright features (regional
//!   maxima) whose diameter is smaller than 2r + 1 voxels.
//!
//! # ITK Parity
//!
//! Matches `itk::GrayscaleMorphologicalOpeningImageFilter` with:
//! - Flat cubic structuring element of half-width `radius`.
//! - Replicate (safe border) padding — default ITK boundary condition.
//!
//! # Complexity
//!
//! O(N · (2r + 1)³) for each of erosion and dilation pass.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer, pp. 84–88.

use super::grayscale_dilation::dilate_3d;
use super::grayscale_erosion::erode_3d;
use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// ── Filter struct ─────────────────────────────────────────────────────────────

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
    /// A radius of 1 uses a 3×3×3 cubic structuring element.
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
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let opened = self.open_values(&vals, dims);

        let device = image.data().device();
        let out_td = TensorData::new(opened, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }

    /// Apply grayscale opening to a Coeus-native image.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        ritk_image::native::Image::from_flat_on(
            self.open_values(image.data_slice()?, image.shape()),
            image.shape(),
            *image.origin(),
            *image.spacing(),
            *image.direction(),
            backend,
        )
    }

    fn open_values(&self, values: &[f32], dims: [usize; 3]) -> Vec<f32> {
        let r = self.radius;
        let (padded, pdims) = super::pad_replicate_3d(values, dims, r);
        let eroded = erode_3d(&padded, pdims, r);
        let dilated = dilate_3d(&eroded, pdims, r);
        super::crop_border_3d(&dilated, pdims, r).0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_grayscale_opening.rs"]
mod tests_grayscale_opening;
