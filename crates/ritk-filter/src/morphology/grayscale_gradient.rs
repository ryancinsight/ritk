//! Grayscale morphological gradient filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! The morphological gradient (Beucher gradient) is defined as the difference
//! between grayscale dilation and grayscale erosion:
//!
//!   `grad_B(f)(x) = D_B(f)(x) âˆ’ E_B(f)(x)`
//!
//! where `B` is a flat cubic structuring element of half-width `r`:
//!
//!   `B = { b âˆˆ â„¤Â³ : |b_i| â‰¤ r  for i âˆˆ {0, 1, 2} }`
//!
//! and:
//!   - `D_B(f)(x) = max_{b âˆˆ B} f(x + b)` (dilation â€” replicate padding)
//!   - `E_B(f)(x) = min_{b âˆˆ B} f(x + b)` (erosion â€” replicate padding)
//!
//! Since `D_B(f)(x) â‰¥ f(x) â‰¥ E_B(f)(x)` for all x (extensivity and
//! anti-extensivity with flat origin-containing SE), the gradient is
//! non-negative everywhere:
//!
//!   `grad_B(f)(x) â‰¥ 0`
//!
//! # Properties
//!
//! - **Non-negativity**: `grad_B(f)(x) â‰¥ 0`.
//! - **Constant field**: `grad_B(c) = 0` for any constant c.
//! - **Edge detection**: `grad_B(f)` is large near morphological boundaries
//!   and small in smooth regions.
//! - **radius = 0**: SE = {0}, so `D_B = E_B = identity` âŸ¹ gradient = 0 everywhere.
//!
//! # Complexity
//!
//! O(N Â· (2r+1)Â³) where N = n_z Â· n_y Â· n_x is the total voxel count.
//!
//! # References
//!
//! - Beucher, S. & LantuÃ©joul, C. (1979). Use of watersheds in contour detection.
//!   In *International Workshop on Image Processing*.
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer, Â§4.3.
//! - ITK `itk::GrayscaleMorphologicalGradientImageFilter`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

use super::grayscale_dilation::dilate_3d;
use super::grayscale_erosion::erode_3d;

// â”€â”€ Filter struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Grayscale morphological gradient filter for 3-D images.
///
/// Computes `D_B(f) âˆ’ E_B(f)` â€” the Beucher gradient â€” which highlights
/// morphological edges in a grayscale volume.
///
/// The output is non-negative everywhere and is zero on regions where the
/// image is locally constant within the structuring element's footprint.
///
/// # Example (conceptual)
/// A sharp step edge (0 â†’ 10) with `radius = 1` produces gradient = 10 at the
/// boundary voxel (dilation = 10, erosion = 0) and gradient = 0 one voxel away
/// from the edge (dilation = erosion = same constant).
#[derive(Debug, Clone)]
pub struct GrayscaleMorphologicalGradientFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleMorphologicalGradientFilter {
    /// Create a new gradient filter with the given structuring element radius.
    ///
    /// `radius = 0` âŸ¹ structuring element = {0} âŸ¹ gradient = 0 everywhere
    /// (degenerate identity case).
    /// `radius = 1` âŸ¹ 3Ã—3Ã—3 cubic SE (standard morphological gradient).
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Set the structuring element radius and return the modified filter.
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }

    /// Apply the morphological gradient to a 3-D image.
    ///
    /// Returns a new image with the same shape and spatial metadata as the
    /// input. All output voxel values are â‰¥ 0.
    ///
    /// # Errors
    /// Returns an error if the image data cannot be converted to `f32` (only
    /// possible with non-f32 backends).
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let gradient = gradient_vec(&vals, dims, self.radius);
        Ok(rebuild(gradient, dims, image))
    }

    /// Coeus-native sister of [`GrayscaleMorphologicalGradientFilter::apply`].
    ///
    /// Runs the identical Beucher gradient (`dilate âˆ’ erode` over a cubic SE)
    /// via the shared `gradient_vec` host core on the image's contiguous host
    /// buffer, so the result is bitwise-identical to the Burn path. No Burn
    /// tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt image fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
        backend: &B,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let radius = self.radius;
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            gradient_vec(vals, dims, radius)
        })
    }
}

/// Substrate-agnostic host core for [`GrayscaleMorphologicalGradientFilter`].
///
/// `grad(x) = dilate_B(f)(x) âˆ’ erode_B(f)(x)` over the flat cubic SE of the
/// given `radius` (replicate padding). Non-negative by extensivity /
/// anti-extensivity.
pub(crate) fn gradient_vec(vals: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    let dilated = dilate_3d(vals, dims, radius);
    let eroded = erode_3d(vals, dims, radius);
    dilated
        .into_iter()
        .zip(eroded)
        .map(|(d, e)| d - e)
        .collect()
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_grayscale_gradient.rs"]
mod tests_grayscale_gradient;
