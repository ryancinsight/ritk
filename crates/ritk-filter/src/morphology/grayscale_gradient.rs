//! Grayscale morphological gradient filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! The morphological gradient (Beucher gradient) is defined as the difference
//! between grayscale dilation and grayscale erosion:
//!
//!   `grad_B(f)(x) = D_B(f)(x) − E_B(f)(x)`
//!
//! where `B` is a flat cubic structuring element of half-width `r`:
//!
//!   `B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }`
//!
//! and:
//!   - `D_B(f)(x) = max_{b ∈ B} f(x + b)` (dilation — replicate padding)
//!   - `E_B(f)(x) = min_{b ∈ B} f(x + b)` (erosion — replicate padding)
//!
//! Since `D_B(f)(x) ≥ f(x) ≥ E_B(f)(x)` for all x (extensivity and
//! anti-extensivity with flat origin-containing SE), the gradient is
//! non-negative everywhere:
//!
//!   `grad_B(f)(x) ≥ 0`
//!
//! # Properties
//!
//! - **Non-negativity**: `grad_B(f)(x) ≥ 0`.
//! - **Constant field**: `grad_B(c) = 0` for any constant c.
//! - **Edge detection**: `grad_B(f)` is large near morphological boundaries
//!   and small in smooth regions.
//! - **radius = 0**: SE = {0}, so `D_B = E_B = identity` ⟹ gradient = 0 everywhere.
//!
//! # Complexity
//!
//! O(N · (2r+1)³) where N = n_z · n_y · n_x is the total voxel count.
//!
//! # References
//!
//! - Beucher, S. & Lantuéjoul, C. (1979). Use of watersheds in contour detection.
//!   In *International Workshop on Image Processing*.
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer, §4.3.
//! - ITK `itk::GrayscaleMorphologicalGradientImageFilter`.

use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

use super::grayscale_dilation::dilate_3d;
use super::grayscale_erosion::erode_3d;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Grayscale morphological gradient filter for 3-D images.
///
/// Computes `D_B(f) − E_B(f)` — the Beucher gradient — which highlights
/// morphological edges in a grayscale volume.
///
/// The output is non-negative everywhere and is zero on regions where the
/// image is locally constant within the structuring element's footprint.
///
/// # Example (conceptual)
/// A sharp step edge (0 → 10) with `radius = 1` produces gradient = 10 at the
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
    /// `radius = 0` ⟹ structuring element = {0} ⟹ gradient = 0 everywhere
    /// (degenerate identity case).
    /// `radius = 1` ⟹ 3×3×3 cubic SE (standard morphological gradient).
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
    /// input. All output voxel values are ≥ 0.
    ///
    /// # Errors
    /// Returns an error if the image data cannot be converted to `f32` (only
    /// possible with non-f32 backends).
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let dilated = dilate_3d(&vals, dims, self.radius);
        let eroded = erode_3d(&vals, dims, self.radius);

        // gradient(x) = dilation(x) - erosion(x); ≥ 0 by extensivity / anti-extensivity.
        let gradient: Vec<f32> = dilated
            .into_iter()
            .zip(eroded)
            .map(|(d, e)| d - e)
            .collect();

        let device = image.data().device();
        let td = TensorData::new(gradient, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_grayscale_gradient.rs"]
mod tests_grayscale_gradient;
