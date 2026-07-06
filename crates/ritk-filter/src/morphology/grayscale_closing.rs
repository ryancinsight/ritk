//! Grayscale morphological closing filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale closing with a flat cubic structuring element B of half-width r:
//!
//!   C_B(f) = E_B(D_B(f))
//!
//! i.e. dilation followed by erosion.  Both operations use replicate (clamp)
//! boundary conditions.
//!
//! # Properties
//!
//! - **Extensivity**: C_B(f)(x) ≥ f(x) for all x.
//!   Proof: D_B(f)(x) ≥ f(x) (extensivity of dilation).  Then, since
//!   E_B(D_B(f)) ≥ E_B(f) (monotonicity of erosion) and E_B(D_B(f)) is at
//!   least as large as f pointwise because the dilation first raised the
//!   minimum of each neighbourhood, erosion cannot reduce it below f(x).
//!
//! - **Idempotence**: C_B(C_B(f)) = C_B(f).
//!   C_B is already extensive (C_B(f) ≥ f), so dilation cannot raise it
//!   further; erosion then restores it to the same level. ∎
//!
//! - **Fills dark holes**: removes dark features (regional minima) whose
//!   diameter is smaller than 2r + 1 voxels.
//!
//! # ITK Parity
//!
//! Matches `itk::GrayscaleMorphologicalClosingImageFilter` with:
//! - Flat cubic structuring element of half-width `radius`.
//! - Safe border mode (replicate padding) — default ITK boundary condition.
//!
//! # Complexity
//!
//! O(N · (2r + 1)³) for each of dilation and erosion pass.
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

/// Grayscale morphological closing filter for 3-D images.
///
/// Applies dilation followed by erosion with a flat cubic structuring
/// element of half-width `radius`.  Fills dark voids smaller than the
/// structuring element without altering large dark regions.
#[derive(Debug, Clone)]
pub struct GrayscaleClosingFilter {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleClosingFilter {
    /// Create a new grayscale closing filter with the given radius.
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

    /// Apply grayscale closing to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction).
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        // C_B(f) = E_B(D_B(f)) with ITK's safe border: replicate-pad by `radius`,
        // run the dilate/erode pair on the padded volume, then crop. Keeps the
        // border band bit-exact to sitk.GrayscaleMorphologicalClosing.
        let r = self.radius;
        let (padded, pdims) = super::pad_replicate_3d(&vals, dims, r);
        let dilated = dilate_3d(&padded, pdims, r);
        let eroded = erode_3d(&dilated, pdims, r);
        let (closed, _) = super::crop_border_3d(&eroded, pdims, r);

        let device = image.data().device();
        let out_td = TensorData::new(closed, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
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
#[path = "tests_grayscale_closing.rs"]
mod tests_grayscale_closing;
