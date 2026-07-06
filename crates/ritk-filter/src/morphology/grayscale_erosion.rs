//! Grayscale erosion filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale erosion with a flat structuring element B is defined as:
//!
//!   (E_B f)(x) = min_{b ∈ B} f(x + b)
//!
//! where B is a cubic neighbourhood of half-width `radius`:
//!
//!   B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }
//!
//! giving |B| = (2r + 1)³ voxels per neighbourhood.
//!
//! # Boundary Handling
//!
//! Replicate (clamp) padding: out-of-bounds indices are clamped to the nearest
//! valid index along each axis, equivalent to extending the boundary voxels
//! infinitely outward.
//!
//! # Properties
//!
//! - **Idempotence on constant fields**: E_B(c) = c for all constants c.
//! - **Anti-extensivity**: (E_B f)(x) ≤ f(x) for all x (with flat B containing
//!   the origin).
//! - **Translation invariance**: E_B(f(· − t))(x) = (E_B f)(x − t).
//! - **Increasing**: f ≤ g ⇒ E_B f ≤ E_B g.
//! - **Duality with dilation**: E_B f = −(D_B(−f)) for flat structuring elements.
//!
//! # Complexity
//!
//! O(N · (2r + 1)³) where N = n_z · n_y · n_x is the total voxel count.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use ritk_image::tensor::Backend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::Image;
use ritk_tensor_ops::extract_vec;

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Grayscale erosion filter for 3-D images.
///
/// Replaces each voxel with the minimum value in its `(2r+1)³` cubic
/// neighbourhood. Out-of-bounds positions use replicate (clamp) padding.
#[derive(Debug, Clone)]
pub struct GrayscaleErosion {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleErosion {
    /// Create a new grayscale erosion filter with the given radius.
    ///
    /// A radius of 0 yields identity (each voxel is its own sole neighbour).
    /// A radius of 1 produces a 3×3×3 cubic structuring element.
    pub fn new(radius: usize) -> Self {
        Self { radius }
    }

    /// Set the structuring element radius.
    pub fn with_radius(mut self, radius: usize) -> Self {
        self.radius = radius;
        self
    }

    /// Apply grayscale erosion to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction). The tensor device of the output matches the input.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let eroded = erode_3d(&vals, dims, self.radius);

        let device = image.data().device();
        let out_td = TensorData::new(eroded, Shape::new(dims));
        let tensor = Tensor::<B, 3>::from_data(out_td, &device);
        Ok(Image::new(
            tensor,
            *image.origin(),
            *image.spacing(),
            *image.direction(),
        ))
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Compute grayscale erosion on a flat 3-D volume stored in Z×Y×X order.
///
/// # Arguments
///
/// * `data`   — flat voxel values in row-major (Z-major) order.
/// * `dims`   — `[nz, ny, nx]`.
/// * `radius` — structuring element half-width in voxels.
///
/// # Invariants
///
/// - Output length equals `nz * ny * nx`.
/// - Each output voxel equals `min_{b ∈ B} data[clamp(x + b)]`.
pub(crate) fn erode_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    super::separable_box_3d(data, dims, radius, super::Extremum::Min)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_grayscale_erosion.rs"]
mod tests_grayscale_erosion;
