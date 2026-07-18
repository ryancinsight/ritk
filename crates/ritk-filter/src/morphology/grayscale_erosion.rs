//! Grayscale erosion filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale erosion with a flat structuring element B is defined as:
//!
//!   (E_B f)(x) = min_{b âˆˆ B} f(x + b)
//!
//! where B is a cubic neighbourhood of half-width `radius`:
//!
//!   B = { b âˆˆ â„¤Â³ : |b_i| â‰¤ r  for i âˆˆ {0, 1, 2} }
//!
//! giving |B| = (2r + 1)Â³ voxels per neighbourhood.
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
//! - **Anti-extensivity**: (E_B f)(x) â‰¤ f(x) for all x (with flat B containing
//!   the origin).
//! - **Translation invariance**: E_B(f(Â· âˆ’ t))(x) = (E_B f)(x âˆ’ t).
//! - **Increasing**: f â‰¤ g â‡’ E_B f â‰¤ E_B g.
//! - **Duality with dilation**: E_B f = âˆ’(D_B(âˆ’f)) for flat structuring elements.
//!
//! # Complexity
//!
//! O(N Â· (2r + 1)Â³) where N = n_z Â· n_y Â· n_x is the total voxel count.
//!
//! # References
//!
//! - Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
//! - Soille, P. (2003). *Morphological Image Analysis*, 2nd ed. Springer.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// â”€â”€ Filter struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Grayscale erosion filter for 3-D images.
///
/// Replaces each voxel with the minimum value in its `(2r+1)Â³` cubic
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
    /// A radius of 1 produces a 3Ã—3Ã—3 cubic structuring element.
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
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let eroded = erode_3d(&vals, dims, self.radius);

        Ok(rebuild(eroded, dims, image))
    }

    /// Coeus-native sister of [`GrayscaleErosion::apply`].
    ///
    /// Runs the identical `(2r+1)Â³` cubic-neighbourhood minimum (replicate
    /// boundary) via the shared `erode_3d` host core on the image's contiguous
    /// host buffer, so the result is bitwise-identical to the Burn path. No Burn
    /// tensor is constructed. Spatial metadata is preserved.
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
            erode_3d(vals, dims, self.radius)
        })
    }
}

// â”€â”€ Core computation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Compute grayscale erosion on a flat 3-D volume stored in ZÃ—YÃ—X order.
///
/// # Arguments
///
/// * `data`   â€” flat voxel values in row-major (Z-major) order.
/// * `dims`   â€” `[nz, ny, nx]`.
/// * `radius` â€” structuring element half-width in voxels.
///
/// # Invariants
///
/// - Output length equals `nz * ny * nx`.
/// - Each output voxel equals `min_{b âˆˆ B} data[clamp(x + b)]`.
pub(crate) fn erode_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    super::separable_box_3d(data, dims, radius, super::Extremum::Min)
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_grayscale_erosion.rs"]
mod tests_grayscale_erosion;
