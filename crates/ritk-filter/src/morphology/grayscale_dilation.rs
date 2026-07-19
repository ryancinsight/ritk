//! Grayscale dilation filter for 3-D images.
//!
//! # Mathematical Specification
//!
//! Grayscale dilation with a flat structuring element B is defined as:
//!
//!   (D_B f)(x) = max_{b ∈ B} f(x - b)
//!
//! where B is a cubic neighbourhood of half-width `radius`:
//!
//!   B = { b ∈ ℤ³ : |b_i| ≤ r  for i ∈ {0, 1, 2} }
//!
//! For a symmetric (origin-centred) structuring element, f(x - b) and f(x + b)
//! range over the same set, so the definition simplifies to:
//!
//!   (D_B f)(x) = max_{b ∈ B} f(x + b)
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
//! - **Idempotence on constant fields**: D_B(c) = c for all constants c.
//! - **Extensivity**: (D_B f)(x) ≥ f(x) for all x (with flat B containing
//!   the origin).
//! - **Translation invariance**: D_B(f(· − t))(x) = (D_B f)(x − t).
//! - **Increasing**: f ≤ g ⇒ D_B f ≤ D_B g.
//! - **Duality with erosion**: D_B f = −(E_B(−f)) for flat structuring elements.
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
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Filter struct ─────────────────────────────────────────────────────────────

/// Grayscale dilation filter for 3-D images.
///
/// Replaces each voxel with the maximum value in its `(2r+1)³` cubic
/// neighbourhood. Out-of-bounds positions use replicate (clamp) padding.
#[derive(Debug, Clone)]
pub struct GrayscaleDilation {
    /// Structuring element half-width in voxels.
    radius: usize,
}

impl GrayscaleDilation {
    /// Create a new grayscale dilation filter with the given radius.
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

    /// Apply grayscale dilation to a 3-D image.
    ///
    /// Returns a new image with identical shape and spatial metadata (origin,
    /// spacing, direction). The tensor device of the output matches the input.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying tensor data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;

        let dilated = dilate_3d(&vals, dims, self.radius);

        Ok(rebuild(dilated, dims, image))
    }

    /// Coeus-native sister of [`GrayscaleDilation::apply`].
    ///
    /// Runs the identical `(2r+1)³` cubic-neighbourhood maximum (replicate
    /// boundary) via the shared `dilate_3d` host core on the image's contiguous
    /// host buffer, so the result is bitwise-identical to the Coeus path. No Burn
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
        crate::native_support::map_flat_image(image, backend, |vals, dims| {
            dilate_3d(vals, dims, self.radius)
        })
    }
}

// ── Core computation ──────────────────────────────────────────────────────────

/// Compute grayscale dilation on a flat 3-D volume stored in Z×Y×X order.
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
/// - Each output voxel equals `max_{b ∈ B} data[clamp(x + b)]`.
pub(crate) fn dilate_3d(data: &[f32], dims: [usize; 3], radius: usize) -> Vec<f32> {
    super::separable_box_3d(data, dims, radius, super::Extremum::Max)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_grayscale_dilation.rs"]
mod tests_grayscale_dilation;
