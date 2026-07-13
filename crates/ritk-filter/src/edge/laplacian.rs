//! Laplacian filter via second-order finite differences.
//!
//! # Mathematical Specification
//!
//! The Laplacian of a 3-D scalar field I is:
//!
//!   ∇²I(x) = ∂²I/∂z² + ∂²I/∂y² + ∂²I/∂x²
//!
//! Each second-order partial derivative is approximated at interior voxel i by:
//!
//! ∂²I/∂z² ≈ (I\[iz+1,iy,ix\] − 2·I\[iz,iy,ix\] + I\[iz−1,iy,ix\]) / sz²
//!
//! At boundary voxels the same `[1, −2, 1]` stencil is evaluated with the
//! out-of-range neighbour clamped to the edge voxel (ZeroFluxNeumann boundary
//! condition); e.g. at iz=0 the lower neighbour is `I[0]`, giving
//! `(I[1] − I[0]) / sz²`. This reproduces ITK's `LaplacianImageFilter`, which
//! couples the Laplacian operator with `ZeroFluxNeumannBoundaryCondition`, to
//! within float rounding.
//!
//! # Reference
//! Press et al., *Numerical Recipes*, 3rd ed., §18.1; boundary handling per ITK
//! `itk::ZeroFluxNeumannBoundaryCondition`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Filter that computes the discrete Laplacian ∇²I of a 3-D image.
///
/// Voxel values in the output are the sum of second-order finite-difference
/// approximations along each axis, divided by the corresponding physical
/// spacing squared.
#[derive(Debug, Clone)]
pub struct LaplacianFilter {
    /// Physical voxel spacing [sz, sy, sx] (same units as image origin).
    pub spacing: Spacing<3>,
}

impl LaplacianFilter {
    /// Create a filter with the given physical spacing.
    pub fn new(spacing: Spacing<3>) -> Self {
        Self { spacing }
    }

    /// Create a filter with unit spacing (1.0 in each direction).
    pub fn unit() -> Self {
        Self {
            spacing: Spacing::uniform(1.0),
        }
    }

    /// Compute the Laplacian image.
    ///
    /// Returns an `Image` whose voxel values are ∇²I(x) at each position x.
    /// The output has the same shape and physical metadata as `image`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        let result = laplacian_vec(&vals, dims, &self.spacing);

        Ok(rebuild(result, dims, image))
    }
}

// ── Coeus-native path ─────────────────────────────────────────────────────────

impl LaplacianFilter {
    /// Coeus-native sister of [`LaplacianFilter::apply`].
    ///
    /// Runs the identical `[1, −2, 1]` second-difference stencil (ZeroFluxNeumann
    /// boundary) via the shared [`laplacian_vec`] host core on the image's
    /// contiguous host buffer, so the result is bitwise-identical to the Burn
    /// path. No Burn tensor is constructed. Spatial metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt tensor fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::native::Image<f32, B, 3>,
    ) -> anyhow::Result<ritk_image::native::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let result = laplacian_vec(&vals, dims, &self.spacing);
        ritk_tensor_ops::native::rebuild_image(result, dims, image, &B::default())
    }
}

// ── Internal computation ──────────────────────────────────────────────────────

/// Compute the Laplacian of a flat 3-D volume.
///
/// # Invariants
/// - `[1, −2, 1]` second-difference stencil divided by spacing² along each axis.
/// - Boundary voxels clamp the out-of-range neighbour to the edge voxel
///   (ZeroFluxNeumann), matching ITK; a length-1 axis contributes 0.
/// - Output length equals `nz * ny * nx`.
pub(crate) fn laplacian_vec(data: &[f32], dims: [usize; 3], spacing: &Spacing<3>) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let sz2 = (spacing[0] * spacing[0]) as f32;
    let sy2 = (spacing[1] * spacing[1]) as f32;
    let sx2 = (spacing[2] * spacing[2]) as f32;

    let idx = |iz: usize, iy: usize, ix: usize| -> usize { iz * ny * nx + iy * nx + ix };

    // Each output voxel depends only on its stencil neighbours, so the grid fans
    // out over the flat index (moirai) — bitwise identical to the serial sweep.
    moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |flat| {
        let iz = flat / (ny * nx);
        let rem = flat % (ny * nx);
        let iy = rem / nx;
        let ix = rem % nx;
        let center = data[flat];

        // ZeroFluxNeumann: clamp out-of-range neighbours to the edge.
        let (zlo, zhi) = (iz.saturating_sub(1), (iz + 1).min(nz - 1));
        let (ylo, yhi) = (iy.saturating_sub(1), (iy + 1).min(ny - 1));
        let (xlo, xhi) = (ix.saturating_sub(1), (ix + 1).min(nx - 1));

        let d2z = (data[idx(zhi, iy, ix)] - 2.0 * center + data[idx(zlo, iy, ix)]) / sz2;
        let d2y = (data[idx(iz, yhi, ix)] - 2.0 * center + data[idx(iz, ylo, ix)]) / sy2;
        let d2x = (data[idx(iz, iy, xhi)] - 2.0 * center + data[idx(iz, iy, xlo)]) / sx2;

        d2z + d2y + d2x
    })
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_laplacian.rs"]
mod tests;
