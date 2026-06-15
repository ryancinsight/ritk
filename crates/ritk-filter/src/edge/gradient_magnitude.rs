//! Gradient magnitude filter using central finite differences.
//!
//! # Mathematical Specification
//!
//! For a 3-D image I defined on a regular grid with physical spacing (sz, sy, sx),
//! the gradient at interior voxel (iz, iy, ix) is estimated by central differences:
//!
//!   ∂I/∂z ≈ (I[iz+1, iy, ix] − I[iz−1, iy, ix]) / (2 · sz)
//!   ∂I/∂y ≈ (I[iz, iy+1, ix] − I[iz, iy−1, ix]) / (2 · sy)
//!   ∂I/∂x ≈ (I[iz, iy, ix+1] − I[iz, iy, ix−1]) / (2 · sx)
//!
//! At boundary voxels the same central stencil is evaluated with the
//! out-of-range neighbour clamped to the edge voxel (ZeroFluxNeumann boundary
//! condition), i.e. at `i = 0` the lower neighbour is `I[0]` and at `i = n−1`
//! the upper neighbour is `I[n−1]`. This reproduces ITK's
//! `GradientMagnitudeImageFilter`, which couples a central `DerivativeOperator`
//! with `ZeroFluxNeumannBoundaryCondition`, to within float rounding.
//!
//! Gradient magnitude: |∇I| = √(gz² + gy² + gx²)
//!
//! # Reference
//! Standard finite difference approximation of the gradient (see e.g., Press et al.,
//! *Numerical Recipes in C*, 3rd ed., §18.1); boundary handling per ITK
//! `itk::ZeroFluxNeumannBoundaryCondition`.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec, rebuild};

/// Filter that computes the gradient magnitude of a 3-D image.
///
/// All gradient components are divided by the corresponding physical spacing so
/// that the result is in units of intensity per millimetre (or whatever unit the
/// spacing is expressed in).
#[derive(Debug, Clone)]
pub struct GradientMagnitudeFilter {
    /// Physical voxel spacing [sz, sy, sx] in each axis direction.
    pub spacing: Spacing<3>,
}

impl GradientMagnitudeFilter {
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

    /// Compute the gradient magnitude image.
    ///
    /// Returns an `Image` whose voxel values are |∇I(x)| at each position x.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        self.apply_from_slice(&vals, dims, image)
    }

    /// Compute the three gradient component images (z, y, x).
    ///
    /// Returns `(grad_z, grad_y, grad_x)`, each an `Image` of the same shape and
    /// physical metadata as `image`.
    pub fn apply_components<B: Backend>(
        &self,
        image: &Image<B, 3>,
    ) -> anyhow::Result<(Image<B, 3>, Image<B, 3>, Image<B, 3>)> {
        let (vals, dims) = extract_vec(image)?;
        let (gz, gy, gx) = gradient_vecs(&vals, dims, &self.spacing);
        Ok((
            rebuild(gz, dims, image),
            rebuild(gy, dims, image),
            rebuild(gx, dims, image),
        ))
    }

    /// Compute the gradient magnitude image from a pre-extracted flat `&[f32]` slice.
    ///
    /// Equivalent to \[`apply`\] but accepts input data already extracted from the
    /// image tensor, enabling zero-copy extraction when the caller has obtained a
    /// slice via `NdArrayTensor::as_slice_memory_order()`.
    ///
    /// # Arguments
    /// * `vals`  — Flat voxel data in \[Z, Y, X\] C-order, length `dims[0]*dims[1]*dims[2]`.
    /// * `dims`  — Image dimensions `[nz, ny, nx]`.
    /// * `src`   — Reference image; spatial metadata (origin, spacing, direction) is cloned.
    ///
    /// # Errors
    /// Returns an error only if `dims` is inconsistent with `vals.len()` (debug_assert).
    pub fn apply_from_slice<B: Backend>(
        &self,
        vals: &[f32],
        dims: [usize; 3],
        src: &Image<B, 3>,
    ) -> anyhow::Result<Image<B, 3>> {
        let [nz, ny, nx] = dims;
        let sz = self.spacing[0] as f32;
        let sy = self.spacing[1] as f32;
        let sx = self.spacing[2] as f32;

        debug_assert_eq!(
            vals.len(),
            nz * ny * nx,
            "apply_from_slice: vals.len() != nz*ny*nx"
        );

        let mag: Vec<f32> =
            moirai::map_collect_index_with::<moirai::Adaptive, _, _>(nz * ny * nx, |flat| {
                let iz = flat / (ny * nx);
                let iy = (flat / nx) % ny;
                let ix = flat % nx;
                let f = |z: usize, y: usize, x: usize| vals[z * ny * nx + y * nx + x];

                // ZeroFluxNeumann central difference: clamp the out-of-range
                // neighbour to the edge voxel. A degenerate (length-1) axis
                // yields a zero numerator, so the partial derivative is 0.
                let (zlo, zhi) = (iz.saturating_sub(1), (iz + 1).min(nz - 1));
                let (ylo, yhi) = (iy.saturating_sub(1), (iy + 1).min(ny - 1));
                let (xlo, xhi) = (ix.saturating_sub(1), (ix + 1).min(nx - 1));

                let gz = (f(zhi, iy, ix) - f(zlo, iy, ix)) / (2.0 * sz);
                let gy = (f(iz, yhi, ix) - f(iz, ylo, ix)) / (2.0 * sy);
                let gx = (f(iz, iy, xhi) - f(iz, iy, xlo)) / (2.0 * sx);

                (gz * gz + gy * gy + gx * gx).sqrt()
            });

        Ok(rebuild(mag, dims, src))
    }
}

// ── gradient_vecs ────────────────────────────────────────────────────────────────

/// Compute gradient component vectors (gz, gy, gx) via finite differences.
///
/// # Invariants
/// - Central second-order differences divided by 2·spacing everywhere.
/// - Boundary voxels clamp the out-of-range neighbour to the edge voxel
///   (ZeroFluxNeumann), matching ITK; a length-1 axis yields a zero component.
/// - Output lengths equal `nz * ny * nx`.
fn gradient_vecs(
    data: &[f32],
    dims: [usize; 3],
    spacing: &Spacing<3>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;

    let mut gz = vec![0.0_f32; n];
    let mut gy = vec![0.0_f32; n];
    let mut gx = vec![0.0_f32; n];

    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

    let idx = |iz: usize, iy: usize, ix: usize| iz * ny * nx + iy * nx + ix;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let flat = idx(iz, iy, ix);
                let (zlo, zhi) = (iz.saturating_sub(1), (iz + 1).min(nz - 1));
                let (ylo, yhi) = (iy.saturating_sub(1), (iy + 1).min(ny - 1));
                let (xlo, xhi) = (ix.saturating_sub(1), (ix + 1).min(nx - 1));

                gz[flat] = (data[idx(zhi, iy, ix)] - data[idx(zlo, iy, ix)]) / (2.0 * sz);
                gy[flat] = (data[idx(iz, yhi, ix)] - data[idx(iz, ylo, ix)]) / (2.0 * sy);
                gx[flat] = (data[idx(iz, iy, xhi)] - data[idx(iz, iy, xlo)]) / (2.0 * sx);
            }
        }
    }

    (gz, gy, gx)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_gradient_magnitude.rs"]
mod tests;
