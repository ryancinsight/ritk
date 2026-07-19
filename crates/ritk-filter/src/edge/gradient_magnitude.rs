//! Gradient magnitude filter using central finite differences.
//!
//! # Mathematical Specification
//!
//! For a 3-D image I defined on a regular grid with physical spacing (sz, sy, sx),
//! the gradient at interior voxel (iz, iy, ix) is estimated by central differences:
//!
//!   âˆ‚I/âˆ‚z â‰ˆ (I[iz+1, iy, ix] âˆ’ I[izâˆ’1, iy, ix]) / (2 Â· sz)
//!   âˆ‚I/âˆ‚y â‰ˆ (I[iz, iy+1, ix] âˆ’ I[iz, iyâˆ’1, ix]) / (2 Â· sy)
//!   âˆ‚I/âˆ‚x â‰ˆ (I[iz, iy, ix+1] âˆ’ I[iz, iy, ixâˆ’1]) / (2 Â· sx)
//!
//! At boundary voxels the same central stencil is evaluated with the
//! out-of-range neighbour clamped to the edge voxel (ZeroFluxNeumann boundary
//! condition), i.e. at `i = 0` the lower neighbour is `I[0]` and at `i = nâˆ’1`
//! the upper neighbour is `I[nâˆ’1]`. This reproduces ITK's
//! `GradientMagnitudeImageFilter`, which couples a central `DerivativeOperator`
//! with `ZeroFluxNeumannBoundaryCondition`, to within float rounding.
//!
//! Gradient magnitude: |âˆ‡I| = âˆš(gzÂ² + gyÂ² + gxÂ²)
//!
//! # Reference
//! Standard finite difference approximation of the gradient (see e.g., Press et al.,
//! *Numerical Recipes in C*, 3rd ed., Â§18.1); boundary handling per ITK
//! `itk::ZeroFluxNeumannBoundaryCondition`.

use ritk_image::tensor::Backend;
use ritk_image::Image;
use ritk_spatial::Spacing;
use ritk_tensor_ops::{extract_vec, rebuild};

type GradientImages<B> = (Image<f32, B, 3>, Image<f32, B, 3>, Image<f32, B, 3>);

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
    /// Returns an `Image` whose voxel values are |âˆ‡I(x)| at each position x.
    pub fn apply<B: Backend>(&self, image: &Image<f32, B, 3>) -> anyhow::Result<Image<f32, B, 3>> {
        let (vals, dims) = extract_vec(image)?;
        self.apply_from_slice(&vals, dims, image)
    }

    /// Compute the three gradient component images (z, y, x).
    ///
    /// Returns `(grad_z, grad_y, grad_x)`, each an `Image` of the same shape and
    /// physical metadata as `image`.
    pub fn apply_components<B: Backend>(
        &self,
        image: &Image<f32, B, 3>,
    ) -> anyhow::Result<GradientImages<B>> {
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
    /// * `vals`  â€” Flat voxel data in \[Z, Y, X\] C-order, length `dims[0]*dims[1]*dims[2]`.
    /// * `dims`  â€” Image dimensions `[nz, ny, nx]`.
    /// * `src`   â€” Reference image; spatial metadata (origin, spacing, direction) is cloned.
    ///
    /// # Errors
    /// Returns an error only if `dims` is inconsistent with `vals.len()` (debug_assert).
    pub fn apply_from_slice<B: Backend>(
        &self,
        vals: &[f32],
        dims: [usize; 3],
        src: &Image<f32, B, 3>,
    ) -> anyhow::Result<Image<f32, B, 3>> {
        let [nz, ny, nx] = dims;

        debug_assert_eq!(
            vals.len(),
            nz * ny * nx,
            "apply_from_slice: vals.len() != nz*ny*nx"
        );

        let mag = gradient_magnitude_vec(vals, dims, &self.spacing);
        Ok(rebuild(mag, dims, src))
    }
}

// â”€â”€ Coeus-native path â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

impl GradientMagnitudeFilter {
    /// Coeus-native sister of [`GradientMagnitudeFilter::apply`].
    ///
    /// Runs the identical central finite-difference gradient magnitude
    /// (ZeroFluxNeumann boundary) via the shared `gradient_magnitude_vec` host
    /// core on the image's contiguous host buffer, so the result is
    /// bitwise-identical to the Burn path. No Burn tensor is constructed. Spatial
    /// metadata is preserved.
    ///
    /// # Errors
    /// Returns an error when the image tensor is not host-addressable/contiguous
    /// or the rebuilt tensor fails shape validation.
    pub fn apply_native<B>(
        &self,
        image: &ritk_image::Image<f32, B, 3>,
    ) -> anyhow::Result<ritk_image::Image<f32, B, 3>>
    where
        B: coeus_core::ComputeBackend + Default,
        B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32>,
    {
        let (vals, dims) = ritk_tensor_ops::native::extract_image_vec(image)?;
        let mag = gradient_magnitude_vec(&vals, dims, &self.spacing);
        ritk_tensor_ops::native::rebuild_image(mag, dims, image, &B::default())
    }
}

// â”€â”€ gradient_magnitude_vec â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Central finite-difference gradient magnitude on a flat 3-D volume.
///
/// # Invariants
/// - Central second-order differences divided by `2Â·spacing` per axis.
/// - Boundary voxels clamp the out-of-range neighbour to the edge voxel
///   (ZeroFluxNeumann), matching ITK; a length-1 axis yields a zero component.
/// - `|âˆ‡I| = âˆš(gzÂ² + gyÂ² + gxÂ²)`; output length equals `nz Â· ny Â· nx`.
///
/// Single host realization shared by the Burn [`GradientMagnitudeFilter::apply`]
/// path and the Coeus-native [`GradientMagnitudeFilter::apply_native`] path.
pub(crate) fn gradient_magnitude_vec(
    vals: &[f32],
    dims: [usize; 3],
    spacing: &Spacing<3>,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;

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
    })
}

// â”€â”€ gradient_vecs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Compute gradient component vectors (gz, gy, gx) via finite differences.
///
/// # Invariants
/// - Central second-order differences divided by 2Â·spacing everywhere.
/// - Boundary voxels clamp the out-of-range neighbour to the edge voxel
///   (ZeroFluxNeumann), matching ITK; a length-1 axis yields a zero component.
/// - Output lengths equal `nz * ny * nx`.
///
/// PERF-378-02: parallelised over the flat voxel index with a single pass that
/// computes all three components per voxel, then scatters into three output
/// vectors. One read of `data` per voxel (cache-friendly); scatter is O(n) serial.
fn gradient_vecs(
    data: &[f32],
    dims: [usize; 3],
    spacing: &Spacing<3>,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let sz = spacing[0] as f32;
    let sy = spacing[1] as f32;
    let sx = spacing[2] as f32;
    let slab = ny * nx;
    let idx = |iz: usize, iy: usize, ix: usize| iz * slab + iy * nx + ix;

    // One parallel pass: compute [gz, gy, gx] per voxel in a single data sweep.
    let triplets: Vec<[f32; 3]> =
        moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |flat| {
            let iz = flat / slab;
            let iy = (flat / nx) % ny;
            let ix = flat % nx;
            let (zlo, zhi) = (iz.saturating_sub(1), (iz + 1).min(nz - 1));
            let (ylo, yhi) = (iy.saturating_sub(1), (iy + 1).min(ny - 1));
            let (xlo, xhi) = (ix.saturating_sub(1), (ix + 1).min(nx - 1));
            [
                (data[idx(zhi, iy, ix)] - data[idx(zlo, iy, ix)]) / (2.0 * sz),
                (data[idx(iz, yhi, ix)] - data[idx(iz, ylo, ix)]) / (2.0 * sy),
                (data[idx(iz, iy, xhi)] - data[idx(iz, iy, xlo)]) / (2.0 * sx),
            ]
        });

    // Scatter into three contiguous component vectors.
    let mut gz = Vec::with_capacity(n);
    let mut gy = Vec::with_capacity(n);
    let mut gx = Vec::with_capacity(n);
    for [z, y, x] in triplets {
        gz.push(z);
        gy.push(y);
        gx.push(x);
    }
    (gz, gy, gx)
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_gradient_magnitude.rs"]
mod tests;
