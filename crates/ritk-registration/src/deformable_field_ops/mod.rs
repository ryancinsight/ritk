//! Shared computational primitives for deformable image registration algorithms.
//!
//! All functions operate on flat `Vec<f32>` buffers with shape `[nz, ny, nx]`
//! (Z-major / row-major order): flat index = `iz * ny * nx + iy * nx + ix`.
//!
//! # Conventions
//! - `dims = [nz, ny, nx]`  — image dimensions
//! - `spacing = [sz, sy, sx]` — physical voxel size (mm or arbitrary units)
//! - Displacement components are stored in voxel units (not physical units)
//!
//! # Boundary conditions
//! All sampling operations use **clamp-to-border** (replicate boundary):
//! coordinates outside `[0, dim − 1]` are clamped to the nearest valid index.

mod compose;
mod gradient;
mod integrate;
mod normalize;
mod smooth;
mod validation;
mod warp;

use burn::tensor::backend::Backend;
use ritk_spatial::VolumeDims;

// Flat `flat` and `trilinear_interpolate` are defined here to avoid peer-module
// resolution cycles: every sub-module accesses them via `use super::{flat, trilinear_interpolate}`.
#[cfg(test)]
pub(crate) use compose::compose_fields;
pub(crate) use compose::compose_fields_into;
pub(crate) use gradient::{compute_gradient, compute_gradient_into};
pub(crate) use integrate::{scaling_and_squaring, scaling_and_squaring_into};
pub(crate) use normalize::normalize_forces_into;
#[cfg(test)]
pub(crate) use smooth::gaussian_smooth_field_inplace;
pub(crate) use smooth::{
    gaussian_smooth_field_inplace_with_scratch, gaussian_smooth_inplace,
    gaussian_smooth_with_scratch,
};

// ── GPU-accelerated smoothing (Burn tensor path) ─────────────────────────────
pub use smooth::GpuFieldSmoother;
pub(crate) use validation::{cc_converged, validate_image, validate_image_pair};
pub(crate) use warp::{compute_mse_inplace, compute_mse_streaming, warp_image, warp_image_into};

// ── FieldSmoother trait — zero-cost abstraction over CPU/GPU smoothing ───────

/// Smooth a 3-component displacement or velocity field in place.
///
/// Implementations include [`CpuFieldSmoother`] (CPU `moirai` path) and
/// [`GpuFieldSmoother`] (Burn GPU path).  Registration engines accept
/// `impl FieldSmoother` so callers choose the backend at the call site
/// without needing separate `register_with_gpu_smoother` entry points.
pub trait FieldSmoother {
    /// Smooth the three components of a vector field **in place**.
    ///
    /// The three slices must have the same length.  A sigma ≤ 0 is a no-op
    /// (the trait does not prescribe how implementations handle this).
    fn smooth_field(&mut self, z: &mut [f32], y: &mut [f32], x: &mut [f32]);
}

/// CPU-based Gaussian field smoother.
///
/// Uses [`gaussian_smooth_field_inplace_with_scratch`] internally.
/// Pre-allocates a scratch buffer so the hot path performs zero heap
/// allocations.
///
/// # Example
///
/// ```no_run
/// use ritk_registration::deformable_field_ops::{CpuFieldSmoother, FieldSmoother};
///
/// let mut smoother = CpuFieldSmoother::new([256, 256, 256], 1.5);
/// smoother.smooth_field(&mut fz, &mut fy, &mut fx);
/// ```
pub struct CpuFieldSmoother {
    smooth_tmp: Vec<f32>,
    dims: VolumeDims,
    sigma: f64,
}

impl CpuFieldSmoother {
    /// Create a CPU smoother for `dims` with isotropic `sigma` (mm).
    ///
    /// Allocates one scratch buffer of size `nz * ny * nx`.
    pub fn new(dims: [usize; 3], sigma: f64) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        Self {
            smooth_tmp: vec![0.0_f32; n],
            dims: VolumeDims::new(dims),
            sigma,
        }
    }
}

impl FieldSmoother for CpuFieldSmoother {
    fn smooth_field(&mut self, z: &mut [f32], y: &mut [f32], x: &mut [f32]) {
        gaussian_smooth_field_inplace_with_scratch(
            z,
            y,
            x,
            self.dims,
            self.sigma,
            &mut self.smooth_tmp,
        );
    }
}

// ── CpuOrGpu enum — static dispatch without Box heap allocation ──────────────

/// Static-dispatch union of [`CpuFieldSmoother`] and [`GpuFieldSmoother`].
///
/// Replaces `Box<dyn FieldSmoother>` in multi-resolution registration loops
/// where a per-level smoother must be created.  The enum is stack-allocated
/// and uses a match arm instead of vtable dispatch — zero heap allocations,
/// zero dynamic dispatch.
///
/// # Type parameter
///
/// `B` is the Burn backend used by the GPU variant.  For CPU-only usage,
/// any backend will work (it is never used by the [`Cpu`](CpuOrGpu::Cpu)
/// arm) — [`burn::backend::Wgpu`] is a convenient choice.
pub enum CpuOrGpu<B: Backend = burn::backend::Wgpu> {
    /// CPU Gaussian smoother with pre-allocated scratch buffer.
    Cpu(CpuFieldSmoother),
    /// GPU Gaussian smoother with Burn backend `B`.
    Gpu(GpuFieldSmoother<B>),
}

impl<B: Backend> FieldSmoother for CpuOrGpu<B> {
    fn smooth_field(&mut self, z: &mut [f32], y: &mut [f32], x: &mut [f32]) {
        match self {
            Self::Cpu(s) => s.smooth_field(z, y, x),
            Self::Gpu(s) => s.smooth_field(z, y, x),
        }
    }
}

// ── 3D vector field grouping structs ─────────────────────────────────────────

/// Immutable 3D vector field: three co-equal flat component slices (z, y, x).
///
/// Groups the three spatial components of a displacement, velocity, momentum, or
/// gradient field so that functions stay within Clippy's `too_many_arguments` limit.
#[derive(Debug, Clone, Copy)]
pub(crate) struct VectorField<'a> {
    /// Z-component (flat buffer, Z-major order).
    pub z: &'a [f32],
    /// Y-component.
    pub y: &'a [f32],
    /// X-component.
    pub x: &'a [f32],
}

/// Mutable 3D vector field: three co-equal flat component slice-muts (z, y, x).
#[derive(Debug)]
pub(crate) struct VectorFieldMut<'a> {
    /// Z-component (flat buffer, Z-major order).
    pub z: &'a mut [f32],
    /// Y-component.
    pub y: &'a mut [f32],
    /// X-component.
    pub x: &'a mut [f32],
}

/// Owned 3-D velocity or displacement field: three flat `Vec<f32>` component buffers.
///
/// Component ordering is `(z, y, x)` matching the Z-major memory layout used
/// throughout `deformable_field_ops`. Named fields eliminate positional ambiguity
/// over the `.0`/`.1`/`.2` tuple access pattern.
#[derive(Debug, Clone)]
pub struct VelocityField {
    /// Z-component buffer (flat, Z-major order).
    pub z: Vec<f32>,
    /// Y-component buffer.
    pub y: Vec<f32>,
    /// X-component buffer.
    pub x: Vec<f32>,
}

impl VelocityField {
    /// Construct from separate component buffers.
    pub fn new(z: Vec<f32>, y: Vec<f32>, x: Vec<f32>) -> Self {
        Self { z, y, x }
    }

    /// Zero-initialize all three components with `n` elements each.
    pub fn zeros(n: usize) -> Self {
        Self {
            z: vec![0.0_f32; n],
            y: vec![0.0_f32; n],
            x: vec![0.0_f32; n],
        }
    }

    /// Total number of voxels (length of each component buffer).
    ///
    /// # Panics
    /// Panics (debug only) if the three component buffers have different lengths.
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.z.len(), self.y.len());
        debug_assert_eq!(self.y.len(), self.x.len());
        self.z.len()
    }

    /// Returns `true` if all component buffers are empty.
    pub fn is_empty(&self) -> bool {
        self.z.is_empty()
    }
}

// ── Indexing ──────────────────────────────────────────────────────────────────

/// Flat voxel index for shape `[nz, ny, nx]`.
#[inline(always)]
pub(crate) fn flat(iz: usize, iy: usize, ix: usize, ny: usize, nx: usize) -> usize {
    iz * ny * nx + iy * nx + ix
}

// ── Trilinear interpolation ───────────────────────────────────────────────────

/// Sample `data` at a continuous position `(z, y, x)` using trilinear
/// interpolation with clamp-to-border boundary condition.
///
/// # Invariants
/// - At integer positions the result equals `data[flat(round(z), round(y), round(x))]`.
/// - Positions outside `[0, nZ−1] × [0, nY−1] × [0, nX−1]` are clamped.
#[inline]
pub(crate) fn trilinear_interpolate(data: &[f32], dims: VolumeDims, z: f32, y: f32, x: f32) -> f32 {
    let [nz, ny, nx] = dims.0;

    let z = z.max(0.0).min((nz as f32) - 1.0);
    let y = y.max(0.0).min((ny as f32) - 1.0);
    let x = x.max(0.0).min((nx as f32) - 1.0);

    let iz0 = z.floor() as usize;
    let iy0 = y.floor() as usize;
    let ix0 = x.floor() as usize;
    let iz1 = (iz0 + 1).min(nz - 1);
    let iy1 = (iy0 + 1).min(ny - 1);
    let ix1 = (ix0 + 1).min(nx - 1);

    let dz = z - iz0 as f32;
    let dy = y - iy0 as f32;
    let dx = x - ix0 as f32;

    let g = |iz: usize, iy: usize, ix: usize| data[flat(iz, iy, ix, ny, nx)];

    let c00 = g(iz0, iy0, ix0) * (1.0 - dx) + g(iz0, iy0, ix1) * dx;
    let c01 = g(iz0, iy1, ix0) * (1.0 - dx) + g(iz0, iy1, ix1) * dx;
    let c10 = g(iz1, iy0, ix0) * (1.0 - dx) + g(iz1, iy0, ix1) * dx;
    let c11 = g(iz1, iy1, ix0) * (1.0 - dx) + g(iz1, iy1, ix1) * dx;

    let c0 = c00 * (1.0 - dy) + c01 * dy;
    let c1 = c10 * (1.0 - dy) + c11 * dy;

    c0 * (1.0 - dz) + c1 * dz
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ramp(dims: VolumeDims) -> Vec<f32> {
        let [nz, ny, nx] = dims.0;
        (0..nz * ny * nx)
            .map(|fi| {
                let ix = fi % nx;
                let iy = (fi / nx) % ny;
                let iz = fi / (ny * nx);
                (iz + iy + ix) as f32
            })
            .collect()
    }

    /// At an integer coordinate the trilinear interpolant must equal the stored value.
    #[test]
    fn trilinear_at_integer_equals_value() {
        let dims = VolumeDims::new([5, 5, 5]);
        let data = make_ramp(dims);
        for iz in 0..5 {
            for iy in 0..5 {
                for ix in 0..5 {
                    let fi = flat(iz, iy, ix, 5, 5);
                    let v = trilinear_interpolate(&data, dims, iz as f32, iy as f32, ix as f32);
                    assert!(
                        (v - data[fi]).abs() < 1e-5,
                        "({iz},{iy},{ix}): expected {}, got {v}",
                        data[fi]
                    );
                }
            }
        }
    }

    /// Trilinear interpolation of a constant field returns the constant everywhere.
    #[test]
    fn trilinear_constant_field() {
        let dims = VolumeDims::new([4, 4, 4]);
        let data = vec![7.0_f32; 4 * 4 * 4];
        let v = trilinear_interpolate(&data, dims, 1.7, 2.3, 0.8);
        assert!((v - 7.0).abs() < 1e-5, "expected 7.0, got {v}");
    }
}
