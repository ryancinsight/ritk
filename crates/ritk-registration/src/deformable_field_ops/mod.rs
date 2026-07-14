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

use ritk_image::tensor::Backend;
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
pub(crate) use smooth::gaussian_smooth_with_scratch_per_axis;
pub(crate) use smooth::{gaussian_smooth_field_with_kernel, gaussian_smooth_inplace};

// ── GPU-accelerated smoothing (Burn tensor path) ─────────────────────────────
pub use smooth::GpuFieldSmoother;
pub(crate) use validation::{cc_converged, validate_image, validate_image_pair};
pub(crate) use warp::{compute_mse_inplace, compute_mse_streaming, warp_image_into};
pub use warp::{warp_image, WarpInterpolation};

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
    /// The three slices must have the same length and match the spatial shape
    /// supplied when the implementation was constructed.
    ///
    /// # Panics
    ///
    /// Panics when a component length differs from the configured shape.
    fn smooth_field(&mut self, z: &mut [f32], y: &mut [f32], x: &mut [f32]);
}

/// CPU-based Gaussian field smoother.
///
/// Caches the Gaussian weights and pre-allocates three scratch components so
/// the hot path performs no per-call buffer or kernel allocation and one final
/// field copy.
///
/// # Example
///
/// ```ignore
/// use ritk_registration::deformable_field_ops::{CpuFieldSmoother, FieldSmoother};
///
/// let mut smoother = CpuFieldSmoother::new([256, 256, 256], 1.5);
/// smoother.smooth_field(&mut fz, &mut fy, &mut fx);
/// ```
pub struct CpuFieldSmoother {
    smooth_tmp_z: Vec<f32>,
    smooth_tmp_y: Vec<f32>,
    smooth_tmp_x: Vec<f32>,
    dims: VolumeDims,
    kernel: Vec<f64>,
}

impl CpuFieldSmoother {
    /// Create a CPU smoother for `dims` with isotropic `sigma` in voxels.
    ///
    /// For positive sigma, allocates three scratch buffers of size
    /// `nz * ny * nx` and caches the isotropic Gaussian weights. A non-positive
    /// sigma constructs an allocation-free no-op.
    pub fn new(dims: [usize; 3], sigma: f64) -> Self {
        let n = dims[0] * dims[1] * dims[2];
        let kernel = if sigma <= 0.0 {
            Vec::new()
        } else {
            ritk_filter::gaussian_kernel(sigma, None)
        };
        let scratch_len = if kernel.is_empty() { 0 } else { n };
        Self {
            smooth_tmp_z: vec![0.0_f32; scratch_len],
            smooth_tmp_y: vec![0.0_f32; scratch_len],
            smooth_tmp_x: vec![0.0_f32; scratch_len],
            dims: VolumeDims::new(dims),
            kernel,
        }
    }
}

impl FieldSmoother for CpuFieldSmoother {
    fn smooth_field(&mut self, z: &mut [f32], y: &mut [f32], x: &mut [f32]) {
        let expected = self.dims.total_voxels();
        assert_eq!(
            z.len(),
            expected,
            "z component length must match dimensions"
        );
        assert_eq!(
            y.len(),
            expected,
            "y component length must match dimensions"
        );
        assert_eq!(
            x.len(),
            expected,
            "x component length must match dimensions"
        );
        gaussian_smooth_field_with_kernel(
            VectorFieldMut { z, y, x },
            self.dims,
            &self.kernel,
            VectorFieldMut {
                z: &mut self.smooth_tmp_z,
                y: &mut self.smooth_tmp_y,
                x: &mut self.smooth_tmp_x,
            },
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
/// the default [`ritk_image::burn::backend::NdArray`] backend keeps the enum usable without
/// selecting a concrete GPU provider.
pub enum CpuOrGpu<B: Backend = ritk_image::burn::backend::NdArray> {
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

    /// Resize all three components to `n`, zero-filling new elements.
    pub fn resize(&mut self, n: usize) {
        self.z.resize(n, 0.0);
        self.y.resize(n, 0.0);
        self.x.resize(n, 0.0);
    }
}

// ── Indexing ──────────────────────────────────────────────────────────────────

/// Flat voxel index for shape `[nz, ny, nx]`.
#[inline(always)]
pub(crate) fn flat(iz: usize, iy: usize, ix: usize, ny: usize, nx: usize) -> usize {
    iz * ny * nx + iy * nx + ix
}

// ── Trilinear interpolation ───────────────────────────────────────────────────

#[derive(Clone, Copy)]
struct TrilinearStencil {
    indices: [usize; 8],
    dz: f32,
    dy: f32,
    dx: f32,
}

impl TrilinearStencil {
    #[inline(always)]
    fn new(dims: VolumeDims, z: f32, y: f32, x: f32) -> Self {
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

        Self {
            indices: [
                flat(iz0, iy0, ix0, ny, nx),
                flat(iz0, iy0, ix1, ny, nx),
                flat(iz0, iy1, ix0, ny, nx),
                flat(iz0, iy1, ix1, ny, nx),
                flat(iz1, iy0, ix0, ny, nx),
                flat(iz1, iy0, ix1, ny, nx),
                flat(iz1, iy1, ix0, ny, nx),
                flat(iz1, iy1, ix1, ny, nx),
            ],
            dz: z - iz0 as f32,
            dy: y - iy0 as f32,
            dx: x - ix0 as f32,
        }
    }

    #[inline(always)]
    fn sample(self, data: &[f32]) -> f32 {
        let [i000, i001, i010, i011, i100, i101, i110, i111] = self.indices;
        let c00 = data[i000] * (1.0 - self.dx) + data[i001] * self.dx;
        let c01 = data[i010] * (1.0 - self.dx) + data[i011] * self.dx;
        let c10 = data[i100] * (1.0 - self.dx) + data[i101] * self.dx;
        let c11 = data[i110] * (1.0 - self.dx) + data[i111] * self.dx;
        let c0 = c00 * (1.0 - self.dy) + c01 * self.dy;
        let c1 = c10 * (1.0 - self.dy) + c11 * self.dy;
        c0 * (1.0 - self.dz) + c1 * self.dz
    }
}

/// Sample `data` at a continuous position `(z, y, x)` using trilinear
/// interpolation with clamp-to-border boundary condition.
///
/// # Invariants
/// - At integer positions the result equals `data[flat(round(z), round(y), round(x))]`.
/// - Positions outside `[0, nZ−1] × [0, nY−1] × [0, nX−1]` are clamped.
#[inline]
pub(crate) fn trilinear_interpolate(data: &[f32], dims: VolumeDims, z: f32, y: f32, x: f32) -> f32 {
    TrilinearStencil::new(dims, z, y, x).sample(data)
}

/// Sample all components of a vector field with one shared trilinear stencil.
#[inline]
pub(crate) fn trilinear_interpolate_field(
    field: VectorField<'_>,
    dims: VolumeDims,
    z: f32,
    y: f32,
    x: f32,
) -> [f32; 3] {
    let stencil = TrilinearStencil::new(dims, z, y, x);
    [
        stencil.sample(field.z),
        stencil.sample(field.y),
        stencil.sample(field.x),
    ]
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

    #[test]
    fn vector_trilinear_matches_scalar_components_exactly() {
        let dims = VolumeDims::new([4, 5, 6]);
        let z = make_ramp(dims);
        let y: Vec<f32> = z.iter().map(|value| value.mul_add(-0.75, 2.0)).collect();
        let x: Vec<f32> = z.iter().map(|value| value.mul_add(1.25, -3.0)).collect();
        let position = [-0.25, 2.3, 5.4];
        let actual = trilinear_interpolate_field(
            VectorField {
                z: &z,
                y: &y,
                x: &x,
            },
            dims,
            position[0],
            position[1],
            position[2],
        );
        let expected = [
            trilinear_interpolate(&z, dims, position[0], position[1], position[2]),
            trilinear_interpolate(&y, dims, position[0], position[1], position[2]),
            trilinear_interpolate(&x, dims, position[0], position[1], position[2]),
        ];
        assert_eq!(actual, expected);
    }
}
