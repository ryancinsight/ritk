pub mod coherence;
pub mod curvature;
pub mod curvature_flow;
pub mod gradient_anisotropic;
pub mod min_max_curvature_flow;
pub mod perona_malik;

// ── Shared finite-difference stencil helpers ──────────────────────────────────
// Used by the ITK-matching anisotropic-diffusion functions (gradient + curvature),
// which share the same spacing-scaled derivatives and ZeroFluxNeumann boundary.

/// Read a voxel from a flat row-major `[nz, ny, nx]` buffer with ZeroFluxNeumann
/// (replicate / index-clamp) boundary handling.
#[inline]
pub(crate) fn clamp_at(buf: &[f32], dims: [usize; 3], z: isize, y: isize, x: isize) -> f64 {
    let [nz, ny, nx] = dims;
    let zc = z.clamp(0, nz as isize - 1) as usize;
    let yc = y.clamp(0, ny as isize - 1) as usize;
    let xc = x.clamp(0, nx as isize - 1) as usize;
    buf[zc * ny * nx + yc * nx + xc] as f64
}

/// Spacing-scaled central first derivative in dimension `d` at `(z, y, x)`:
/// `(I(+d) − I(−d)) / (2·spacing[d])`, where `inv_2sp[d] = 0.5 / spacing[d]`.
#[inline]
pub(crate) fn central_diff(
    buf: &[f32],
    dims: [usize; 3],
    inv_2sp: [f64; 3],
    d: usize,
    z: isize,
    y: isize,
    x: isize,
) -> f64 {
    match d {
        0 => (clamp_at(buf, dims, z + 1, y, x) - clamp_at(buf, dims, z - 1, y, x)) * inv_2sp[0],
        1 => (clamp_at(buf, dims, z, y + 1, x) - clamp_at(buf, dims, z, y - 1, x)) * inv_2sp[1],
        _ => (clamp_at(buf, dims, z, y, x + 1) - clamp_at(buf, dims, z, y, x - 1)) * inv_2sp[2],
    }
}

pub use coherence::{CoherenceConfig, CoherenceEnhancingDiffusionFilter};
pub use curvature::{CurvatureAnisotropicDiffusionFilter, CurvatureConfig};
pub use curvature_flow::{CurvatureFlowConfig, CurvatureFlowImageFilter};
pub use gradient_anisotropic::{GradientAnisotropicDiffusionFilter, GradientDiffusionConfig};
pub use min_max_curvature_flow::{
    BinaryMinMaxCurvatureFlowConfig, BinaryMinMaxCurvatureFlowImageFilter,
    MinMaxCurvatureFlowConfig, MinMaxCurvatureFlowImageFilter,
};
pub use perona_malik::{
    AnisotropicDiffusionFilter, ConductanceFunction, ConductanceKernel, DiffusionConfig,
    ExponentialConductance, QuadraticConductance,
};
