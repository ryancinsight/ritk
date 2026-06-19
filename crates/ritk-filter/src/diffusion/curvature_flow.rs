//! Mean curvature flow image filter.
//!
//! # Mathematical Specification
//!
//! Implements the pure mean curvature flow PDE (Osher & Sethian 1988):
//!
//!   ∂I/∂t = κ = div(∇I / |∇I|)
//!
//! where κ is the mean curvature of the iso-intensity level set through each voxel.
//!
//! Unlike `CurvatureAnisotropicDiffusionFilter` (which uses `∂I/∂t = |∇I| · κ`),
//! this filter does NOT weight the update by the gradient magnitude. This matches
//! ITK `itk::CurvatureFlowImageFilter` exactly.
//!
//! # 3-D Finite-Difference Discretisation
//!
//! Let first derivatives (central differences at interior, one-sided at boundaries):
//! I_x = (I\[p+x\] − I\[p−x\]) / 2
//! I_y = (I\[p+y\] − I\[p−y\]) / 2
//! I_z = (I\[p+z\] − I\[p−z\]) / 2
//!
//! Let second derivatives (symmetric 3-point stencils):
//! I_xx = I\[p+x\] − 2·I\[p\] + I\[p−x\]
//!   ... (similarly for yy, zz, xy, xz, yz)
//!
//! Mean curvature numerator (Caselles, Kimmel, Sapiro 1997):
//!   N = I_xx·(I_y² + I_z²)
//!     + I_yy·(I_x² + I_z²)
//!     + I_zz·(I_x² + I_y²)
//!     − 2·I_x·I_y·I_xy
//!     − 2·I_x·I_z·I_xz
//!     − 2·I_y·I_z·I_yz
//!
//! Gradient magnitude: |∇I|² = I_x² + I_y² + I_z²
//! Denominator: D = |∇I|³ = (I_x² + I_y² + I_z²)^(3/2)
//!
//! Update:
//!   κ(p) = N / D  if D > ε, else 0
//!   I_new(p) = I(p) + Δt · κ(p)
//!
//! # Stability
//!
//! For 3-D isotropic grids (Δx = Δy = Δz = 1), the explicit-Euler stability bound is:
//!   Δt ≤ 1/6 ≈ 0.1667
//!
//! ITK default: iterations=5, Δt=0.0625 (well within stability bound).
//!
//! # References
//! - ITK `itk::CurvatureFlowImageFilter<TInputImage, TOutputImage>`.
//! - Osher, S. and Sethian, J. A. (1988). "Fronts propagating with
//!   curvature-dependent speed: Algorithms based on Hamilton-Jacobi
//!   formulations." J. Comput. Phys. 79(1):12-49.
//! - Caselles, V., Kimmel, R., Sapiro, G. (1997). "Geodesic active contours."
//!   Int. J. Comput. Vis. 22(1):61-79.

use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Stability constant ────────────────────────────────────────────────────────

/// Minimum denominator magnitude below which curvature is clamped to zero.
/// Prevents division by zero in flat regions.
const GRAD_MAG_EPSILON: f32 = 1e-9;

// ── CurvatureFlowConfig ───────────────────────────────────────────────────────

/// Configuration for `CurvatureFlowImageFilter`.
///
/// ITK defaults: `num_iterations = 5`, `time_step = 0.0625`.
#[derive(Debug, Clone, Copy)]
pub struct CurvatureFlowConfig {
    /// Number of explicit-Euler iterations. ITK default: 5.
    pub num_iterations: usize,
    /// Explicit Euler time step Δt. Must satisfy `Δt ≤ 1/6` for 3-D stability.
    /// ITK default: 0.0625.
    pub time_step: f32,
}

impl Default for CurvatureFlowConfig {
    fn default() -> Self {
        Self {
            num_iterations: 5,
            time_step: 0.0625,
        }
    }
}

// ── CurvatureFlowImageFilter ──────────────────────────────────────────────────

/// Pure mean curvature flow filter.
///
/// Evolves the image by `∂I/∂t = |∇I|·κ` (level-set mean curvature flow,
/// matching ITK `CurvatureFlowImageFilter`) for a fixed number of explicit-Euler
/// iterations. The `|∇I|` factor cancels the flat-region singularity of the bare
/// curvature `κ = div(∇I/|∇I|)`, keeping the evolution stable. This smooths small
/// structures while preserving larger geometric features longer than Gaussian
/// smoothing.
///
/// # Differences from `CurvatureAnisotropicDiffusionFilter`
/// - This filter: `∂I/∂t = |∇I|·κ` (pure level-set curvature flow).
/// - `CurvatureAnisotropicDiffusionFilter`: gradient-weighted anisotropic
///   diffusion with a conductance term.
///
/// # Construction
/// ```rust,ignore
/// let filter = CurvatureFlowImageFilter::new(CurvatureFlowConfig {
///     num_iterations: 5,
///     time_step: 0.0625,
/// });
/// let smoothed = filter.apply(&image)?;
/// ```
#[derive(Debug, Clone)]
pub struct CurvatureFlowImageFilter {
    /// Filter configuration.
    pub config: CurvatureFlowConfig,
}

impl CurvatureFlowImageFilter {
    /// Construct a new `CurvatureFlowImageFilter` with the given configuration.
    pub fn new(config: CurvatureFlowConfig) -> Self {
        Self { config }
    }

    /// Apply mean curvature flow to a 3-D image.
    ///
    /// Returns `anyhow::Error` if the voxel data cannot be extracted as `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let vals: &[f32] = &vals_vec;
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let slab = ny * nx;
        let mut cur: Vec<f32> = vals.to_vec();
        let dt = self.config.time_step;

        let sp = image.spacing();
        let inv_sp = [
            (1.0 / sp[2]) as f32,
            (1.0 / sp[1]) as f32,
            (1.0 / sp[0]) as f32,
        ];

        for _ in 0..self.config.num_iterations {
            // Each output voxel reads only from the stencil neighbourhood of
            // `cur` (the 6 first-neighbours + the 3 mixed second-derivatives).
            // That makes each iteration a pure Jacobi update over the flat
            // voxel index — fully data-parallel, bit-exact to the serial
            // sweep, no per-iteration copy (the original `next.copy_from_slice
            // (&cur)` was redundant once the inner loop fanned out).
            cur = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(n, |flat| {
                let iz = flat / slab;
                let rem = flat - iz * slab;
                let iy = rem / nx;
                let ix = rem - iy * nx;

                let z = iz as isize;
                let y = iy as isize;
                let x = ix as isize;

                // Helper: clamp-boundary neighbour access
                let get = |zz: isize, yy: isize, xx: isize| -> f32 {
                    let zc = zz.clamp(0, nz as isize - 1) as usize;
                    let yc = yy.clamp(0, ny as isize - 1) as usize;
                    let xc = xx.clamp(0, nx as isize - 1) as usize;
                    cur[zc * slab + yc * nx + xc]
                };

                // First derivatives (central differences, half-step spacing)
                let ix_ = (get(z, y, x + 1) - get(z, y, x - 1)) * 0.5 * inv_sp[0];
                let iy_ = (get(z, y + 1, x) - get(z, y - 1, x)) * 0.5 * inv_sp[1];
                let iz_ = (get(z + 1, y, x) - get(z - 1, y, x)) * 0.5 * inv_sp[2];

                // Second derivatives
                let c = cur[flat];
                let ixx = (get(z, y, x + 1) - 2.0 * c + get(z, y, x - 1)) * inv_sp[0] * inv_sp[0];
                let iyy = (get(z, y + 1, x) - 2.0 * c + get(z, y - 1, x)) * inv_sp[1] * inv_sp[1];
                let izz = (get(z + 1, y, x) - 2.0 * c + get(z - 1, y, x)) * inv_sp[2] * inv_sp[2];

                // Mixed second derivatives (cross terms)
                let ixy = (get(z, y + 1, x + 1) - get(z, y + 1, x - 1) - get(z, y - 1, x + 1)
                    + get(z, y - 1, x - 1))
                    * 0.25
                    * inv_sp[0]
                    * inv_sp[1];
                let ixz = (get(z + 1, y, x + 1) - get(z + 1, y, x - 1) - get(z - 1, y, x + 1)
                    + get(z - 1, y, x - 1))
                    * 0.25
                    * inv_sp[0]
                    * inv_sp[2];
                let iyz = (get(z + 1, y + 1, x) - get(z + 1, y - 1, x) - get(z - 1, y + 1, x)
                    + get(z - 1, y - 1, x))
                    * 0.25
                    * inv_sp[1]
                    * inv_sp[2];

                // Mean curvature numerator N
                let num = ixx * (iy_ * iy_ + iz_ * iz_)
                    + iyy * (ix_ * ix_ + iz_ * iz_)
                    + izz * (ix_ * ix_ + iy_ * iy_)
                    - 2.0 * ix_ * iy_ * ixy
                    - 2.0 * ix_ * iz_ * ixz
                    - 2.0 * iy_ * iz_ * iyz;

                // ITK CurvatureFlow speed = |∇I|·κ = N / |∇I|², NOT pure
                // κ = N / |∇I|³. The |∇I| factor cancels the flat-region
                // singularity (κ alone is 0/0 where ∇I → 0 and blows up).
                let grad_sq = ix_ * ix_ + iy_ * iy_ + iz_ * iz_;

                let speed = if grad_sq > GRAD_MAG_EPSILON {
                    num / grad_sq
                } else {
                    0.0
                };

                c + dt * speed
            });
        }

        Ok(rebuild(cur, dims, image))
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_curvature_flow.rs"]
mod tests;
