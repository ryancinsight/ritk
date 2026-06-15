//! Curvature anisotropic diffusion filter.
//!
//! # Mathematical Specification
//!
//! Implements ITK's `CurvatureNDAnisotropicDiffusionFunction` — the Modified
//! Curvature Diffusion Equation (MCDE) of Whitaker & Xue (2001):
//!
//!   ∂I/∂t = |∇I| · ∇·( c(|∇I|) · ∇I/|∇I| )
//!
//! the curvature-driven analogue of Perona-Malik: the conductance-weighted flux
//! is normalised by the gradient magnitude (so each level set evolves by its own
//! conductance-modulated mean curvature) and the divergence is multiplied by an
//! upwind approximation of |∇I|.
//!
//! # Discretisation (per voxel, ITK-exact)
//!
//! With spacing-scaled forward/backward/central differences `dx_fwd[i]`,
//! `dx_bwd[i]`, `dx[i]` (derivative scaled by `1/spacing[i]`):
//!
//! For each dimension `i`, the gradient magnitude squared at the `±i` faces is
//! the face-normal difference plus the averaged tangential central differences:
//!
//!   gms±  = dx_{fwd,bwd}[i]² + Σ_{j≠i} ¼·(dx[j] + dx[j]^{±i})²
//!
//! The conductance is `c± = exp(gms± / m_K)` with the average-gradient-magnitude
//! `m_K = avgGradMagSq · K² · −2` (recomputed each iteration, shared with the
//! gradient filter), and the normalised flux is
//!
//!   speed = Σ_i [ (dx_fwd[i]/√(ε+gms⁺))·c⁺ − (dx_bwd[i]/√(ε+gms⁻))·c⁻ ],  ε = 1e-10
//!
//! Finally the update is `√(propagation_gradient) · speed`, where the upwind
//! `propagation_gradient` selects forward/backward squared differences by the
//! sign of `speed`. Boundary conditions are ZeroFluxNeumann (index-clamp).
//!
//! # Invariants
//! - Constant image: all differences 0 → update 0 → image unchanged.
//!
//! # References
//! - Whitaker, R. & Xue, X. (2001). Variable-conductance, level-set curvature
//!   for image denoising. *Proc. ICIP*.
//! - ITK `itkCurvatureNDAnisotropicDiffusionFunction.hxx`.

use super::{central_diff, clamp_at};
use burn::tensor::backend::Backend;
use ritk_image::Image;
use ritk_tensor_ops::{extract_vec, rebuild};

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for the curvature anisotropic diffusion filter.
#[derive(Debug, Clone)]
pub struct CurvatureConfig {
    /// Number of explicit Euler time steps to perform.
    pub num_iterations: usize,
    /// Time step Δt.  Must satisfy Δt ≤ 1/6 for unit spacing.
    /// Default: 0.0625 = 1/16.
    pub time_step: f32,
    /// Conductance parameter K. Larger K → more isotropic smoothing.
    /// ITK default: 3.0.
    pub conductance: f32,
}

impl Default for CurvatureConfig {
    fn default() -> Self {
        Self {
            num_iterations: 20,
            time_step: 1.0 / 16.0,
            conductance: 3.0,
        }
    }
}

/// Curvature anisotropic diffusion filter (mean curvature motion of level sets).
///
/// Smooths images by evolving each iso-intensity level set according to its own
/// mean curvature.  Geometry of edges is preserved while noise is removed.
#[derive(Debug, Clone)]
pub struct CurvatureAnisotropicDiffusionFilter {
    /// Algorithm configuration.
    pub config: CurvatureConfig,
}

impl CurvatureAnisotropicDiffusionFilter {
    /// Create a filter with the given configuration.
    pub fn new(config: CurvatureConfig) -> Self {
        Self { config }
    }

    /// Apply the filter to `image`, returning a smoothed copy.
    ///
    /// # Errors
    /// Returns an error if the image tensor cannot be converted to `f32`.
    pub fn apply<B: Backend>(&self, image: &Image<B, 3>) -> anyhow::Result<Image<B, 3>> {
        let (vals_vec, dims) = extract_vec(image)?;
        let vals = &vals_vec;
        let sp = image.spacing();
        let spacing = [sp[0], sp[1], sp[2]];

        let result = curvature_diffuse(vals, dims, spacing, &self.config);

        Ok(rebuild(result, dims, image))
    }
}

// ── Core computation (ITK CurvatureNDAnisotropicDiffusionFunction) ────────────

/// Gradient-magnitude floor (ITK `m_MIN_NORM`).
const MIN_NORM: f64 = 1.0e-10;

/// Run the MCDE explicit Euler curvature diffusion for the requested iterations.
///
/// Per iteration: recompute `m_K = avgGradMagSq · K² · −2` (shared with the
/// gradient filter), then apply the conductance-weighted, gradient-normalised
/// curvature flux multiplied by the upwind `√(propagation_gradient)`. Boundary
/// conditions are ZeroFluxNeumann (index-clamp); derivatives are spacing-scaled.
#[allow(clippy::needless_range_loop)]
fn curvature_diffuse(
    data: &[f32],
    dims: [usize; 3],
    spacing: [f64; 3],
    config: &CurvatureConfig,
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut cur = data.to_vec();
    let mut nxt = vec![0.0f32; n];

    let dt = config.time_step as f64;
    let cond = config.conductance as f64;
    let inv_sp = [1.0 / spacing[0], 1.0 / spacing[1], 1.0 / spacing[2]];
    let inv_2sp = [0.5 / spacing[0], 0.5 / spacing[1], 0.5 / spacing[2]];

    for _ in 0..config.num_iterations {
        // Average gradient magnitude squared → m_K (identical to the gradient
        // anisotropic filter; ITK shares this via the base diffusion function).
        let mut sum_gms = 0.0_f64;
        for z in 0..nz as isize {
            for y in 0..ny as isize {
                for x in 0..nx as isize {
                    let g0 = central_diff(&cur, dims, inv_2sp, 0, z, y, x);
                    let g1 = central_diff(&cur, dims, inv_2sp, 1, z, y, x);
                    let g2 = central_diff(&cur, dims, inv_2sp, 2, z, y, x);
                    sum_gms += g0 * g0 + g1 * g1 + g2 * g2;
                }
            }
        }
        let m_k = (sum_gms / n as f64) * cond * cond * -2.0;

        for z in 0..nz as isize {
            for y in 0..ny as isize {
                for x in 0..nx as isize {
                    let center = clamp_at(&cur, dims, z, y, x);

                    // Spacing-scaled forward/backward/central differences per dim.
                    let mut dxf = [0.0_f64; 3];
                    let mut dxb = [0.0_f64; 3];
                    let mut dxc = [0.0_f64; 3];
                    for i in 0..3 {
                        let (fp, fm) = match i {
                            0 => (clamp_at(&cur, dims, z + 1, y, x), clamp_at(&cur, dims, z - 1, y, x)),
                            1 => (clamp_at(&cur, dims, z, y + 1, x), clamp_at(&cur, dims, z, y - 1, x)),
                            _ => (clamp_at(&cur, dims, z, y, x + 1), clamp_at(&cur, dims, z, y, x - 1)),
                        };
                        dxf[i] = (fp - center) * inv_sp[i];
                        dxb[i] = (center - fm) * inv_sp[i];
                        dxc[i] = central_diff(&cur, dims, inv_2sp, i, z, y, x);
                    }

                    // Conductance-weighted, gradient-normalised curvature flux.
                    let mut speed = 0.0_f64;
                    for i in 0..3 {
                        let mut gms = dxf[i] * dxf[i];
                        let mut gms_d = dxb[i] * dxb[i];
                        for j in 0..3 {
                            if j == i {
                                continue;
                            }
                            let (aug, dim_) = match i {
                                0 => (
                                    central_diff(&cur, dims, inv_2sp, j, z + 1, y, x),
                                    central_diff(&cur, dims, inv_2sp, j, z - 1, y, x),
                                ),
                                1 => (
                                    central_diff(&cur, dims, inv_2sp, j, z, y + 1, x),
                                    central_diff(&cur, dims, inv_2sp, j, z, y - 1, x),
                                ),
                                _ => (
                                    central_diff(&cur, dims, inv_2sp, j, z, y, x + 1),
                                    central_diff(&cur, dims, inv_2sp, j, z, y, x - 1),
                                ),
                            };
                            let sf = dxc[j] + aug;
                            let sb = dxc[j] + dim_;
                            gms += 0.25 * sf * sf;
                            gms_d += 0.25 * sb * sb;
                        }
                        let grad_mag = (MIN_NORM + gms).sqrt();
                        let grad_mag_d = (MIN_NORM + gms_d).sqrt();
                        let (cx, cxd) = if m_k == 0.0 {
                            (0.0, 0.0)
                        } else {
                            ((gms / m_k).exp(), (gms_d / m_k).exp())
                        };
                        speed += (dxf[i] / grad_mag) * cx - (dxb[i] / grad_mag_d) * cxd;
                    }

                    // Upwind |∇I| (propagation gradient), selected by sign of speed.
                    let mut prop = 0.0_f64;
                    if speed > 0.0 {
                        for i in 0..3 {
                            prop += dxb[i].min(0.0).powi(2) + dxf[i].max(0.0).powi(2);
                        }
                    } else {
                        for i in 0..3 {
                            prop += dxb[i].max(0.0).powi(2) + dxf[i].min(0.0).powi(2);
                        }
                    }
                    let update = prop.sqrt() * speed;

                    let p = (z as usize) * ny * nx + (y as usize) * nx + (x as usize);
                    nxt[p] = (center + dt * update) as f32;
                }
            }
        }
        cur.copy_from_slice(&nxt);
    }
    cur
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_curvature.rs"]
mod tests;
