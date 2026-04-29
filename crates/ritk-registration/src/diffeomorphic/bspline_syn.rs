//! B-Spline Symmetric Normalization (BSplineSyN) registration.
//!
//! # Mathematical Specification
//!
//! BSplineSyN parameterises the stationary velocity fields `v₁, v₂` of the
//! SyN framework using cubic B-spline control-point lattices instead of dense
//! voxel grids. This reduces the number of free parameters and provides
//! intrinsic C²-smooth velocity fields.
//!
//! ## Cubic B-Spline Basis
//!
//! The uniform cubic B-spline basis over a knot span with parameter `u ∈ [0,1)`:
//!
//! ```text
//! B(u) = (1/6) [u³ u² u 1] · [[-1, 3,-3, 1],
//!                               [ 3,-6, 3, 0],
//!                               [-3, 0, 3, 0],
//!                               [ 1, 4, 1, 0]]^T
//! ```
//!
//! Yielding four basis functions:
//!   - `B₀(u) = (1 − u)³ / 6`
//!   - `B₁(u) = (3u³ − 6u² + 4) / 6`
//!   - `B₂(u) = (−3u³ + 3u² + 3u + 1) / 6`
//!   - `B₃(u) = u³ / 6`
//!
//! **Partition of unity**: `Σₖ Bₖ(u) = 1  ∀ u ∈ [0,1]` (theorem for uniform
//! B-splines; verified in tests).
//!
//! ## Dense Field Evaluation
//!
//! For voxel position `(z, y, x)` and control-point spacing `s`:
//!
//!   `t_d = d / s_d`,  `span_d = ⌊t_d⌋`,  `u_d = t_d − span_d`
//!
//!   `v(z,y,x) = Σ_{l,m,n=0}^{3} Bₗ(u_z) Bₘ(u_y) Bₙ(u_x) · cp[span_z+l, span_y+m, span_x+n]`
//!
//! ## Per-Iteration Update
//!
//! 1. Evaluate dense velocity fields from CPs via cubic B-spline
//! 2. Compute `φ₁ = exp(v₁)`, `φ₂ = exp(v₂)` via scaling-and-squaring
//! 3. Warp images, compute CC forces (dense)
//! 4. Optionally smooth dense forces with Gaussian kernel
//! 5. Accumulate forces at CPs (weighted average over B-spline support)
//! 6. Regularise CPs via discrete Laplacian (bending energy penalty):
//!    `cp ← cp + force + λ · Δcp`
//! 7. Check convergence via CC variance window
//!
//! ## Bending Energy Regularisation
//!
//! The discrete 6-connected Laplacian on the CP lattice:
//!
//!   `Δcp[i,j,k] = Σ_neighbours cp[n] − 6·cp[i,j,k]`
//!
//! drives CPs toward the local average, penalising curvature. Weight `λ`
//! controls the regularisation strength.
//!
//! # References
//! - Tustison, N. J. & Avants, B. B. (2013). Explicit B-spline regularization
//!   in diffeomorphic image registration. *Frontiers in Neuroinformatics* 7:39.
//! - Rueckert, D., Sonoda, L. I., Hayes, C., Hill, D. L. G., Leach, M. O. &
//!   Hawkes, D. J. (1999). Nonrigid registration using free-form deformations:
//!   application to breast MR images. *IEEE TMI* 18(8):712–721.

use std::collections::VecDeque;

use crate::deformable_field_ops::{
    compute_gradient, flat, gaussian_smooth_inplace, scaling_and_squaring, warp_image,
};
use crate::error::RegistrationError;

// ── Public types ──────────────────────────────────────────────────────────────

/// Configuration for BSplineSyN registration.
#[derive(Debug, Clone)]
pub struct BSplineSyNConfig {
    /// Maximum number of iterations.
    pub max_iterations: usize,
    /// Control-point spacing in voxels per axis `[sz, sy, sx]`.
    pub control_spacing: [usize; 3],
    /// Gaussian σ (voxels) applied to dense CC forces before CP accumulation.
    pub sigma_smooth: f64,
    /// Stop when CC variance over the convergence window falls below this.
    pub convergence_threshold: f64,
    /// Number of recent CC values for convergence checking.
    pub convergence_window: usize,
    /// Number of scaling-and-squaring steps for `exp(v)`.
    pub n_squarings: usize,
    /// Radius of local CC window (voxels).
    pub cc_window_radius: usize,
    /// Maximum per-step displacement (voxels) used to normalise the CC gradient
    /// before accumulating into the velocity field.  Mirrors the ANTs
    /// `gradientStep` parameter.  Default: 0.25.
    pub gradient_step: f64,
    /// Bending energy regularisation weight (Laplacian smoothing on CPs).
    pub regularization_weight: f64,
}

/// Result returned by [`BSplineSyNRegistration::register`].
#[derive(Debug, Clone)]
pub struct BSplineSyNResult {
    /// Forward dense velocity field `v₁` components (fixed→midpoint).
    pub forward_field: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Inverse dense velocity field `v₂` components (moving→midpoint).
    pub inverse_field: (Vec<f32>, Vec<f32>, Vec<f32>),
    /// Fixed image warped to the midpoint by `φ₁ = exp(v₁)`.
    pub warped_fixed: Vec<f32>,
    /// Moving image warped to the midpoint by `φ₂ = exp(v₂)`.
    pub warped_moving: Vec<f32>,
    /// Final mean local CC value (higher is better; 1.0 = perfect alignment).
    pub final_cc: f64,
    /// Number of iterations actually performed.
    pub num_iterations: usize,
}

/// BSplineSyN registration engine.
///
/// Represents velocity fields via cubic B-spline control-point lattices,
/// providing intrinsic C²-smoothness and reduced parameter count compared to
/// dense SyN.
#[derive(Debug, Clone)]
pub struct BSplineSyNRegistration {
    /// Algorithm configuration.
    pub config: BSplineSyNConfig,
}

impl BSplineSyNRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: BSplineSyNConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` using BSplineSyN with local CC metric.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `[f32]` in Z-major order.
    /// - `moving`  — moving image, same shape as `fixed`.
    /// - `dims`    — `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] on dimension mismatch or invalid config.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<BSplineSyNResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        if fixed.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "fixed length {} != dims product {}",
                fixed.len(),
                n
            )));
        }
        if moving.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "moving length {} != dims product {}",
                moving.len(),
                n
            )));
        }
        for d in 0..3 {
            if self.config.control_spacing[d] == 0 {
                return Err(RegistrationError::InvalidConfiguration(format!(
                    "control_spacing[{}] must be >= 1",
                    d
                )));
            }
        }

        let cs = self.config.control_spacing;
        let cp_d = [
            cp_count(nz, cs[0]),
            cp_count(ny, cs[1]),
            cp_count(nx, cs[2]),
        ];
        let cp_n = cp_d[0] * cp_d[1] * cp_d[2];

        // Forward velocity CPs (fixed → midpoint).
        let mut cp1z = vec![0.0_f32; cp_n];
        let mut cp1y = vec![0.0_f32; cp_n];
        let mut cp1x = vec![0.0_f32; cp_n];
        // Inverse velocity CPs (moving → midpoint).
        let mut cp2z = vec![0.0_f32; cp_n];
        let mut cp2y = vec![0.0_f32; cp_n];
        let mut cp2x = vec![0.0_f32; cp_n];

        let mut cc_history: VecDeque<f64> = VecDeque::new();
        let mut final_cc = 0.0_f64;
        let mut iter = 0usize;
        let r = self.config.cc_window_radius;
        let rw = self.config.regularization_weight as f32;
        let sigma = self.config.sigma_smooth;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // 1. Evaluate dense velocity fields from CPs.
            let v1z = evaluate_dense(&cp1z, cp_d, dims, cs);
            let v1y = evaluate_dense(&cp1y, cp_d, dims, cs);
            let v1x = evaluate_dense(&cp1x, cp_d, dims, cs);
            let v2z = evaluate_dense(&cp2z, cp_d, dims, cs);
            let v2y = evaluate_dense(&cp2y, cp_d, dims, cs);
            let v2x = evaluate_dense(&cp2x, cp_d, dims, cs);

            // 2. Exponential maps.
            let (phi1z, phi1y, phi1x) =
                scaling_and_squaring(&v1z, &v1y, &v1x, dims, self.config.n_squarings);
            let (phi2z, phi2y, phi2x) =
                scaling_and_squaring(&v2z, &v2y, &v2x, dims, self.config.n_squarings);

            // 3. Warp images and compute gradients.
            let i_w = warp_image(fixed, dims, &phi1z, &phi1y, &phi1x);
            let j_w = warp_image(moving, dims, &phi2z, &phi2y, &phi2x);
            let (giz, giy, gix) = compute_gradient(&i_w, dims, spacing);
            let (gjz, gjy, gjx) = compute_gradient(&j_w, dims, spacing);

            // 4. Dense CC forces.
            let (u1z, u1y, u1x) = cc_forces(&i_w, &j_w, &giz, &giy, &gix, dims, r);
            let (u2z, u2y, u2x) = cc_forces(&j_w, &i_w, &gjz, &gjy, &gjx, dims, r);

            // Normalise per-step displacement to `gradient_step` voxels (inf-norm).
            let max_u1 = u1z
                .iter()
                .chain(u1y.iter())
                .chain(u1x.iter())
                .map(|&v| (v as f64).abs())
                .fold(0.0_f64, f64::max);
            let (mut u1z, mut u1y, mut u1x) = (u1z, u1y, u1x);
            if max_u1 > 1e-10 {
                let s = (self.config.gradient_step / max_u1) as f32;
                u1z.iter_mut().for_each(|v| *v *= s);
                u1y.iter_mut().for_each(|v| *v *= s);
                u1x.iter_mut().for_each(|v| *v *= s);
            }
            let max_u2 = u2z
                .iter()
                .chain(u2y.iter())
                .chain(u2x.iter())
                .map(|&v| (v as f64).abs())
                .fold(0.0_f64, f64::max);
            let (mut u2z, mut u2y, mut u2x) = (u2z, u2y, u2x);
            if max_u2 > 1e-10 {
                let s = (self.config.gradient_step / max_u2) as f32;
                u2z.iter_mut().for_each(|v| *v *= s);
                u2y.iter_mut().for_each(|v| *v *= s);
                u2x.iter_mut().for_each(|v| *v *= s);
            }

            // 5. Smooth dense forces (optional).
            if sigma > 0.0 {
                gaussian_smooth_inplace(&mut u1z, dims, sigma);
                gaussian_smooth_inplace(&mut u1y, dims, sigma);
                gaussian_smooth_inplace(&mut u1x, dims, sigma);
                gaussian_smooth_inplace(&mut u2z, dims, sigma);
                gaussian_smooth_inplace(&mut u2y, dims, sigma);
                gaussian_smooth_inplace(&mut u2x, dims, sigma);
            }

            // 6. Accumulate forces at CPs.
            let d1z = accumulate_to_cp(&u1z, dims, cp_d, cs);
            let d1y = accumulate_to_cp(&u1y, dims, cp_d, cs);
            let d1x = accumulate_to_cp(&u1x, dims, cp_d, cs);
            let d2z = accumulate_to_cp(&u2z, dims, cp_d, cs);
            let d2y = accumulate_to_cp(&u2y, dims, cp_d, cs);
            let d2x = accumulate_to_cp(&u2x, dims, cp_d, cs);

            // 7. Bending energy regularisation + update.
            let l1z = cp_laplacian(&cp1z, cp_d);
            let l1y = cp_laplacian(&cp1y, cp_d);
            let l1x = cp_laplacian(&cp1x, cp_d);
            let l2z = cp_laplacian(&cp2z, cp_d);
            let l2y = cp_laplacian(&cp2y, cp_d);
            let l2x = cp_laplacian(&cp2x, cp_d);
            for i in 0..cp_n {
                cp1z[i] += d1z[i] + rw * l1z[i];
                cp1y[i] += d1y[i] + rw * l1y[i];
                cp1x[i] += d1x[i] + rw * l1x[i];
                cp2z[i] += d2z[i] + rw * l2z[i];
                cp2y[i] += d2y[i] + rw * l2y[i];
                cp2x[i] += d2x[i] + rw * l2x[i];
            }

            // 8. Convergence check.
            final_cc = mean_local_cc(&i_w, &j_w, dims, r);
            cc_history.push_back(final_cc);
            if cc_history.len() > self.config.convergence_window {
                cc_history.pop_front();
            }
            if cc_history.len() == self.config.convergence_window {
                let mu = cc_history.iter().sum::<f64>() / cc_history.len() as f64;
                let var = cc_history.iter().map(|&v| (v - mu).powi(2)).sum::<f64>()
                    / cc_history.len() as f64;
                if var < self.config.convergence_threshold {
                    break;
                }
            }
        }

        // Final dense fields and warps.
        let v1z = evaluate_dense(&cp1z, cp_d, dims, cs);
        let v1y = evaluate_dense(&cp1y, cp_d, dims, cs);
        let v1x = evaluate_dense(&cp1x, cp_d, dims, cs);
        let v2z = evaluate_dense(&cp2z, cp_d, dims, cs);
        let v2y = evaluate_dense(&cp2y, cp_d, dims, cs);
        let v2x = evaluate_dense(&cp2x, cp_d, dims, cs);

        let (phi1z, phi1y, phi1x) =
            scaling_and_squaring(&v1z, &v1y, &v1x, dims, self.config.n_squarings);
        let (phi2z, phi2y, phi2x) =
            scaling_and_squaring(&v2z, &v2y, &v2x, dims, self.config.n_squarings);

        Ok(BSplineSyNResult {
            forward_field: (v1z, v1y, v1x),
            inverse_field: (v2z, v2y, v2x),
            warped_fixed: warp_image(fixed, dims, &phi1z, &phi1y, &phi1x),
            warped_moving: warp_image(moving, dims, &phi2z, &phi2y, &phi2x),
            final_cc,
            num_iterations: iter,
        })
    }
}

// ── B-Spline primitives ───────────────────────────────────────────────────────

/// Evaluate the `k`-th uniform cubic B-spline basis function at parameter
/// `u ∈ [0, 1]`.
///
/// # Partition of unity
/// `Σ_{k=0}^{3} Bₖ(u) = 1` for all `u ∈ [0, 1]`.
#[inline]
fn bspline_basis(k: usize, u: f64) -> f64 {
    let u2 = u * u;
    let u3 = u2 * u;
    match k {
        0 => (1.0 - 3.0 * u + 3.0 * u2 - u3) / 6.0,
        1 => (4.0 - 6.0 * u2 + 3.0 * u3) / 6.0,
        2 => (1.0 + 3.0 * u + 3.0 * u2 - 3.0 * u3) / 6.0,
        3 => u3 / 6.0,
        _ => 0.0,
    }
}

/// Number of control points along one axis for image dimension `dim` and
/// control-point spacing `spacing`.
///
/// Formula: `⌊(dim − 1) / spacing⌋ + 4`, ensuring the cubic B-spline support
/// (4 CPs per knot span) covers the entire image domain.
#[inline]
fn cp_count(dim: usize, spacing: usize) -> usize {
    if dim <= 1 {
        return 4;
    }
    (dim - 1) / spacing + 4
}

/// Evaluate a single dense displacement-field component from its
/// control-point lattice via cubic B-spline evaluation.
///
/// For each voxel, the displacement is the tensor-product sum over the
/// 4×4×4 local CP neighbourhood weighted by the B-spline basis.
fn evaluate_dense(
    cp: &[f32],
    cp_dims: [usize; 3],
    dims: [usize; 3],
    control_spacing: [usize; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut out = vec![0.0_f32; n];

    for iz in 0..nz {
        let tz = iz as f64 / control_spacing[0] as f64;
        let sz = (tz.floor() as usize).min(cp_dims[0].saturating_sub(4));
        let uz = tz - sz as f64;

        for iy in 0..ny {
            let ty = iy as f64 / control_spacing[1] as f64;
            let sy = (ty.floor() as usize).min(cp_dims[1].saturating_sub(4));
            let uy = ty - sy as f64;

            for ix in 0..nx {
                let tx = ix as f64 / control_spacing[2] as f64;
                let sx = (tx.floor() as usize).min(cp_dims[2].saturating_sub(4));
                let ux = tx - sx as f64;

                let mut val = 0.0_f64;
                for kz in 0..4 {
                    let bz = bspline_basis(kz, uz);
                    for ky in 0..4 {
                        let bzy = bz * bspline_basis(ky, uy);
                        for kx in 0..4 {
                            let w = bzy * bspline_basis(kx, ux);
                            let ci = flat(sz + kz, sy + ky, sx + kx, cp_dims[1], cp_dims[2]);
                            val += w * cp[ci] as f64;
                        }
                    }
                }
                out[flat(iz, iy, ix, ny, nx)] = val as f32;
            }
        }
    }
    out
}

/// Accumulate dense voxel-wise forces to the control-point lattice.
///
/// Each CP receives a weighted average of the forces from voxels in its
/// B-spline support region:
///
///   `cp_force[c] = Σ_v w(v,c) · force[v]  /  Σ_v w(v,c)`
///
/// where `w(v,c) = Bₗ(u_z) Bₘ(u_y) Bₙ(u_x)` is the tensor-product weight.
fn accumulate_to_cp(
    force: &[f32],
    dims: [usize; 3],
    cp_dims: [usize; 3],
    control_spacing: [usize; 3],
) -> Vec<f32> {
    let cp_n = cp_dims[0] * cp_dims[1] * cp_dims[2];
    let mut accum = vec![0.0_f64; cp_n];
    let mut weight = vec![0.0_f64; cp_n];
    let [nz, ny, nx] = dims;

    for iz in 0..nz {
        let tz = iz as f64 / control_spacing[0] as f64;
        let sz = (tz.floor() as usize).min(cp_dims[0].saturating_sub(4));
        let uz = tz - sz as f64;

        for iy in 0..ny {
            let ty = iy as f64 / control_spacing[1] as f64;
            let sy = (ty.floor() as usize).min(cp_dims[1].saturating_sub(4));
            let uy = ty - sy as f64;

            for ix in 0..nx {
                let tx = ix as f64 / control_spacing[2] as f64;
                let sx = (tx.floor() as usize).min(cp_dims[2].saturating_sub(4));
                let ux = tx - sx as f64;

                let fv = force[flat(iz, iy, ix, ny, nx)] as f64;

                for kz in 0..4 {
                    let bz = bspline_basis(kz, uz);
                    for ky in 0..4 {
                        let bzy = bz * bspline_basis(ky, uy);
                        for kx in 0..4 {
                            let w = bzy * bspline_basis(kx, ux);
                            let ci = flat(sz + kz, sy + ky, sx + kx, cp_dims[1], cp_dims[2]);
                            accum[ci] += w * fv;
                            weight[ci] += w;
                        }
                    }
                }
            }
        }
    }

    accum
        .iter()
        .zip(weight.iter())
        .map(|(&a, &w)| if w > 1e-12 { (a / w) as f32 } else { 0.0 })
        .collect()
}

/// Discrete 6-connected Laplacian on the control-point lattice.
///
/// `Δcp[i,j,k] = Σ_face_neighbours cp[n] − count · cp[i,j,k]`
///
/// At boundaries, missing neighbours are omitted and `count` is reduced
/// accordingly (Neumann-like boundary condition).
fn cp_laplacian(cp: &[f32], cp_dims: [usize; 3]) -> Vec<f32> {
    let [cnz, cny, cnx] = cp_dims;
    let cn = cnz * cny * cnx;
    let mut lap = vec![0.0_f32; cn];

    for ci in 0..cnz {
        for cj in 0..cny {
            for ck in 0..cnx {
                let idx = flat(ci, cj, ck, cny, cnx);
                let c = cp[idx] as f64;
                let mut sum = 0.0_f64;
                let mut cnt = 0u32;

                if ci > 0 {
                    sum += cp[flat(ci - 1, cj, ck, cny, cnx)] as f64;
                    cnt += 1;
                }
                if ci + 1 < cnz {
                    sum += cp[flat(ci + 1, cj, ck, cny, cnx)] as f64;
                    cnt += 1;
                }
                if cj > 0 {
                    sum += cp[flat(ci, cj - 1, ck, cny, cnx)] as f64;
                    cnt += 1;
                }
                if cj + 1 < cny {
                    sum += cp[flat(ci, cj + 1, ck, cny, cnx)] as f64;
                    cnt += 1;
                }
                if ck > 0 {
                    sum += cp[flat(ci, cj, ck - 1, cny, cnx)] as f64;
                    cnt += 1;
                }
                if ck + 1 < cnx {
                    sum += cp[flat(ci, cj, ck + 1, cny, cnx)] as f64;
                    cnt += 1;
                }

                lap[idx] = (sum - cnt as f64 * c) as f32;
            }
        }
    }
    lap
}

// ── CC metric (local reimplementation — private in parent module) ─────────────

/// Compute local CC gradient forces (Avants 2008, eq. 10).
///
/// For each voxel `p` with window `W` of radius `r`, gradient ascent on local CC:
///
/// ```text
/// force_scale(p) = (J_w(p)−μ_J) / √(σ_I²·σ_J²)  −  CC · (I_w(p)−μ_I) / σ_I²
/// f_k(p)        = force_scale(p) · ∇_k I_w(p)
/// ```
///
/// where `CC = Σ(I_w−μ_I)(J_w−μ_J) / √(σ_I²·σ_J²)` over the local window.
fn cc_forces(
    i_w: &[f32],
    j_w: &[f32],
    gi_z: &[f32],
    gi_y: &[f32],
    gi_x: &[f32],
    dims: [usize; 3],
    radius: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let [nz, ny, nx] = dims;
    let n = nz * ny * nx;
    let mut fz = vec![0.0_f32; n];
    let mut fy = vec![0.0_f32; n];
    let mut fx = vec![0.0_f32; n];
    let r = radius as isize;

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let (mut si, mut sj, mut cnt) = (0.0_f64, 0.0_f64, 0u32);
                for dz in -r..=r {
                    let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            let qi = flat(qz, qy, qx, ny, nx);
                            si += i_w[qi] as f64;
                            sj += j_w[qi] as f64;
                            cnt += 1;
                        }
                    }
                }
                if cnt == 0 {
                    continue;
                }
                let (mu_i, mu_j) = (si / cnt as f64, sj / cnt as f64);

                let (mut num, mut vi, mut vj) = (0.0_f64, 0.0_f64, 0.0_f64);
                for dz in -r..=r {
                    let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            let qi = flat(qz, qy, qx, ny, nx);
                            let di = i_w[qi] as f64 - mu_i;
                            let dj = j_w[qi] as f64 - mu_j;
                            num += di * dj;
                            vi += di * di;
                            vj += dj * dj;
                        }
                    }
                }
                if vi < 1e-10 {
                    continue;
                }

                let fi = flat(iz, iy, ix, ny, nx);
                let iw_c = i_w[fi] as f64 - mu_i;
                let jw_c = j_w[fi] as f64 - mu_j;
                // Avants 2008, eq. 10 — gradient ascent on local CC.
                // ∂CC/∂v₁_k(x) = [(J_w−μ_J)/√(σ_I²·σ_J²) − CC·(I_w−μ_I)/σ_I²] · ∇_k I_w
                let denom = (vi * vj).sqrt() + 1e-10;
                let cc = num / denom;
                let force_scale = jw_c / denom - cc * iw_c / (vi + 1e-10);
                fz[fi] = (force_scale * gi_z[fi] as f64) as f32;
                fy[fi] = (force_scale * gi_y[fi] as f64) as f32;
                fx[fi] = (force_scale * gi_x[fi] as f64) as f32;
            }
        }
    }
    (fz, fy, fx)
}

/// Compute mean local CC over all voxels (same window radius as CC forces).
fn mean_local_cc(i_w: &[f32], j_w: &[f32], dims: [usize; 3], radius: usize) -> f64 {
    let [nz, ny, nx] = dims;
    let r = radius as isize;
    let (mut total, mut count) = (0.0_f64, 0u64);

    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                let (mut si, mut sj, mut nw) = (0.0_f64, 0.0_f64, 0u32);
                for dz in -r..=r {
                    let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            let qi = flat(qz, qy, qx, ny, nx);
                            si += i_w[qi] as f64;
                            sj += j_w[qi] as f64;
                            nw += 1;
                        }
                    }
                }
                let (mi, mj) = (si / nw as f64, sj / nw as f64);

                let (mut num, mut di2, mut dj2) = (0.0_f64, 0.0_f64, 0.0_f64);
                for dz in -r..=r {
                    let qz = (iz as isize + dz).max(0).min(nz as isize - 1) as usize;
                    for dy in -r..=r {
                        let qy = (iy as isize + dy).max(0).min(ny as isize - 1) as usize;
                        for dx in -r..=r {
                            let qx = (ix as isize + dx).max(0).min(nx as isize - 1) as usize;
                            let qi = flat(qz, qy, qx, ny, nx);
                            let a = i_w[qi] as f64 - mi;
                            let b = j_w[qi] as f64 - mj;
                            num += a * b;
                            di2 += a * a;
                            dj2 += b * b;
                        }
                    }
                }
                let denom = (di2 * dj2).sqrt();
                if denom > 1e-10 {
                    total += num / denom;
                    count += 1;
                }
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::{
        accumulate_to_cp, bspline_basis, cp_count, cp_laplacian, evaluate_dense, mean_local_cc,
        BSplineSyNConfig, BSplineSyNRegistration,
    };
    use crate::deformable_field_ops::flat;

    fn make_default_config() -> BSplineSyNConfig {
        BSplineSyNConfig {
            max_iterations: 15,
            control_spacing: [3, 3, 3],
            sigma_smooth: 1.5,
            convergence_threshold: 1e-7,
            convergence_window: 10,
            n_squarings: 6,
            cc_window_radius: 2,
            gradient_step: 0.25,
            regularization_weight: 0.01,
        }
    }

    /// Smooth test image: `I[z,y,x] = sin(π·z/nz) · cos(π·y/ny) · (x + 1)`.
    /// Analytically derived to produce non-trivial gradients in all three axes.
    fn make_test_image(dims: [usize; 3]) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        (0..nz * ny * nx)
            .map(|fi| {
                let ix = fi % nx;
                let iy = (fi / nx) % ny;
                let iz = fi / (ny * nx);
                let sz = std::f32::consts::PI * iz as f32 / nz as f32;
                let sy = std::f32::consts::PI * iy as f32 / ny as f32;
                sz.sin() * sy.cos() * (ix as f32 + 1.0)
            })
            .collect()
    }

    /// Shift image +`shift` voxels in x with zero-padding at the left boundary.
    fn translate_x(data: &[f32], dims: [usize; 3], shift: usize) -> Vec<f32> {
        let [nz, ny, nx] = dims;
        let mut out = vec![0.0_f32; nz * ny * nx];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in shift..nx {
                    out[iz * ny * nx + iy * nx + ix] = data[iz * ny * nx + iy * nx + (ix - shift)];
                }
            }
        }
        out
    }

    // ── B-spline basis tests ──────────────────────────────────────────────────

    /// Partition of unity: `Σ_{k=0}^{3} Bₖ(u) = 1` for all `u ∈ [0, 1]`.
    /// Verified at 101 uniformly-spaced parameter values (analytically exact
    /// for uniform cubic B-splines).
    #[test]
    fn bspline_basis_partition_of_unity() {
        for i in 0..=100 {
            let u = i as f64 / 100.0;
            let sum: f64 = (0..4).map(|k| bspline_basis(k, u)).sum();
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "B-spline basis at u={u} sums to {sum}, expected 1.0"
            );
        }
    }

    /// Boundary values: `B₀(0) = 1/6`, `B₁(0) = 4/6`, `B₂(0) = 1/6`, `B₃(0) = 0`.
    #[test]
    fn bspline_basis_boundary_values() {
        let tol = 1e-14;
        assert!((bspline_basis(0, 0.0) - 1.0 / 6.0).abs() < tol);
        assert!((bspline_basis(1, 0.0) - 4.0 / 6.0).abs() < tol);
        assert!((bspline_basis(2, 0.0) - 1.0 / 6.0).abs() < tol);
        assert!(bspline_basis(3, 0.0).abs() < tol);

        assert!(bspline_basis(0, 1.0).abs() < tol);
        assert!((bspline_basis(1, 1.0) - 1.0 / 6.0).abs() < tol);
        assert!((bspline_basis(2, 1.0) - 4.0 / 6.0).abs() < tol);
        assert!((bspline_basis(3, 1.0) - 1.0 / 6.0).abs() < tol);
    }

    /// All basis values are non-negative for `u ∈ [0, 1]`.
    #[test]
    fn bspline_basis_non_negative() {
        for i in 0..=1000 {
            let u = i as f64 / 1000.0;
            for k in 0..4 {
                let v = bspline_basis(k, u);
                assert!(v >= -1e-15, "B_{k}({u}) = {v} is negative");
            }
        }
    }

    // ── CP lattice tests ──────────────────────────────────────────────────────

    /// `cp_count` formula: `(dim - 1) / spacing + 4` for `dim > 1`.
    #[test]
    fn cp_count_formula() {
        assert_eq!(cp_count(10, 3), 3 + 4); // (9/3) + 4 = 7
        assert_eq!(cp_count(12, 4), 2 + 4); // (11/4) + 4 = 6
        assert_eq!(cp_count(1, 5), 4); // edge case: single voxel
        assert_eq!(cp_count(13, 3), 4 + 4); // (12/3) + 4 = 8
    }

    /// Evaluating a constant CP lattice produces a constant dense field equal
    /// to the CP value (by partition of unity).
    #[test]
    fn constant_cp_produces_constant_field() {
        let dims = [8, 8, 8];
        let cs = [3, 3, 3];
        let cp_dims = [cp_count(8, 3), cp_count(8, 3), cp_count(8, 3)];
        let cp_n = cp_dims[0] * cp_dims[1] * cp_dims[2];
        let cp = vec![5.0_f32; cp_n];
        let dense = evaluate_dense(&cp, cp_dims, dims, cs);
        assert_eq!(dense.len(), 8 * 8 * 8);
        for (i, &v) in dense.iter().enumerate() {
            assert!((v - 5.0).abs() < 1e-5, "voxel {i}: expected 5.0, got {v}");
        }
    }

    /// Laplacian of a constant CP lattice is zero (no curvature).
    #[test]
    fn laplacian_constant_cp_is_zero() {
        let cp_dims = [5, 5, 5];
        let cp_n = 5 * 5 * 5;
        let cp = vec![3.0_f32; cp_n];
        let lap = cp_laplacian(&cp, cp_dims);
        for (i, &v) in lap.iter().enumerate() {
            assert!(
                v.abs() < 1e-6,
                "Laplacian of constant field at {i} should be 0, got {v}"
            );
        }
    }

    /// Laplacian at the centre of a CP lattice with a single non-zero point
    /// matches the analytical 6-connected discrete Laplacian.
    #[test]
    fn laplacian_single_spike() {
        let cp_dims = [5, 5, 5];
        let cp_n = 5 * 5 * 5;
        let mut cp = vec![0.0_f32; cp_n];
        let centre = flat(2, 2, 2, 5, 5);
        cp[centre] = 1.0;
        let lap = cp_laplacian(&cp, cp_dims);
        // At centre: Δ = 0 + 0 + 0 + 0 + 0 + 0 − 6 · 1 = −6
        assert!(
            (lap[centre] - (-6.0)).abs() < 1e-6,
            "centre Laplacian should be -6.0, got {}",
            lap[centre]
        );
        // At each face neighbour: Δ = 1 − 6·0 = 1 (if interior with 6 neighbours)
        // but if the neighbour is not on the boundary of the lattice and only has
        // one non-zero neighbour (the centre), Δ = 1 - count*0 = 1.
        let nb = flat(2, 2, 3, 5, 5);
        assert!(
            (lap[nb] - 1.0).abs() < 1e-6,
            "neighbour Laplacian should be 1.0, got {}",
            lap[nb]
        );
    }

    /// Accumulation of a constant force field to CPs yields the constant value
    /// (by the weighted-average normalisation).
    #[test]
    fn accumulate_constant_force() {
        let dims = [8, 8, 8];
        let cs = [3, 3, 3];
        let cp_dims = [cp_count(8, 3), cp_count(8, 3), cp_count(8, 3)];
        let force = vec![2.0_f32; 8 * 8 * 8];
        let acc = accumulate_to_cp(&force, dims, cp_dims, cs);
        for (i, &v) in acc.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-4,
                "CP {i}: accumulated constant force should be 2.0, got {v}"
            );
        }
    }

    // ── Registration tests ────────────────────────────────────────────────────

    /// Registering identical images produces CC > 0.9.
    #[test]
    fn identity_registration_high_cc() {
        let dims = [10, 10, 10];
        let image = make_test_image(dims);
        let cfg = make_default_config();
        let reg = BSplineSyNRegistration::new(cfg);
        let result = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]).unwrap();
        assert!(
            result.final_cc > 0.9,
            "identity registration CC should be > 0.9, got {}",
            result.final_cc
        );
    }

    /// BSplineSyN on a translated pair: non-divergence and non-trivial fields.
    #[test]
    fn bspline_registration_non_divergence() {
        let dims = [12, 12, 16];
        let n = dims[0] * dims[1] * dims[2];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 2);

        let cfg = BSplineSyNConfig {
            max_iterations: 20,
            control_spacing: [4, 4, 4],
            sigma_smooth: 1.5,
            convergence_threshold: 1e-7,
            convergence_window: 10,
            n_squarings: 6,
            cc_window_radius: 2,
            gradient_step: 0.25,
            regularization_weight: 0.01,
        };
        let reg = BSplineSyNRegistration::new(cfg);
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        // Velocity fields must have non-trivial x-magnitude.
        let fwd_rms_x: f64 = (result
            .forward_field
            .2
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        let inv_rms_x: f64 = (result
            .inverse_field
            .2
            .iter()
            .map(|&v| (v as f64).powi(2))
            .sum::<f64>()
            / n as f64)
            .sqrt();
        assert!(
            fwd_rms_x > 0.001 || inv_rms_x > 0.001,
            "at least one x-field must be non-trivial: fwd={fwd_rms_x:.6} inv={inv_rms_x:.6}"
        );

        // CC must remain high.
        assert!(
            result.final_cc > 0.8,
            "final CC must be > 0.8, got {}",
            result.final_cc
        );

        // All field values must be finite.
        for &v in result
            .forward_field
            .0
            .iter()
            .chain(result.forward_field.1.iter())
            .chain(result.forward_field.2.iter())
            .chain(result.inverse_field.0.iter())
            .chain(result.inverse_field.1.iter())
            .chain(result.inverse_field.2.iter())
        {
            assert!(v.is_finite(), "field contains non-finite value: {v}");
        }
    }

    /// B-spline velocity fields are intrinsically smooth: the dense field
    /// Laplacian RMS must be bounded. This is a structural property of the
    /// cubic B-spline representation.
    #[test]
    fn bspline_field_smoothness() {
        let dims = [10, 10, 10];
        let n = dims[0] * dims[1] * dims[2];
        let fixed = make_test_image(dims);
        let moving = translate_x(&fixed, dims, 1);

        let cfg = make_default_config();
        let reg = BSplineSyNRegistration::new(cfg);
        let result = reg
            .register(&fixed, &moving, dims, [1.0, 1.0, 1.0])
            .unwrap();

        // Compute discrete Laplacian of the dense forward x-field.
        let vx = &result.forward_field.2;
        let [nz, ny, nx] = dims;
        let mut lap_ss = 0.0_f64;
        let mut field_ss = 0.0_f64;
        for iz in 1..nz - 1 {
            for iy in 1..ny - 1 {
                for ix in 1..nx - 1 {
                    let fi = flat(iz, iy, ix, ny, nx);
                    let c = vx[fi] as f64;
                    let lap = vx[flat(iz - 1, iy, ix, ny, nx)] as f64
                        + vx[flat(iz + 1, iy, ix, ny, nx)] as f64
                        + vx[flat(iz, iy - 1, ix, ny, nx)] as f64
                        + vx[flat(iz, iy + 1, ix, ny, nx)] as f64
                        + vx[flat(iz, iy, ix - 1, ny, nx)] as f64
                        + vx[flat(iz, iy, ix + 1, ny, nx)] as f64
                        - 6.0 * c;
                    lap_ss += lap * lap;
                    field_ss += c * c;
                }
            }
        }
        let field_rms = (field_ss / n as f64).sqrt();
        let lap_rms = (lap_ss / n as f64).sqrt();
        // For a zero field (identity registration), both are near zero.
        // For non-trivial fields, the Laplacian RMS should be small relative
        // to the field RMS, or both should be small.
        if field_rms > 1e-6 {
            let ratio = lap_rms / field_rms;
            assert!(
                ratio < 50.0,
                "Laplacian/field RMS ratio {ratio:.2} too large; field not smooth"
            );
        }
        // In any case, the Laplacian should not be enormous.
        assert!(
            lap_rms < 100.0,
            "Laplacian RMS {lap_rms:.4} too large for B-spline field"
        );
    }

    // ── Error-case tests ──────────────────────────────────────────────────────

    /// Mismatched fixed-image length returns DimensionMismatch.
    #[test]
    fn mismatched_fixed_length_returns_error() {
        let dims = [4, 4, 4];
        let fixed = vec![0.0_f32; 4 * 4 * 5]; // wrong length
        let moving = vec![0.0_f32; 4 * 4 * 4];
        let cfg = make_default_config();
        let reg = BSplineSyNRegistration::new(cfg);
        let err = reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0]);
        assert!(err.is_err(), "should error for mismatched fixed length");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("fixed length"),
            "error should mention fixed: {msg}"
        );
    }

    /// Mismatched moving-image length returns DimensionMismatch.
    #[test]
    fn mismatched_moving_length_returns_error() {
        let dims = [4, 4, 4];
        let fixed = vec![0.0_f32; 4 * 4 * 4];
        let moving = vec![0.0_f32; 4 * 4 * 5]; // wrong length
        let cfg = make_default_config();
        let reg = BSplineSyNRegistration::new(cfg);
        let err = reg.register(&fixed, &moving, dims, [1.0, 1.0, 1.0]);
        assert!(err.is_err(), "should error for mismatched moving length");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("moving length"),
            "error should mention moving: {msg}"
        );
    }

    /// Zero control spacing returns InvalidConfiguration.
    #[test]
    fn zero_control_spacing_returns_error() {
        let dims = [4, 4, 4];
        let image = vec![0.0_f32; 4 * 4 * 4];
        let mut cfg = make_default_config();
        cfg.control_spacing = [0, 3, 3];
        let reg = BSplineSyNRegistration::new(cfg);
        let err = reg.register(&image, &image, dims, [1.0, 1.0, 1.0]);
        assert!(err.is_err(), "should error for zero control spacing");
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("control_spacing"),
            "error should mention control_spacing: {msg}"
        );
    }

    // ── CC primitive tests ────────────────────────────────────────────────────

    /// mean_local_cc of identical non-constant images is close to 1.0.
    #[test]
    fn mean_local_cc_identical_images() {
        let dims = [6, 6, 6];
        let image = make_test_image(dims);
        let cc = mean_local_cc(&image, &image, dims, 1);
        assert!(
            cc > 0.99,
            "CC of identical images should be ≈ 1.0, got {cc}"
        );
    }

    /// mean_local_cc of constant images is 0 (zero variance → degenerate).
    #[test]
    fn mean_local_cc_constant_images_is_zero() {
        let dims = [5, 5, 5];
        let a = vec![3.0_f32; 5 * 5 * 5];
        let cc = mean_local_cc(&a, &a, dims, 1);
        assert!(
            cc.is_finite(),
            "CC of constant images must be finite, got {cc}"
        );
        assert!(
            cc.abs() < 1e-6,
            "CC of constant images should be 0, got {cc}"
        );
    }
}
