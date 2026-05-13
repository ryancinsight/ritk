//! B-Spline Symmetric Normalization (BSplineSyN) registration.
//!
//! # Mathematical Specification
//!
//! BSplineSyN parameterises the stationary velocity fields `v₁, v₂` of the
//! SyN framework using cubic B-spline control-point lattices instead of dense
//! voxel grids. This reduces the number of free parameters and provides
//! intrinsic C²-smooth velocity fields.
//!
//! ## Dense Field Evaluation
//!
//! For voxel position `(z, y, x)` and control-point spacing `s`:
//!
//!   `t_d = d / s_d`,  `span_d = ⌊t_d⌋`,  `u_d = t_d − span_d`
//!
//!   `v(z,y,x) = Σ_{l,m,n=0}^{3} Bₗ(u_z) Bₘ(u_y) Bₙ(u_x) · cp[span_z+l, span_y+m, span_x+n]`
//!
//! ## Bending Energy Regularisation
//!
//! Discrete 6-connected Laplacian on the CP lattice:
//!
//!   `Δcp[i,j,k] = Σ_face_neighbours cp[n] − count · cp[i,j,k]`
//!
//! Weight `λ` controls regularisation strength.
//!
//! # References
//! - Tustison, N. J. & Avants, B. B. (2013). Explicit B-spline regularization
//!   in diffeomorphic image registration. *Frontiers in Neuroinformatics* 7:39.
//! - Rueckert, D. et al. (1999). Nonrigid registration using free-form deformations.
//!   *IEEE TMI* 18(8):712–721.

use std::collections::VecDeque;

use crate::deformable_field_ops::{
    compute_gradient, gaussian_smooth_inplace, scaling_and_squaring, warp_image,
};
use crate::error::RegistrationError;

use self::primitives::{accumulate_to_cp, cp_count, cp_laplacian, evaluate_dense};
use super::local_cc::{cc_forces, mean_local_cc};

pub(crate) mod primitives;
#[cfg(test)]
mod tests;

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

        let mut cp1z = vec![0.0_f32; cp_n];
        let mut cp1y = vec![0.0_f32; cp_n];
        let mut cp1x = vec![0.0_f32; cp_n];
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

            let i_w = warp_image(fixed, dims, &phi1z, &phi1y, &phi1x);
            let j_w = warp_image(moving, dims, &phi2z, &phi2y, &phi2x);
            let (giz, giy, gix) = compute_gradient(&i_w, dims, spacing);
            let (gjz, gjy, gjx) = compute_gradient(&j_w, dims, spacing);

            let (u1z, u1y, u1x) = cc_forces(&i_w, &j_w, &giz, &giy, &gix, dims, r);
            let (u2z, u2y, u2x) = cc_forces(&j_w, &i_w, &gjz, &gjy, &gjx, dims, r);

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

            if sigma > 0.0 {
                gaussian_smooth_inplace(&mut u1z, dims, sigma);
                gaussian_smooth_inplace(&mut u1y, dims, sigma);
                gaussian_smooth_inplace(&mut u1x, dims, sigma);
                gaussian_smooth_inplace(&mut u2z, dims, sigma);
                gaussian_smooth_inplace(&mut u2y, dims, sigma);
                gaussian_smooth_inplace(&mut u2x, dims, sigma);
            }

            let d1z = accumulate_to_cp(&u1z, dims, cp_d, cs);
            let d1y = accumulate_to_cp(&u1y, dims, cp_d, cs);
            let d1x = accumulate_to_cp(&u1x, dims, cp_d, cs);
            let d2z = accumulate_to_cp(&u2z, dims, cp_d, cs);
            let d2y = accumulate_to_cp(&u2y, dims, cp_d, cs);
            let d2x = accumulate_to_cp(&u2x, dims, cp_d, cs);

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
