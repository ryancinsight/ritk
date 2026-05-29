//! Inverse-consistent diffeomorphic Demons registration.

use super::super::config::DemonsConfig;
use super::super::inverse::invert_velocity_field;
use super::super::thirion::thirion_forces_into;
use super::ic_residual::compute_ic_residual;
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_streaming, gaussian_smooth_inplace, scaling_and_squaring,
    warp_image, VectorField3D, VectorFieldMut3D,
};
use crate::error::RegistrationError;

/// Configuration for InverseConsistentDiffeomorphicDemonsRegistration.
#[derive(Debug, Clone)]
pub struct InverseConsistentDemonsConfig {
    /// Shared Demons parameters.
    pub demons: DemonsConfig,
    /// Weight of the backward (inverse) force. Range `[0, 1]`. Default 0.5.
    pub inverse_consistency_weight: f64,
    /// Scaling-and-squaring steps for exp(v). Default 6.
    pub n_squarings: usize,
}

impl Default for InverseConsistentDemonsConfig {
    fn default() -> Self {
        Self {
            demons: DemonsConfig::default(),
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
        }
    }
}

/// Result of InverseConsistentDiffeomorphicDemonsRegistration.
pub struct InverseConsistentDemonsResult {
    /// Moving image warped onto fixed using phi_fwd = exp(v).
    pub warped: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), z-component.
    pub disp_z: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), y-component.
    pub disp_y: Vec<f32>,
    /// Forward displacement phi_fwd = exp(v), x-component.
    pub disp_x: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), z-component.
    pub inv_disp_z: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), y-component.
    pub inv_disp_y: Vec<f32>,
    /// Exact inverse displacement phi_inv = exp(-v), x-component.
    pub inv_disp_x: Vec<f32>,
    /// Stationary velocity field, z-component.
    pub vel_z: Vec<f32>,
    /// Stationary velocity field, y-component.
    pub vel_y: Vec<f32>,
    /// Stationary velocity field, x-component.
    pub vel_x: Vec<f32>,
    /// Final MSE(F, M o phi_fwd) at convergence.
    pub final_mse: f64,
    /// Number of iterations executed.
    pub num_iterations: usize,
    /// IC residual: mean‖φ_fwd(φ_inv(x)) − x‖₂.
    pub inverse_consistency_residual: f64,
}

/// Inverse-consistent diffeomorphic Demons registration.
///
/// Maintains φ_fwd = exp(v) and φ_inv = exp(−v) simultaneously.
/// Uses a symmetric bilateral force: (1−w)·forward + w·(−backward).
///
/// # Bilateral Objective
///
///   E(v) = (1−w)·‖F − M∘exp(v)‖² + w·‖M − F∘exp(−v)‖²
///
/// # Update Rule (first-order BCH)
///
///   v ← v + (1−w)·u_fwd − w·u_bwd
///   v ← G_{σ_diff} ∗ v
#[derive(Debug, Clone)]
pub struct InverseConsistentDiffeomorphicDemonsRegistration {
    pub config: InverseConsistentDemonsConfig,
}

impl InverseConsistentDiffeomorphicDemonsRegistration {
    pub fn new(config: InverseConsistentDemonsConfig) -> Self {
        Self { config }
    }

    /// Register moving onto fixed.
    ///
    /// # Errors
    /// Returns [`RegistrationError::DimensionMismatch`] on shape mismatch.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<InverseConsistentDemonsResult, RegistrationError> {
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

        let w = self.config.inverse_consistency_weight.clamp(0.0, 1.0);
        let cfg = &self.config.demons;
        let n_sq = self.config.n_squarings;

        let (gf_z, gf_y, gf_x) = compute_gradient(fixed, dims, spacing);
        let (gm_z, gm_y, gm_x) = compute_gradient(moving, dims, spacing);

        let mut vel_z = vec![0.0_f32; n];
        let mut vel_y = vec![0.0_f32; n];
        let mut vel_x = vec![0.0_f32; n];
        let mut fz_fwd = vec![0.0_f32; n];
        let mut fy_fwd = vec![0.0_f32; n];
        let mut fx_fwd = vec![0.0_f32; n];
        let mut fz_bwd = vec![0.0_f32; n];
        let mut fy_bwd = vec![0.0_f32; n];
        let mut fx_bwd = vec![0.0_f32; n];

        let mut final_mse: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&fi, &mi)| {
                let d = (fi - mi) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        let mut iter = 0usize;

        for it in 0..cfg.max_iterations {
            iter = it + 1;

            let (phi_z, phi_y, phi_x) = scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, n_sq);

            let (inv_vel_z, inv_vel_y, inv_vel_x) = invert_velocity_field(&vel_z, &vel_y, &vel_x);
            let (psi_z, psi_y, psi_x) =
                scaling_and_squaring(&inv_vel_z, &inv_vel_y, &inv_vel_x, dims, n_sq);

            let m_warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);
            let f_warped = warp_image(fixed, dims, &psi_z, &psi_y, &psi_x);

            thirion_forces_into(
                fixed,
                &m_warped,
                VectorField3D {
                    z: &gf_z,
                    y: &gf_y,
                    x: &gf_x,
                },
                cfg.max_step_length,
                VectorFieldMut3D {
                    z: &mut fz_fwd,
                    y: &mut fy_fwd,
                    x: &mut fx_fwd,
                },
            );
            thirion_forces_into(
                moving,
                &f_warped,
                VectorField3D {
                    z: &gm_z,
                    y: &gm_y,
                    x: &gm_x,
                },
                cfg.max_step_length,
                VectorFieldMut3D {
                    z: &mut fz_bwd,
                    y: &mut fy_bwd,
                    x: &mut fx_bwd,
                },
            );

            let w_fwd = (1.0 - w) as f32;
            let w_bwd = w as f32;
            for i in 0..n {
                vel_z[i] += w_fwd * fz_fwd[i] - w_bwd * fz_bwd[i];
                vel_y[i] += w_fwd * fy_fwd[i] - w_bwd * fy_bwd[i];
                vel_x[i] += w_fwd * fx_fwd[i] - w_bwd * fx_bwd[i];
            }

            if cfg.sigma_diffusion > 0.0 {
                gaussian_smooth_inplace(&mut vel_z, dims, cfg.sigma_diffusion);
                gaussian_smooth_inplace(&mut vel_y, dims, cfg.sigma_diffusion);
                gaussian_smooth_inplace(&mut vel_x, dims, cfg.sigma_diffusion);
            }

            {
                let (pz, py, px) = scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, n_sq);
                final_mse = compute_mse_streaming(fixed, moving, dims, &pz, &py, &px);
            }
        }

        let (phi_z, phi_y, phi_x) = scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, n_sq);
        let warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);

        let (inv_vel_z, inv_vel_y, inv_vel_x) = invert_velocity_field(&vel_z, &vel_y, &vel_x);
        let (psi_z, psi_y, psi_x) =
            scaling_and_squaring(&inv_vel_z, &inv_vel_y, &inv_vel_x, dims, n_sq);

        let ic_residual = compute_ic_residual(&phi_z, &phi_y, &phi_x, &psi_z, &psi_y, &psi_x, dims);

        Ok(InverseConsistentDemonsResult {
            warped,
            disp_z: phi_z,
            disp_y: phi_y,
            disp_x: phi_x,
            inv_disp_z: psi_z,
            inv_disp_y: psi_y,
            inv_disp_x: psi_x,
            vel_z,
            vel_y,
            vel_x,
            final_mse,
            num_iterations: iter,
            inverse_consistency_residual: ic_residual,
        })
    }
}
