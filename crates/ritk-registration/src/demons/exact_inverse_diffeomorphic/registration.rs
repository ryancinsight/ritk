//! Inverse-consistent diffeomorphic Demons registration.
//!
//! # Memory discipline
//! All scratch buffers are pre-allocated before the iteration loop.
//! The loop body performs **zero heap allocations**; all `_into` variants
//! write into caller-provided buffers. Total pre-allocation: ~22n f32
//! (3 velocity + 3 fixed grad + 3 moving grad + 3 forward displacement +
//! 3 inverted velocity + 3 backward displacement + 1 forward warped +
//! 1 backward warped + 3 forward forces + 3 backward forces +
//! 3 SS scratch + 1 smooth scratch = ~30n, with some sharing).

use super::super::config::DemonsConfig;
use super::super::inverse::invert_velocity_field_into;
use super::super::thirion::thirion_forces_into;
use super::ic_residual::compute_ic_residual;
use crate::deformable_field_ops::{
    compute_gradient_into, compute_mse_streaming, gaussian_smooth_with_scratch,
    scaling_and_squaring_into, warp_image_into,
};
use crate::error::RegistrationError;

/// Configuration for InverseConsistentDiffeomorphicDemonsRegistration.
#[derive(Debug, Clone)]
pub struct InverseConsistentDemonsConfig {
    /// Shared Demons parameters.
    pub demons: DemonsConfig,
    /// Weight of the backward (inverse) force. Range [0,1]. Default 0.5.
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

        let mut gf_z = vec![0.0_f32; n];
        let mut gf_y = vec![0.0_f32; n];
        let mut gf_x = vec![0.0_f32; n];
        compute_gradient_into(fixed, dims, spacing, &mut gf_z, &mut gf_y, &mut gf_x);
        let mut gm_z = vec![0.0_f32; n];
        let mut gm_y = vec![0.0_f32; n];
        let mut gm_x = vec![0.0_f32; n];
        compute_gradient_into(moving, dims, spacing, &mut gm_z, &mut gm_y, &mut gm_x);

        let mut vel_z = vec![0.0_f32; n];
        let mut vel_y = vec![0.0_f32; n];
        let mut vel_x = vec![0.0_f32; n];

        // ── Pre-allocated scratch buffers (zero alloc inside the loop) ──
        let mut phi_z = vec![0.0_f32; n];
        let mut phi_y = vec![0.0_f32; n];
        let mut phi_x = vec![0.0_f32; n];
        let mut inv_vel_z = vec![0.0_f32; n];
        let mut inv_vel_y = vec![0.0_f32; n];
        let mut inv_vel_x = vec![0.0_f32; n];
        let mut psi_z = vec![0.0_f32; n];
        let mut psi_y = vec![0.0_f32; n];
        let mut psi_x = vec![0.0_f32; n];
        let mut m_warped = vec![0.0_f32; n];
        let mut f_warped = vec![0.0_f32; n];
        let mut fz_fwd = vec![0.0_f32; n];
        let mut fy_fwd = vec![0.0_f32; n];
        let mut fx_fwd = vec![0.0_f32; n];
        let mut fz_bwd = vec![0.0_f32; n];
        let mut fy_bwd = vec![0.0_f32; n];
        let mut fx_bwd = vec![0.0_f32; n];
        let mut scratch_ss_z = vec![0.0_f32; n];
        let mut scratch_ss_y = vec![0.0_f32; n];
        let mut scratch_ss_x = vec![0.0_f32; n];
        let mut smooth_tmp = vec![0.0_f32; n];

        // Initial MSE: v = 0 (identity) — MSE of raw fixed vs moving.
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

            // φ_fwd = exp(v)
            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims,
                n_sq,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );

            // φ_inv = exp(−v)
            invert_velocity_field_into(
                &vel_z,
                &vel_y,
                &vel_x,
                &mut inv_vel_z,
                &mut inv_vel_y,
                &mut inv_vel_x,
            );
            scaling_and_squaring_into(
                &inv_vel_z,
                &inv_vel_y,
                &inv_vel_x,
                dims,
                n_sq,
                &mut psi_z,
                &mut psi_y,
                &mut psi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );

            warp_image_into(moving, dims, &phi_z, &phi_y, &phi_x, &mut m_warped);
            warp_image_into(fixed, dims, &psi_z, &psi_y, &psi_x, &mut f_warped);

            thirion_forces_into(
                fixed,
                &m_warped,
                &gf_z,
                &gf_y,
                &gf_x,
                cfg.max_step_length,
                &mut fz_fwd,
                &mut fy_fwd,
                &mut fx_fwd,
            );
            thirion_forces_into(
                moving,
                &f_warped,
                &gm_z,
                &gm_y,
                &gm_x,
                cfg.max_step_length,
                &mut fz_bwd,
                &mut fy_bwd,
                &mut fx_bwd,
            );

            let w_fwd = (1.0 - w) as f32;
            let w_bwd = w as f32;
            for i in 0..n {
                vel_z[i] += w_fwd * fz_fwd[i] - w_bwd * fz_bwd[i];
                vel_y[i] += w_fwd * fy_fwd[i] - w_bwd * fy_bwd[i];
                vel_x[i] += w_fwd * fx_fwd[i] - w_bwd * fx_bwd[i];
            }

            if cfg.sigma_diffusion > 0.0 {
                gaussian_smooth_with_scratch(
                    &mut vel_z,
                    dims,
                    cfg.sigma_diffusion,
                    &mut smooth_tmp,
                );
                gaussian_smooth_with_scratch(
                    &mut vel_y,
                    dims,
                    cfg.sigma_diffusion,
                    &mut smooth_tmp,
                );
                gaussian_smooth_with_scratch(
                    &mut vel_x,
                    dims,
                    cfg.sigma_diffusion,
                    &mut smooth_tmp,
                );
            }

            // Reuse psi buffer for MSE displacement — psi is consumed here and
            // overwritten at the top of the next iteration.
            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims,
                n_sq,
                &mut psi_z,
                &mut psi_y,
                &mut psi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            final_mse = compute_mse_streaming(fixed, moving, dims, &psi_z, &psi_y, &psi_x);
        }

        // Final φ_fwd = exp(v) — reuses phi buffers.
        scaling_and_squaring_into(
            &vel_z,
            &vel_y,
            &vel_x,
            dims,
            n_sq,
            &mut phi_z,
            &mut phi_y,
            &mut phi_x,
            &mut scratch_ss_z,
            &mut scratch_ss_y,
            &mut scratch_ss_x,
        );
        let mut warped = vec![0.0_f32; n];
        warp_image_into(moving, dims, &phi_z, &phi_y, &phi_x, &mut warped);

        // Final φ_inv = exp(−v)
        invert_velocity_field_into(
            &vel_z,
            &vel_y,
            &vel_x,
            &mut inv_vel_z,
            &mut inv_vel_y,
            &mut inv_vel_x,
        );
        scaling_and_squaring_into(
            &inv_vel_z,
            &inv_vel_y,
            &inv_vel_x,
            dims,
            n_sq,
            &mut psi_z,
            &mut psi_y,
            &mut psi_x,
            &mut scratch_ss_z,
            &mut scratch_ss_y,
            &mut scratch_ss_x,
        );

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
