//! Inverse-consistent diffeomorphic Demons registration engine.

use super::super::inverse::invert_velocity_field_into;
use super::super::thirion::thirion_forces_into;
use super::ic_residual::compute_ic_residual;
use super::types::{InverseConsistentDemonsConfig, InverseConsistentDemonsResult};
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_streaming, gaussian_smooth_field_inplace_with_scratch,
    scaling_and_squaring_into, warp_image_into, GpuFieldSmoother, VectorField, VectorFieldMut,
};
use crate::error::RegistrationError;
use burn::tensor::backend::Backend;

/// Inverse-consistent diffeomorphic Demons registration.
///
/// Maintains φ_fwd = exp(v) and φ_inv = exp(−v) simultaneously.
/// Uses a symmetric bilateral force: (1−w)·forward + w·(−backward).
///
/// # Bilateral Objective
///
/// E(v) = (1−w)·‖F − M∘exp(v)‖² + w·‖M − F∘exp(−v)‖²
///
/// # Update Rule (first-order BCH)
///
/// v ← v + (1−w)·u_fwd − w·u_bwd
/// v ← G_{σ_diff} ∗ v
#[derive(Debug, Clone)]
pub struct InverseConsistentDiffeomorphicDemonsRegistration {
    pub config: InverseConsistentDemonsConfig,
}

impl InverseConsistentDiffeomorphicDemonsRegistration {
    pub fn new(config: InverseConsistentDemonsConfig) -> Self {
        Self { config }
    }

    /// Register moving onto fixed with **GPU-accelerated** Gaussian field
    /// smoothing for diffusion regularisation.
    ///
    /// The [`GpuFieldSmoother`] manages pre-allocated staging tensors and a
    /// single [`ritk_filter::GaussianFilter`] reused across all iterations.
    pub fn register_with_gpu_smoother<B: Backend>(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
        gpu_smoother: &mut GpuFieldSmoother<B>,
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

        let gf = compute_gradient(fixed, dims.into(), spacing);
        let gm = compute_gradient(moving, dims.into(), spacing);

        let mut vel_z = vec![0.0_f32; n];
        let mut vel_y = vec![0.0_f32; n];
        let mut vel_x = vec![0.0_f32; n];
        let mut fz_fwd = vec![0.0_f32; n];
        let mut fy_fwd = vec![0.0_f32; n];
        let mut fx_fwd = vec![0.0_f32; n];
        let mut fz_bwd = vec![0.0_f32; n];
        let mut fy_bwd = vec![0.0_f32; n];
        let mut fx_bwd = vec![0.0_f32; n];

        // Pre-allocated scratch (zero alloc inside the iteration loop).
        let mut phi_z = vec![0.0_f32; n];
        let mut phi_y = vec![0.0_f32; n];
        let mut phi_x = vec![0.0_f32; n];
        let mut psi_z = vec![0.0_f32; n];
        let mut psi_y = vec![0.0_f32; n];
        let mut psi_x = vec![0.0_f32; n];
        let mut inv_vel_z = vec![0.0_f32; n];
        let mut inv_vel_y = vec![0.0_f32; n];
        let mut inv_vel_x = vec![0.0_f32; n];
        let mut scratch_ss_z = vec![0.0_f32; n];
        let mut scratch_ss_y = vec![0.0_f32; n];
        let mut scratch_ss_x = vec![0.0_f32; n];
        let mut m_warped = vec![0.0_f32; n];
        let mut f_warped = vec![0.0_f32; n];

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

            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims.into(),
                n_sq,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
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
                dims.into(),
                n_sq,
                &mut psi_z,
                &mut psi_y,
                &mut psi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            warp_image_into(moving, dims.into(), &phi_z, &phi_y, &phi_x, &mut m_warped);
            warp_image_into(fixed, dims.into(), &psi_z, &psi_y, &psi_x, &mut f_warped);

            thirion_forces_into(
                fixed,
                &m_warped,
                VectorField {
                    z: &gf.z,
                    y: &gf.y,
                    x: &gf.x,
                },
                cfg.max_step_length,
                VectorFieldMut {
                    z: &mut fz_fwd,
                    y: &mut fy_fwd,
                    x: &mut fx_fwd,
                },
            );
            thirion_forces_into(
                moving,
                &f_warped,
                VectorField {
                    z: &gm.z,
                    y: &gm.y,
                    x: &gm.x,
                },
                cfg.max_step_length,
                VectorFieldMut {
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

            // GPU-accelerated diffusion regularisation
            if cfg.sigma_diffusion.is_some() {
                gpu_smoother.smooth_field_inplace(&mut vel_z, &mut vel_y, &mut vel_x);
            }

            // Re-compute exp(vel) for updated velocity before MSE.
            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims.into(),
                n_sq,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            final_mse = compute_mse_streaming(fixed, moving, dims.into(), &phi_z, &phi_y, &phi_x);
        }

        // phi_z/y/x already holds exp(vel) from the final MSE step of the loop.
        warp_image_into(moving, dims.into(), &phi_z, &phi_y, &phi_x, &mut m_warped);
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
            dims.into(),
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
            warped: m_warped,
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

        let gf = compute_gradient(fixed, dims.into(), spacing);
        let gm = compute_gradient(moving, dims.into(), spacing);

        let mut vel_z = vec![0.0_f32; n];
        let mut vel_y = vec![0.0_f32; n];
        let mut vel_x = vec![0.0_f32; n];
        let mut fz_fwd = vec![0.0_f32; n];
        let mut fy_fwd = vec![0.0_f32; n];
        let mut fx_fwd = vec![0.0_f32; n];
        let mut fz_bwd = vec![0.0_f32; n];
        let mut fy_bwd = vec![0.0_f32; n];
        let mut fx_bwd = vec![0.0_f32; n];

        // ── Pre-allocated scratch (zero alloc inside the iteration loop) ─────────
        let mut phi_z = vec![0.0_f32; n];
        let mut phi_y = vec![0.0_f32; n];
        let mut phi_x = vec![0.0_f32; n];
        let mut psi_z = vec![0.0_f32; n];
        let mut psi_y = vec![0.0_f32; n];
        let mut psi_x = vec![0.0_f32; n];
        let mut inv_vel_z = vec![0.0_f32; n];
        let mut inv_vel_y = vec![0.0_f32; n];
        let mut inv_vel_x = vec![0.0_f32; n];
        let mut scratch_ss_z = vec![0.0_f32; n];
        let mut scratch_ss_y = vec![0.0_f32; n];
        let mut scratch_ss_x = vec![0.0_f32; n];
        let mut m_warped = vec![0.0_f32; n];
        let mut f_warped = vec![0.0_f32; n];
        // Pre-hoisted scratch: reused by the smooth call, eliminates 3×n f32 allocs per iter.
        let mut smooth_tmp = vec![0.0_f32; n];

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

            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims.into(),
                n_sq,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
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
                dims.into(),
                n_sq,
                &mut psi_z,
                &mut psi_y,
                &mut psi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            warp_image_into(moving, dims.into(), &phi_z, &phi_y, &phi_x, &mut m_warped);
            warp_image_into(fixed, dims.into(), &psi_z, &psi_y, &psi_x, &mut f_warped);

            thirion_forces_into(
                fixed,
                &m_warped,
                VectorField {
                    z: &gf.z,
                    y: &gf.y,
                    x: &gf.x,
                },
                cfg.max_step_length,
                VectorFieldMut {
                    z: &mut fz_fwd,
                    y: &mut fy_fwd,
                    x: &mut fx_fwd,
                },
            );
            thirion_forces_into(
                moving,
                &f_warped,
                VectorField {
                    z: &gm.z,
                    y: &gm.y,
                    x: &gm.x,
                },
                cfg.max_step_length,
                VectorFieldMut {
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

            if let Some(sigma) = cfg.sigma_diffusion {
                gaussian_smooth_field_inplace_with_scratch(
                    &mut vel_z,
                    &mut vel_y,
                    &mut vel_x,
                    dims.into(),
                    sigma.get(),
                    &mut smooth_tmp,
                );
            }

            // Re-compute exp(vel) for updated velocity before MSE.
            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims.into(),
                n_sq,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            final_mse = compute_mse_streaming(fixed, moving, dims.into(), &phi_z, &phi_y, &phi_x);
        }

        // phi_z/y/x already holds exp(vel) from the final MSE step of the loop.
        warp_image_into(moving, dims.into(), &phi_z, &phi_y, &phi_x, &mut m_warped);
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
            dims.into(),
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
            warped: m_warped,
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
