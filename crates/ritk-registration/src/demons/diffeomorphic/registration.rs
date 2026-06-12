//! Diffeomorphic Demons registration struct and iteration loop.

use super::super::config::{DemonsConfig, DemonsResult};
use super::super::inverse::invert_velocity_field;
use super::super::thirion::thirion_forces_into;
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_streaming, gaussian_smooth_field_inplace_with_scratch,
    scaling_and_squaring, scaling_and_squaring_into, warp_image_into, GpuFieldSmoother,
    VectorField, VectorFieldMut, VelocityField,
};
use crate::error::RegistrationError;
use burn::tensor::backend::Backend;

/// Diffeomorphic Demons registration using stationary velocity fields.
///
/// Produces topologically correct (invertible) deformation fields by
/// representing the transformation as the exponential map of a velocity field.
#[derive(Debug, Clone)]
pub struct DiffeomorphicDemonsRegistration {
    /// Algorithm configuration (shared with Thirion Demons).
    pub config: DemonsConfig,
    /// Number of scaling-and-squaring steps for computing `exp(v)`.
    /// Standard value: 6 (2⁶ = 64 integration steps).
    pub n_squarings: usize,
}

impl DiffeomorphicDemonsRegistration {
    /// Create a registration instance with the given configuration.
    ///
    /// `n_squarings` defaults to 6.
    pub fn new(config: DemonsConfig) -> Self {
        Self {
            config,
            n_squarings: 6,
        }
    }

    /// Create with a custom number of squaring steps.
    pub fn with_squarings(config: DemonsConfig, n_squarings: usize) -> Self {
        Self {
            config,
            n_squarings,
        }
    }

    /// Register `moving` to `fixed` using a stationary velocity field with
    /// **CPU-based** Gaussian field smoothing.
    ///
    /// Prefer [`register_with_gpu_smoother`] when a Burn GPU backend is available.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `Vec<f32>` in Z-major order.
    /// - `moving`  — moving image, same shape as `fixed`.
    /// - `dims`    — image dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if image lengths are inconsistent with `dims`.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<DemonsResult, RegistrationError> {
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

        let mut vel_z = vec![0.0_f32; n];
        let mut vel_y = vec![0.0_f32; n];
        let mut vel_x = vec![0.0_f32; n];
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];

        let mut phi_z = vec![0.0_f32; n];
        let mut phi_y = vec![0.0_f32; n];
        let mut phi_x = vec![0.0_f32; n];
        let mut scratch_ss_z = vec![0.0_f32; n];
        let mut scratch_ss_y = vec![0.0_f32; n];
        let mut scratch_ss_x = vec![0.0_f32; n];
        let mut m_warped = vec![0.0_f32; n];
        let mut smooth_tmp = vec![0.0_f32; n];

        let grad = compute_gradient(fixed, dims.into(), spacing);

        scaling_and_squaring_into(
            &vel_z,
            &vel_y,
            &vel_x,
            dims.into(),
            self.n_squarings,
            &mut phi_z,
            &mut phi_y,
            &mut phi_x,
            &mut scratch_ss_z,
            &mut scratch_ss_y,
            &mut scratch_ss_x,
        );
        let mut final_mse =
            compute_mse_streaming(fixed, moving, dims.into(), &phi_z, &phi_y, &phi_x);
        let mut iter = 0usize;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims.into(),
                self.n_squarings,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            warp_image_into(moving, dims.into(), &phi_z, &phi_y, &phi_x, &mut m_warped);

            thirion_forces_into(
                fixed,
                &m_warped,
                VectorField {
                    z: &grad.z,
                    y: &grad.y,
                    x: &grad.x,
                },
                self.config.max_step_length,
                VectorFieldMut {
                    z: &mut fz,
                    y: &mut fy,
                    x: &mut fx,
                },
            );

            for i in 0..n {
                vel_z[i] += fz[i];
                vel_y[i] += fy[i];
                vel_x[i] += fx[i];
            }

            if let Some(sigma) = self.config.sigma_diffusion {
                gaussian_smooth_field_inplace_with_scratch(
                    &mut vel_z,
                    &mut vel_y,
                    &mut vel_x,
                    dims.into(),
                    sigma.get(),
                    &mut smooth_tmp,
                );
            }

            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims.into(),
                self.n_squarings,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            final_mse = compute_mse_streaming(fixed, moving, dims.into(), &phi_z, &phi_y, &phi_x);
        }

        warp_image_into(moving, dims.into(), &phi_z, &phi_y, &phi_x, &mut m_warped);

        Ok(DemonsResult {
            warped: m_warped,
            disp_z: phi_z,
            disp_y: phi_y,
            disp_x: phi_x,
            vel_z: Some(vel_z),
            vel_y: Some(vel_y),
            vel_x: Some(vel_x),
            final_mse,
            num_iterations: iter,
        })
    }

    /// Register `moving` to `fixed` using a stationary velocity field with
    /// **GPU-accelerated** Gaussian field smoothing.
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
    ) -> Result<DemonsResult, RegistrationError> {
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

        let mut vel_z = vec![0.0_f32; n];
        let mut vel_y = vec![0.0_f32; n];
        let mut vel_x = vec![0.0_f32; n];
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];

        let mut phi_z = vec![0.0_f32; n];
        let mut phi_y = vec![0.0_f32; n];
        let mut phi_x = vec![0.0_f32; n];
        let mut scratch_ss_z = vec![0.0_f32; n];
        let mut scratch_ss_y = vec![0.0_f32; n];
        let mut scratch_ss_x = vec![0.0_f32; n];
        let mut m_warped = vec![0.0_f32; n];

        let grad = compute_gradient(fixed, dims.into(), spacing);

        scaling_and_squaring_into(
            &vel_z,
            &vel_y,
            &vel_x,
            dims.into(),
            self.n_squarings,
            &mut phi_z,
            &mut phi_y,
            &mut phi_x,
            &mut scratch_ss_z,
            &mut scratch_ss_y,
            &mut scratch_ss_x,
        );
        let mut final_mse =
            compute_mse_streaming(fixed, moving, dims.into(), &phi_z, &phi_y, &phi_x);
        let mut iter = 0usize;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims.into(),
                self.n_squarings,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            warp_image_into(moving, dims.into(), &phi_z, &phi_y, &phi_x, &mut m_warped);

            thirion_forces_into(
                fixed,
                &m_warped,
                VectorField {
                    z: &grad.z,
                    y: &grad.y,
                    x: &grad.x,
                },
                self.config.max_step_length,
                VectorFieldMut {
                    z: &mut fz,
                    y: &mut fy,
                    x: &mut fx,
                },
            );

            for i in 0..n {
                vel_z[i] += fz[i];
                vel_y[i] += fy[i];
                vel_x[i] += fx[i];
            }

            // GPU-accelerated diffusion regularisation
            if self.config.sigma_diffusion.is_some() {
                gpu_smoother.smooth_field_inplace(&mut vel_z, &mut vel_y, &mut vel_x);
            }

            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims.into(),
                self.n_squarings,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            final_mse = compute_mse_streaming(fixed, moving, dims.into(), &phi_z, &phi_y, &phi_x);
        }

        warp_image_into(moving, dims.into(), &phi_z, &phi_y, &phi_x, &mut m_warped);

        Ok(DemonsResult {
            warped: m_warped,
            disp_z: phi_z,
            disp_y: phi_y,
            disp_x: phi_x,
            vel_z: Some(vel_z),
            vel_y: Some(vel_y),
            vel_x: Some(vel_x),
            final_mse,
            num_iterations: iter,
        })
    }

    /// Compute the inverse displacement field of a registration result.
    pub fn invert_result(&self, result: &DemonsResult, dims: [usize; 3]) -> VelocityField {
        match (&result.vel_z, &result.vel_y, &result.vel_x) {
            (Some(vel_z), Some(vel_y), Some(vel_x)) => {
                let inv_vel = invert_velocity_field(vel_z, vel_y, vel_x);
                scaling_and_squaring(
                    &inv_vel.z,
                    &inv_vel.y,
                    &inv_vel.x,
                    dims.into(),
                    self.n_squarings,
                )
            }
            _ => {
                use crate::demons::inverse::{invert_displacement_field, InverseFieldConfig};
                let config = InverseFieldConfig::default();
                let (inv, _) = invert_displacement_field(
                    &result.disp_z,
                    &result.disp_y,
                    &result.disp_x,
                    dims,
                    &config,
                );
                inv
            }
        }
    }
}
