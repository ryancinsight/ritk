//! Diffeomorphic Demons registration struct and iteration loop.

use super::super::config::{DemonsConfig, DemonsResult};
use super::super::inverse::invert_velocity_field;
use super::super::thirion::thirion_forces_into;
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_streaming, gaussian_smooth_inplace, scaling_and_squaring,
    warp_image, VectorField3D, VectorFieldMut3D,
};
use crate::error::RegistrationError;

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

    /// Register `moving` to `fixed` using a stationary velocity field.
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

        let (grad_z, grad_y, grad_x) = compute_gradient(fixed, dims, spacing);

        let mut final_mse = compute_mse_direct(
            fixed,
            moving,
            &vel_z,
            &vel_y,
            &vel_x,
            dims,
            self.n_squarings,
        );
        let mut iter = 0usize;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            let (phi_z, phi_y, phi_x) =
                scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, self.n_squarings);

            let m_warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);

            thirion_forces_into(
                fixed,
                &m_warped,
                VectorField3D {
                    z: &grad_z,
                    y: &grad_y,
                    x: &grad_x,
                },
                self.config.max_step_length,
                VectorFieldMut3D {
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

            if self.config.sigma_diffusion > 0.0 {
                gaussian_smooth_inplace(&mut vel_z, dims, self.config.sigma_diffusion);
                gaussian_smooth_inplace(&mut vel_y, dims, self.config.sigma_diffusion);
                gaussian_smooth_inplace(&mut vel_x, dims, self.config.sigma_diffusion);
            }

            final_mse = compute_mse_direct(
                fixed,
                moving,
                &vel_z,
                &vel_y,
                &vel_x,
                dims,
                self.n_squarings,
            );
        }

        let (phi_z, phi_y, phi_x) =
            scaling_and_squaring(&vel_z, &vel_y, &vel_x, dims, self.n_squarings);
        let warped = warp_image(moving, dims, &phi_z, &phi_y, &phi_x);

        Ok(DemonsResult {
            warped,
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
    ///
    /// Uses `exp(−v)` when the SVF is available; falls back to fixed-point
    /// inversion of the displacement field otherwise.
    pub fn invert_result(
        &self,
        result: &DemonsResult,
        dims: [usize; 3],
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        match (&result.vel_z, &result.vel_y, &result.vel_x) {
            (Some(vel_z), Some(vel_y), Some(vel_x)) => {
                let (inv_vel_z, inv_vel_y, inv_vel_x) = invert_velocity_field(vel_z, vel_y, vel_x);
                scaling_and_squaring(&inv_vel_z, &inv_vel_y, &inv_vel_x, dims, self.n_squarings)
            }
            _ => {
                use crate::demons::inverse::{invert_displacement_field, InverseFieldConfig};
                let config = InverseFieldConfig::default();
                let (inv_z, inv_y, inv_x, _) = invert_displacement_field(
                    &result.disp_z,
                    &result.disp_y,
                    &result.disp_x,
                    dims,
                    &config,
                );
                (inv_z, inv_y, inv_x)
            }
        }
    }
}

fn compute_mse_direct(
    fixed: &[f32],
    moving: &[f32],
    vel_z: &[f32],
    vel_y: &[f32],
    vel_x: &[f32],
    dims: [usize; 3],
    n_squarings: usize,
) -> f64 {
    let (phi_z, phi_y, phi_x) = scaling_and_squaring(vel_z, vel_y, vel_x, dims, n_squarings);
    compute_mse_streaming(fixed, moving, dims, &phi_z, &phi_y, &phi_x)
}
