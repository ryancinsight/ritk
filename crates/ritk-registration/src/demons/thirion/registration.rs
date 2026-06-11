//! Thirion Demons registration struct and iteration loop.

use super::super::config::{DemonsConfig, DemonsResult};
use super::forces::thirion_forces_into;
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_streaming, gaussian_smooth_field_inplace_with_scratch,
    warp_image_into, VectorField3D, VectorFieldMut3D, VelocityField,
};
use crate::error::RegistrationError;

/// Thirion Demons registration (classic variant).
///
/// Computes a dense displacement field by iterating optical-flow-like forces
/// derived from image intensity difference and the fixed-image gradient.
#[derive(Debug, Clone)]
pub struct ThirionDemonsRegistration {
    /// Algorithm configuration.
    pub config: DemonsConfig,
}

impl ThirionDemonsRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: DemonsConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` and return the displacement field and
    /// warped moving image.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `Vec<f32>` in Z-major order.
    /// - `moving`  — moving image, same shape as `fixed`.
    /// - `dims`    — image dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]` (used for gradient).
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if `fixed` and `moving` have different
    /// lengths or `dims` are inconsistent.
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

        let mut disp_z = vec![0.0_f32; n];
        let mut disp_y = vec![0.0_f32; n];
        let mut disp_x = vec![0.0_f32; n];

        let grad = compute_gradient(fixed, dims.into(), spacing);

        let mut final_mse =
            compute_mse_streaming(fixed, moving, dims.into(), &disp_z, &disp_y, &disp_x);
        let mut iter = 0usize;
        let mut m_warped = vec![0.0_f32; n];
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];
        // Pre-hoisted scratch: reused by both smooth calls, eliminates 3×n f32 allocs per iter.
        let mut smooth_tmp = vec![0.0_f32; n];

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            warp_image_into(
                moving,
                dims.into(),
                &disp_z,
                &disp_y,
                &disp_x,
                &mut m_warped,
            );

            thirion_forces_into(
                fixed,
                &m_warped,
                VectorField3D {
                    z: &grad.z,
                    y: &grad.y,
                    x: &grad.x,
                },
                self.config.max_step_length,
                VectorFieldMut3D {
                    z: &mut fz,
                    y: &mut fy,
                    x: &mut fx,
                },
            );

            if let Some(sigma) = self.config.sigma_fluid {
                gaussian_smooth_field_inplace_with_scratch(
                    &mut fz,
                    &mut fy,
                    &mut fx,
                    dims.into(),
                    sigma.get(),
                    &mut smooth_tmp,
                );
            }

            for i in 0..n {
                disp_z[i] += fz[i];
                disp_y[i] += fy[i];
                disp_x[i] += fx[i];
            }

            if let Some(sigma) = self.config.sigma_diffusion {
                gaussian_smooth_field_inplace_with_scratch(
                    &mut disp_z,
                    &mut disp_y,
                    &mut disp_x,
                    dims.into(),
                    sigma.get(),
                    &mut smooth_tmp,
                );
            }

            final_mse =
                compute_mse_streaming(fixed, moving, dims.into(), &disp_z, &disp_y, &disp_x);
        }

        warp_image_into(
            moving,
            dims.into(),
            &disp_z,
            &disp_y,
            &disp_x,
            &mut m_warped,
        );

        Ok(DemonsResult {
            warped: m_warped,
            disp_z,
            disp_y,
            disp_x,
            vel_z: None,
            vel_y: None,
            vel_x: None,
            final_mse,
            num_iterations: iter,
        })
    }

    /// Compute the inverse displacement field of a registration result.
    ///
    /// # Mathematical Basis
    ///
    /// `DemonsResult` stores `disp = u` (the accumulated displacement field).
    /// The inverse `u^{-1}` satisfies `u^{-1}(x) = −u(x + u^{-1}(x))` and is
    /// approximated via fixed-point iteration (Christensen & Johnson 2001):
    ///
    ///   `u^{-1}_0(x)      = −u(x)`
    ///   `u^{-1}_{k+1}(x)  = −u(x + u^{-1}_k(x))`
    pub fn invert_result(&self, result: &DemonsResult, dims: [usize; 3]) -> VelocityField {
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
