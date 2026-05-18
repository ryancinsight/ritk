//! Diffeomorphic Demons registration struct and iteration loop.
//!
//! # Algorithm
//! Stationary-velocity-field (SVF) Demons: iteratively updates a velocity
//! field whose exponential map (scaling-and-squaring) gives the deformation.
//! Thirion optical-flow forces drive the update; Gaussian smoothing
//! regularises the velocity field.
//!
//! **Per-iteration steps:**
//! 1. φ = exp(v)  (scaling-and-squaring)
//! 2. I_w = warp(M, φ)
//! 3. f = Thirion_forces(F, I_w, ∇F)
//! 4. v ← v + f
//! 5. v ← G_σ ∗ v  (if σ > 0)
//! 6. MSE = MSE(F, M, φ)  (reuses φ from step 1)
//!
//! # Memory discipline
//! All scratch buffers are pre-allocated before the iteration loop.
//! The loop body performs **zero heap allocations**; all `_into` variants
//! write into caller-provided buffers. Total pre-allocation: ~14n f32
//! (3 velocity + 3 displacement + 3 SS scratch + 1 warped image +
//! 3 forces + 1 smooth scratch = 14n). The fixed-image gradient (3n)
//! is computed once before the loop and does not contribute to per-iteration
//! allocation.

use super::super::config::{DemonsConfig, DemonsResult};
use super::super::inverse::invert_velocity_field;
use super::super::thirion::thirion_forces_into;
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_streaming, gaussian_smooth_with_scratch,
    scaling_and_squaring_into, warp_image_into,
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
    /// - `fixed` — reference image, flat `Vec<f32>` in Z-major order.
    /// - `moving` — moving image, same shape as `fixed`.
    /// - `dims` — image dimensions `[nz, ny, nx]`.
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

        // Fixed-image gradient (computed once, not per-iteration)
        let (grad_z, grad_y, grad_x) = compute_gradient(fixed, dims, spacing);

        // ── Pre-allocated scratch buffers (zero alloc inside the loop) ──
        let mut phi_z = vec![0.0_f32; n];
        let mut phi_y = vec![0.0_f32; n];
        let mut phi_x = vec![0.0_f32; n];
        let mut scratch_ss_z = vec![0.0_f32; n];
        let mut scratch_ss_y = vec![0.0_f32; n];
        let mut scratch_ss_x = vec![0.0_f32; n];
        let mut m_warped = vec![0.0_f32; n];
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];
        let mut smooth_tmp = vec![0.0_f32; n];

        // Initial MSE: φ = 0 (identity) — buffers already zero-initialised.
        let mut final_mse = compute_mse_streaming(fixed, moving, dims, &phi_z, &phi_y, &phi_x);

        let mut iter = 0usize;
        for it in 0..self.config.max_iterations {
            iter = it + 1;

            scaling_and_squaring_into(
                &vel_z,
                &vel_y,
                &vel_x,
                dims,
                self.n_squarings,
                &mut phi_z,
                &mut phi_y,
                &mut phi_x,
                &mut scratch_ss_z,
                &mut scratch_ss_y,
                &mut scratch_ss_x,
            );
            warp_image_into(moving, dims, &phi_z, &phi_y, &phi_x, &mut m_warped);
            thirion_forces_into(
                fixed,
                &m_warped,
                &grad_z,
                &grad_y,
                &grad_x,
                self.config.max_step_length,
                &mut fz,
                &mut fy,
                &mut fx,
            );

            for i in 0..n {
                vel_z[i] += fz[i];
                vel_y[i] += fy[i];
                vel_x[i] += fx[i];
            }

            if self.config.sigma_diffusion > 0.0 {
                gaussian_smooth_with_scratch(
                    &mut vel_z,
                    dims,
                    self.config.sigma_diffusion,
                    &mut smooth_tmp,
                );
                gaussian_smooth_with_scratch(
                    &mut vel_y,
                    dims,
                    self.config.sigma_diffusion,
                    &mut smooth_tmp,
                );
                gaussian_smooth_with_scratch(
                    &mut vel_x,
                    dims,
                    self.config.sigma_diffusion,
                    &mut smooth_tmp,
                );
            }

            // Reuse φ already computed at loop top — avoids redundant
            // scaling-and-squaring allocation that `compute_mse_direct` incurred.
            final_mse = compute_mse_streaming(fixed, moving, dims, &phi_z, &phi_y, &phi_x);
        }

        // Final displacement fields (zero alloc, reuse scratch)
        scaling_and_squaring_into(
            &vel_z,
            &vel_y,
            &vel_x,
            dims,
            self.n_squarings,
            &mut phi_z,
            &mut phi_y,
            &mut phi_x,
            &mut scratch_ss_z,
            &mut scratch_ss_y,
            &mut scratch_ss_x,
        );
        let mut warped = vec![0.0_f32; n];
        warp_image_into(moving, dims, &phi_z, &phi_y, &phi_x, &mut warped);

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
                // Not in a hot loop — allocating variant is acceptable.
                use crate::deformable_field_ops::scaling_and_squaring;
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
