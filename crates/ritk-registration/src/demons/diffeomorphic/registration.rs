//! Diffeomorphic Demons registration struct and iteration loop.

use super::super::config::{DemonsConfig, DemonsResult};
use super::super::inverse::invert_velocity_field;
use super::super::thirion::thirion_forces_into;
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_inplace, scaling_and_squaring, scaling_and_squaring_into,
    validate_image_pair, warp_image_into, CpuFieldSmoother, FieldSmoother, VectorField,
    VectorFieldMut, VelocityField };
use crate::error::RegistrationError;

/// Diffeomorphic Demons registration using stationary velocity fields.
///
/// Produces topologically correct (invertible) deformation fields by
/// representing the transformation as the exponential map of a velocity field.
///
/// # Smoothing backends
///
/// Use [`register`](DiffeomorphicDemonsRegistration::register) for CPU
/// smoothing.  Use [`register_with`](DiffeomorphicDemonsRegistration::register_with)
/// to pass a `GpuFieldSmoother` or custom [`FieldSmoother`].
#[derive(Debug, Clone)]
pub struct DiffeomorphicDemonsRegistration {
    /// Algorithm configuration (shared with Thirion Demons).
    pub config: DemonsConfig,
    /// Number of scaling-and-squaring steps for computing `exp(v)`.
    /// Standard value: 6 (2â¶ = 64 integration steps).
    pub n_squarings: usize }

impl DiffeomorphicDemonsRegistration {
    /// Create a registration instance with the given configuration.
    ///
    /// `n_squarings` defaults to 6.
    pub fn new(config: DemonsConfig) -> Self {
        Self {
            config,
            n_squarings: 6 }
    }

    /// Create with a custom number of squaring steps.
    pub fn with_squarings(config: DemonsConfig, n_squarings: usize) -> Self {
        Self {
            config,
            n_squarings }
    }

    /// Register `moving` to `fixed` with CPU Gaussian field smoothing.
    ///
    /// Convenience wrapper.  Prefer [`register_with`](Self::register_with)
    /// when a GPU backend is available.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<DemonsResult, RegistrationError> {
        let sigma = self.config.sigma_diffusion.map(|s| s.get()).unwrap_or(0.0);
        let mut smoother = CpuFieldSmoother::new(dims, sigma);
        self.register_with(fixed, moving, dims, spacing, &mut smoother)
    }

    /// Register `moving` to `fixed` using a stationary velocity field with
    /// a pluggable [`FieldSmoother`] backend.
    ///
    /// # Arguments
    /// - `smoother` â€” field smoother.  Its sigma must match
    ///   `self.config.sigma_diffusion`.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if image lengths are inconsistent with `dims`.
    pub fn register_with(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
        smoother: &mut impl FieldSmoother,
    ) -> Result<DemonsResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        validate_image_pair(fixed, moving, dims)?;

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

        let grad = compute_gradient(fixed, dims.into(), spacing);

        // m_warped is the iter's working warped buffer.  Initialise to
        // identity-warp (== moving) so iter 0's forces see the unwarped
        // moving image.  Each subsequent iter re-uses the previous iter's
        // post-update re-warp, eliminating the redundant top warp per iter.
        let mut m_warped = moving.to_vec();
        let mut final_mse = compute_mse_inplace(fixed, &m_warped);
        let mut iter = 0usize;

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // m_warped is at the iter's starting phi: identity (== moving) at
            // iter 0, and the previous iter's post-update re-warp for iter â‰¥ 1.
            // Reading it directly (no top S&S phi, no top warp) saves a
            // `scaling_and_squaring_into` + `warp_image_into` per iter â€” the
            // dominant cost of the previous top-warp path on 256Â³ fields.
            thirion_forces_into(
                fixed,
                &m_warped,
                VectorField {
                    z: &grad.z,
                    y: &grad.y,
                    x: &grad.x },
                self.config.max_step_length,
                VectorFieldMut {
                    z: &mut fz,
                    y: &mut fy,
                    x: &mut fx },
                dims,
            );

            for i in 0..n {
                vel_z[i] += fz[i];
                vel_y[i] += fy[i];
                vel_x[i] += fx[i];
            }

            // Diffusion regularisation â€” dispatched via FieldSmoother trait
            if self.config.sigma_diffusion.is_some() {
                smoother.smooth_field(&mut vel_z, &mut vel_y, &mut vel_x);
            }

            // BOTTOM S&S phi (with the post-update velocity) + re-warp
            // m_warped so the next iter's forces and this iter's final_mse
            // see the post-update state â€” matching the previous code's
            // semantics â€” without a streaming-warp inside compute_mse_streaming.
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
            final_mse = compute_mse_inplace(fixed, &m_warped);
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
            num_iterations: iter })
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
