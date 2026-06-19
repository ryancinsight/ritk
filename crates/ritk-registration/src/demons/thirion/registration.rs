//! Thirion Demons registration struct and iteration loop.

use super::super::config::{DemonsConfig, DemonsResult};
use super::forces::thirion_forces_into;
use crate::deformable_field_ops::{
    compute_gradient, compute_mse_inplace, validate_image_pair, warp_image_into, CpuFieldSmoother,
    FieldSmoother, VectorField, VectorFieldMut, VelocityField,
};
use crate::error::RegistrationError;

/// Thirion Demons registration (classic variant).
///
/// Computes a dense displacement field by iterating optical-flow-like forces
/// derived from image intensity difference and the fixed-image gradient.
///
/// # Smoothing backends
///
/// Use [`register`](ThirionDemonsRegistration::register) for CPU smoothing.
/// Use [`register_with`](ThirionDemonsRegistration::register_with) to pass
/// `GpuFieldSmoother` or custom [`FieldSmoother`] implementations.
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
        let fluid_sigma = self.config.sigma_fluid.map(|s| s.get()).unwrap_or(0.0);
        let diff_sigma = self.config.sigma_diffusion.map(|s| s.get()).unwrap_or(0.0);
        let mut fluid = CpuFieldSmoother::new(dims, fluid_sigma);
        let mut diffusion = CpuFieldSmoother::new(dims, diff_sigma);
        self.register_with(fixed, moving, dims, spacing, &mut fluid, &mut diffusion)
    }

    /// Register `moving` to `fixed` with pluggable [`FieldSmoother`] backends.
    ///
    /// # Arguments
    /// - `fluid` — smoother for fluid regularisation (`sigma_fluid`).
    /// - `diffusion` — smoother for diffusion regularisation (`sigma_diffusion`).
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if `fixed` and `moving` have different
    /// lengths or `dims` are inconsistent.
    pub fn register_with(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
        fluid: &mut impl FieldSmoother,
        diffusion: &mut impl FieldSmoother,
    ) -> Result<DemonsResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        validate_image_pair(fixed, moving, dims)?;

        let mut disp_z = vec![0.0_f32; n];
        let mut disp_y = vec![0.0_f32; n];
        let mut disp_x = vec![0.0_f32; n];

        let grad = compute_gradient(fixed, dims.into(), spacing);

        // m_warped is the iter's working warped buffer.  We initialise it to
        // the identity-warp (a copy of `moving` — the cost of one `n`-element
        // memcpy is amortised across all iterations).  This lets the iter
        // body skip the top warp on every iteration: the first iteration
        // starts with m_warped == moving (identity), and each subsequent
        // iteration re-uses the previous iteration's post-update re-warp.
        let mut m_warped = moving.to_vec();
        let mut final_mse = compute_mse_inplace(fixed, &m_warped);
        let mut iter = 0usize;
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];

        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // m_warped is at the iter's starting disp: identity at iter 0, and
            // the previous iter's post-update re-warp for iter ≥ 1.  Reading
            // it directly (no top warp) saves a `warp_image_into` per iter.
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
                dims,
            );

            // Fluid regularisation
            if self.config.sigma_fluid.is_some() {
                fluid.smooth_field(&mut fz, &mut fy, &mut fx);
            }

            for i in 0..n {
                disp_z[i] += fz[i];
                disp_y[i] += fy[i];
                disp_x[i] += fx[i];
            }

            // Diffusion regularisation
            if self.config.sigma_diffusion.is_some() {
                diffusion.smooth_field(&mut disp_z, &mut disp_y, &mut disp_x);
            }

            // Re-warp m_warped with the post-update disp so the next iter's
            // forces see the new state and this iter's MSE is reported at the
            // post-update (look-ahead) disp — matching the previous code's
            // semantics — without a streaming-warp inside compute_mse_streaming.
            warp_image_into(
                moving,
                dims.into(),
                &disp_z,
                &disp_y,
                &disp_x,
                &mut m_warped,
            );
            final_mse = compute_mse_inplace(fixed, &m_warped);
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
