use super::super::config::{DemonsConfig, DemonsResult};
use super::SymmetricDemonsRegistration;
use crate::deformable_field_ops::{
    compute_gradient_into, validate_image_pair, warp_image_into, CpuFieldSmoother, FieldSmoother };
use crate::error::RegistrationError;

impl SymmetricDemonsRegistration {
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

    /// Register `moving` to `fixed` using symmetric optical-flow forces
    /// with pluggable [`FieldSmoother`] backends.
    ///
    /// # Arguments
    /// - `fluid` â€” smoother for fluid regularisation (`sigma_fluid`).
    /// - `diffusion` â€” smoother for diffusion regularisation (`sigma_diffusion`).
    ///
    /// # Errors
    /// Returns [`RegistrationError`] if image lengths are inconsistent with `dims`.
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

        // Initialise displacement field to zero.
        let mut disp_z = vec![0.0_f32; n];
        let mut disp_y = vec![0.0_f32; n];
        let mut disp_x = vec![0.0_f32; n];

        // Pre-compute fixed-image gradient (constant across iterations).
        let mut grad_fz = vec![0.0_f32; n];
        let mut grad_fy = vec![0.0_f32; n];
        let mut grad_fx = vec![0.0_f32; n];
        compute_gradient_into(
            fixed,
            dims.into(),
            spacing,
            &mut grad_fz,
            &mut grad_fy,
            &mut grad_fx,
        );

        let sigma_x = self.config.max_step_length;
        let sigma_x2 = sigma_x * sigma_x;

        // Pre-allocated scratch buffers (zero alloc inside the loop).
        let mut m_warped = vec![0.0_f32; n];
        let mut grad_mz = vec![0.0_f32; n];
        let mut grad_my = vec![0.0_f32; n];
        let mut grad_mx = vec![0.0_f32; n];
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];

        // Initial MSE: D = 0 (identity).
        let mut final_mse: f64 = fixed
            .iter()
            .zip(moving.iter())
            .map(|(&f, &m)| ((f - m) as f64).powi(2))
            .sum::<f64>()
            / n as f64;

        let mut iter = 0usize;
        for it in 0..self.config.max_iterations {
            iter = it + 1;

            // 1. Warp moving with current displacement.
            warp_image_into(
                moving,
                dims.into(),
                &disp_z,
                &disp_y,
                &disp_x,
                &mut m_warped,
            );

            // 2. Compute gradient of the warped moving image.
            compute_gradient_into(
                &m_warped,
                dims.into(),
                spacing,
                &mut grad_mz,
                &mut grad_my,
                &mut grad_mx,
            );

            // 3. Compute symmetric forces.
            symmetric_forces_into(
                fixed,
                &m_warped,
                &grad_fz,
                &grad_fy,
                &grad_fx,
                &grad_mz,
                &grad_my,
                &grad_mx,
                sigma_x2,
                self.config.max_step_length,
                &mut fz,
                &mut fy,
                &mut fx,
            );

            // 4. Fluid regularisation.
            if self.config.sigma_fluid.is_some() {
                fluid.smooth_field(&mut fz, &mut fy, &mut fx);
            }

            // 5. Accumulate displacement field.
            for i in 0..n {
                disp_z[i] += fz[i];
                disp_y[i] += fy[i];
                disp_x[i] += fx[i];
            }

            // 6. Diffusive regularisation.
            if self.config.sigma_diffusion.is_some() {
                diffusion.smooth_field(&mut disp_z, &mut disp_y, &mut disp_x);
            }

            // 7. Compute MSE from m_warped.
            final_mse = fixed
                .iter()
                .zip(m_warped.iter())
                .map(|(&f, &m)| ((f - m) as f64).powi(2))
                .sum::<f64>()
                / n as f64;
        }

        let mut warped = vec![0.0_f32; n];
        warp_image_into(moving, dims.into(), &disp_z, &disp_y, &disp_x, &mut warped);

        Ok(DemonsResult {
            warped,
            disp_z,
            disp_y,
            disp_x,
            vel_z: None,
            vel_y: None,
            vel_x: None,
            final_mse,
            num_iterations: iter })
    }
}

// â”€â”€ Private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[allow(clippy::too_many_arguments)]
fn symmetric_forces_into(
    fixed: &[f32],
    m_warped: &[f32],
    grad_fz: &[f32],
    grad_fy: &[f32],
    grad_fx: &[f32],
    grad_mz: &[f32],
    grad_my: &[f32],
    grad_mx: &[f32],
    sigma_x2: f32,
    max_step_length: f32,
    fz: &mut [f32],
    fy: &mut [f32],
    fx: &mut [f32],
) {
    let n = fixed.len();
    let max2 = max_step_length * max_step_length;

    for i in 0..n {
        let diff = fixed[i] - m_warped[i];
        let cgz = grad_fz[i] + grad_mz[i];
        let cgy = grad_fy[i] + grad_my[i];
        let cgx = grad_fx[i] + grad_mx[i];
        let grad_sq_over4 = (cgz * cgz + cgy * cgy + cgx * cgx) * 0.25;
        let denom = grad_sq_over4 + diff * diff / sigma_x2 + 1e-5;
        let scale = diff / denom;

        let mut fzi = scale * cgz;
        let mut fyi = scale * cgy;
        let mut fxi = scale * cgx;

        let mag2 = fzi * fzi + fyi * fyi + fxi * fxi;
        if mag2 > max2 {
            let s = max_step_length / mag2.sqrt();
            fzi *= s;
            fyi *= s;
            fxi *= s;
        }

        fz[i] = fzi;
        fy[i] = fyi;
        fx[i] = fxi;
    }
}
