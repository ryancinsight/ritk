use super::super::config::{DemonsConfig, DemonsResult};
use super::SymmetricDemonsRegistration;
use crate::deformable_field_ops::{
    compute_gradient_into, gaussian_smooth_field_inplace_with_scratch, warp_image_into,
    GpuFieldSmoother,
};
use crate::error::RegistrationError;
use burn::tensor::backend::Backend;

impl SymmetricDemonsRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: DemonsConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` with **GPU-accelerated** Gaussian field
    /// smoothing for both fluid regularisation and diffusion regularisation.
    ///
    /// The [`GpuFieldSmoother`] instances manage pre-allocated staging tensors
    /// and a single [`ritk_filter::GaussianFilter`] each, reused across all
    /// iterations.
    ///
    /// # Arguments
    /// - `gpu_fluid_smoother` — GPU smoother for fluid regularisation
    ///   (`sigma_fluid`). Called on the per-iteration forces.
    /// - `gpu_diffusion_smoother` — GPU smoother for diffusion regularisation
    ///   (`sigma_diffusion`). Called on the accumulated displacement field.
    pub fn register_with_gpu_smoother<B: Backend>(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
        gpu_fluid_smoother: &mut GpuFieldSmoother<B>,
        gpu_diffusion_smoother: &mut GpuFieldSmoother<B>,
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

            // 4. GPU-accelerated fluid regularisation.
            if self.config.sigma_fluid.is_some() {
                gpu_fluid_smoother.smooth_field_inplace(&mut fz, &mut fy, &mut fx);
            }

            // 5. Accumulate displacement field.
            for i in 0..n {
                disp_z[i] += fz[i];
                disp_y[i] += fy[i];
                disp_x[i] += fx[i];
            }

            // 6. GPU-accelerated diffusive regularisation.
            if self.config.sigma_diffusion.is_some() {
                gpu_diffusion_smoother.smooth_field_inplace(&mut disp_z, &mut disp_y, &mut disp_x);
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
            num_iterations: iter,
        })
    }

    /// Register `moving` to `fixed` using symmetric optical-flow forces.
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

        // ── Pre-allocated scratch buffers (zero alloc inside the loop) ──
        let mut m_warped = vec![0.0_f32; n];
        let mut grad_mz = vec![0.0_f32; n];
        let mut grad_my = vec![0.0_f32; n];
        let mut grad_mx = vec![0.0_f32; n];
        let mut fz = vec![0.0_f32; n];
        let mut fy = vec![0.0_f32; n];
        let mut fx = vec![0.0_f32; n];
        let mut smooth_tmp = vec![0.0_f32; n];

        // Initial MSE: D = 0 (identity) — displacement buffers already zero.
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

            // 2. Compute gradient of the warped moving image (changes each iteration).
            compute_gradient_into(
                &m_warped,
                dims.into(),
                spacing,
                &mut grad_mz,
                &mut grad_my,
                &mut grad_mx,
            );

            // 3. Compute symmetric forces into pre-allocated buffers.
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

            // 4. Optional fluid regularisation (smooth forces before accumulation).
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

            // 5. Accumulate displacement field.
            for i in 0..n {
                disp_z[i] += fz[i];
                disp_y[i] += fy[i];
                disp_x[i] += fx[i];
            }

            // 6. Diffusive regularisation (smooth total field).
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

            // 7. Compute MSE from m_warped (already computed at step 1).
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
            num_iterations: iter,
        })
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Write symmetric Demons forces into caller-provided buffers.
///
/// Force formula:
/// f(p) = diff · (gF + gM_w) / (|gF + gM_w|² / 4 + diff² / σₓ² + ε)
///
/// where `diff = F(p) − M_w(p)`, `gF = ∇F(p)`, `gM_w = ∇M_w(p)`.
///
/// Performs zero heap allocation. All output buffers must have length `fixed.len()`.
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
