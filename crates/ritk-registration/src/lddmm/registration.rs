//! LDDMM registration engine â€” gradient-descent optimisation of vâ‚€.
//!
//! # Memory discipline
//! All scratch buffers are pre-allocated before the iteration loop.
//! The loop body performs **zero heap allocations**; all `_into` variants
//! write into caller-provided buffers. Geodesic integration uses its own
//! pre-allocated scratch set (8n + 3 per-step reuse).

use crate::deformable_field_ops::{
    compute_gradient_into, validate_image_pair, warp_image_into, CpuFieldSmoother, FieldSmoother,
};
use crate::error::RegistrationError;

use super::{
    config::{LddmmConfig, LddmmResult},
    geodesic::integrate_geodesic_into_with_smoother,
};

/// LDDMM registration engine.
///
/// Optimises the initial velocity vâ‚€ of a geodesic in diffeomorphism space
/// to align a moving image to a fixed image under the MSE similarity metric
/// with Sobolev-norm regularisation.
#[derive(Debug, Clone)]
pub struct LddmmRegistration {
    /// Algorithm configuration.
    pub config: LddmmConfig,
}

impl LddmmRegistration {
    /// Create a registration instance with the given configuration.
    pub fn new(config: LddmmConfig) -> Self {
        Self { config }
    }

    /// Register `moving` to `fixed` via LDDMM geodesic shooting.
    ///
    /// Convenience wrapper that constructs a [`CpuFieldSmoother`] internally
    /// and delegates to [`register_with`](LddmmRegistration::register_with).
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<LddmmResult, RegistrationError> {
        let mut smoother = CpuFieldSmoother::new(dims, self.config.kernel_sigma.get());
        self.register_with(fixed, moving, dims, spacing, &mut smoother)
    }

    /// Register `moving` to `fixed` via LDDMM geodesic shooting with a
    /// user-provided [`FieldSmoother`].
    ///
    /// When `smoother` is a [`crate::deformable_field_ops::GpuFieldSmoother`],
    /// the per-iteration momentum, adjoint, and body-force smoothing runs on
    /// the GPU â€” 10â€“50Ã— faster than the CPU path for typical 256Â³ fields.
    ///
    /// # Arguments
    /// - `fixed`   â€” reference image, flat `[f32]` in Z-major order.
    /// - `moving`  â€” moving image, same length as `fixed`.
    /// - `dims`    â€” volume dimensions `[nz, ny, nx]`.
    /// - `spacing` â€” physical voxel spacing `[sz, sy, sx]`.
    /// - `smoother` â€” field smoother (CPU or GPU backend).
    ///
    /// # Errors
    /// Returns [`RegistrationError::DimensionMismatch`] when image lengths
    /// differ from `nz * ny * nx`.
    pub fn register_with(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
        smoother: &mut impl FieldSmoother,
    ) -> Result<LddmmResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        validate_image_pair(fixed, moving, dims)?;

        let cfg = &self.config;
        let lr = cfg.learning_rate as f32;
        let lam = cfg.regularization_weight as f32;

        let mut v0z = vec![0.0_f32; n];
        let mut v0y = vec![0.0_f32; n];
        let mut v0x = vec![0.0_f32; n];

        // â”€â”€ Pre-allocated scratch buffers (zero alloc inside the loop) â”€â”€
        let mut dz = vec![0.0_f32; n];
        let mut dy = vec![0.0_f32; n];
        let mut dx = vec![0.0_f32; n];
        let mut warped = vec![0.0_f32; n];
        let mut gw_z = vec![0.0_f32; n];
        let mut gw_y = vec![0.0_f32; n];
        let mut gw_x = vec![0.0_f32; n];
        let mut bf_z = vec![0.0_f32; n];
        let mut bf_y = vec![0.0_f32; n];
        let mut bf_x = vec![0.0_f32; n];
        // Geodesic integration scratch buffers (per-step reuse)
        let mut gs_mz = vec![0.0_f32; n];
        let mut gs_my = vec![0.0_f32; n];
        let mut gs_mx = vec![0.0_f32; n];
        let mut gs_adz = vec![0.0_f32; n];
        let mut gs_ady = vec![0.0_f32; n];
        let mut gs_adx = vec![0.0_f32; n];
        let mut gs_step_z = vec![0.0_f32; n];
        let mut gs_step_y = vec![0.0_f32; n];
        let mut gs_step_x = vec![0.0_f32; n];
        let mut gs_comp_z = vec![0.0_f32; n];
        let mut gs_comp_y = vec![0.0_f32; n];
        let mut gs_comp_x = vec![0.0_f32; n];

        let mut prev_mse = f64::MAX;
        let mut num_iters = 0_usize;

        for iter in 0..cfg.max_iterations {
            integrate_geodesic_into_with_smoother(
                &v0z,
                &v0y,
                &v0x,
                dims,
                spacing,
                cfg.num_time_steps,
                smoother,
                &mut dz,
                &mut dy,
                &mut dx,
                &mut gs_mz,
                &mut gs_my,
                &mut gs_mx,
                &mut gs_adz,
                &mut gs_ady,
                &mut gs_adx,
                &mut gs_step_z,
                &mut gs_step_y,
                &mut gs_step_x,
                &mut gs_comp_z,
                &mut gs_comp_y,
                &mut gs_comp_x,
            );
            warp_image_into(moving, dims.into(), &dz, &dy, &dx, &mut warped);

            let mse: f64 = warped
                .iter()
                .zip(fixed.iter())
                .map(|(&w, &f)| {
                    let d = (w - f) as f64;
                    d * d
                })
                .sum::<f64>()
                / n as f64;

            num_iters = iter + 1;

            if iter > 0 {
                let rel = (prev_mse - mse).abs() / (prev_mse + 1e-12);
                if rel < cfg.convergence_threshold {
                    break;
                }
            }
            prev_mse = mse;

            compute_gradient_into(
                &warped,
                dims.into(),
                spacing,
                &mut gw_z,
                &mut gw_y,
                &mut gw_x,
            );

            for i in 0..n {
                let residual = 2.0 * (warped[i] - fixed[i]);
                bf_z[i] = residual * gw_z[i];
                bf_y[i] = residual * gw_y[i];
                bf_x[i] = residual * gw_x[i];
            }
            smoother.smooth_field(&mut bf_z, &mut bf_y, &mut bf_x);

            for i in 0..n {
                v0z[i] -= lr * (2.0 * lam * v0z[i] + bf_z[i]);
                v0y[i] -= lr * (2.0 * lam * v0y[i] + bf_y[i]);
                v0x[i] -= lr * (2.0 * lam * v0x[i] + bf_x[i]);
            }
        }

        // Final geodesic (reuses all buffers)
        integrate_geodesic_into_with_smoother(
            &v0z,
            &v0y,
            &v0x,
            dims,
            spacing,
            cfg.num_time_steps,
            smoother,
            &mut dz,
            &mut dy,
            &mut dx,
            &mut gs_mz,
            &mut gs_my,
            &mut gs_mx,
            &mut gs_adz,
            &mut gs_ady,
            &mut gs_adx,
            &mut gs_step_z,
            &mut gs_step_y,
            &mut gs_step_x,
            &mut gs_comp_z,
            &mut gs_comp_y,
            &mut gs_comp_x,
        );
        warp_image_into(moving, dims.into(), &dz, &dy, &dx, &mut warped);
        let final_mse: f64 = warped
            .iter()
            .zip(fixed.iter())
            .map(|(&w, &f)| {
                let d = (w - f) as f64;
                d * d
            })
            .sum::<f64>()
            / n as f64;

        Ok(LddmmResult {
            initial_velocity: (v0z, v0y, v0x),
            displacement_field: (dz, dy, dx),
            warped_moving: warped,
            final_metric: final_mse,
            num_iterations: num_iters,
        })
    }
}
