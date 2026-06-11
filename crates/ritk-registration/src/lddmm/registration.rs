//! LDDMM registration engine — gradient-descent optimisation of v₀.
//!
//! # Memory discipline
//! All scratch buffers are pre-allocated before the iteration loop.
//! The loop body performs **zero heap allocations**; all `_into` variants
//! write into caller-provided buffers. Geodesic integration uses its own
//! pre-allocated scratch set (8n + 3 per-step reuse).

use crate::deformable_field_ops::{
    compute_gradient_into, gaussian_smooth_field_inplace_with_scratch, warp_image_into,
};
use crate::error::RegistrationError;

use super::{
    config::{LddmmConfig, LddmmResult},
    geodesic::integrate_geodesic_into,
};

/// LDDMM registration engine.
///
/// Optimises the initial velocity v₀ of a geodesic in diffeomorphism space
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
    /// # Arguments
    /// - `fixed`   — reference image, flat `[f32]` in Z-major order.
    /// - `moving`  — moving image, same length as `fixed`.
    /// - `dims`    — volume dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    ///
    /// # Errors
    /// Returns [`RegistrationError::DimensionMismatch`] when image lengths
    /// differ from `nz * ny * nx`.
    pub fn register(
        &self,
        fixed: &[f32],
        moving: &[f32],
        dims: [usize; 3],
        spacing: [f64; 3],
    ) -> Result<LddmmResult, RegistrationError> {
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;

        if fixed.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "fixed length {} != nz*ny*nx = {}",
                fixed.len(),
                n
            )));
        }
        if moving.len() != n {
            return Err(RegistrationError::DimensionMismatch(format!(
                "moving length {} != nz*ny*nx = {}",
                moving.len(),
                n
            )));
        }

        let cfg = &self.config;
        let lr = cfg.learning_rate as f32;
        let lam = cfg.regularization_weight as f32;

        let mut v0z = vec![0.0_f32; n];
        let mut v0y = vec![0.0_f32; n];
        let mut v0x = vec![0.0_f32; n];

        // ── Pre-allocated scratch buffers (zero alloc inside the loop) ──
        // Displacement field from geodesic
        let mut dz = vec![0.0_f32; n];
        let mut dy = vec![0.0_f32; n];
        let mut dx = vec![0.0_f32; n];
        // Warped moving image
        let mut warped = vec![0.0_f32; n];
        // Gradient of warped
        let mut gw_z = vec![0.0_f32; n];
        let mut gw_y = vec![0.0_f32; n];
        let mut gw_x = vec![0.0_f32; n];
        // Body force buffers
        let mut bf_z = vec![0.0_f32; n];
        let mut bf_y = vec![0.0_f32; n];
        let mut bf_x = vec![0.0_f32; n];
        // Smooth scratch
        let mut smooth_tmp = vec![0.0_f32; n];
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
            integrate_geodesic_into(
                &v0z,
                &v0y,
                &v0x,
                dims,
                spacing,
                cfg.num_time_steps,
                cfg.kernel_sigma.get(),
                &mut dz,
                &mut dy,
                &mut dx,
                &mut smooth_tmp,
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

            // ACCUMULATOR: f64 — sum over n voxels; f32 would lose precision.
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

            // Body force: K_σ ∗ [2 (warped − fixed) · ∇(warped)]
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
            gaussian_smooth_field_inplace_with_scratch(
                &mut bf_z,
                &mut bf_y,
                &mut bf_x,
                dims.into(),
                cfg.kernel_sigma.get(),
                &mut smooth_tmp,
            );

            for i in 0..n {
                v0z[i] -= lr * (2.0 * lam * v0z[i] + bf_z[i]);
                v0y[i] -= lr * (2.0 * lam * v0y[i] + bf_y[i]);
                v0x[i] -= lr * (2.0 * lam * v0x[i] + bf_x[i]);
            }
        }

        // Final geodesic integration for output displacement (reuses all buffers)
        integrate_geodesic_into(
            &v0z,
            &v0y,
            &v0x,
            dims,
            spacing,
            cfg.num_time_steps,
            cfg.kernel_sigma.get(),
            &mut dz,
            &mut dy,
            &mut dx,
            &mut smooth_tmp,
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
        // ACCUMULATOR: f64 — sum over n voxels; f32 would lose precision.
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
