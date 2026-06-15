//! B-Spline FFD registration engine.

use super::basis::{
    evaluate_bspline_displacement, evaluate_bspline_displacement_fast_into, init_control_grid,
    BasisCache,
};
use super::config::{BSplineFFDConfig, BSplineFFDResult};
use super::metric::{
    compute_metric_gradient_fast_into, compute_ncc, MetricGradientScratch, NCC_SIGMA_GUARD,
};
use super::pyramid::refine_control_grid;
use super::regularization::bending_energy_gradient;
use super::volume_dims::VolumeDims;
use crate::deformable_field_ops::{warp_image, warp_image_into, WarpInterpolation};
use crate::error::RegistrationError;

/// B-Spline FFD registration engine.
///
/// Stateless entry point; all parameters are passed via [`BSplineFFDConfig`].
pub struct BSplineFFDRegistration;

impl BSplineFFDRegistration {
    /// Register `moving` to `fixed` using multi-resolution B-Spline FFD.
    ///
    /// # Arguments
    /// - `fixed`   — reference image, flat `&[f32]` in Z-major order.
    /// - `moving`  — moving image, same shape as `fixed`.
    /// - `dims`    — image dimensions `[nz, ny, nx]`.
    /// - `spacing` — physical voxel spacing `[sz, sy, sx]`.
    /// - `config`  — algorithm parameters.
    ///
    /// # Errors
    /// Returns [`RegistrationError`] on dimension mismatch or invalid
    /// configuration.
    pub fn register(
        fixed: &[f32],
        moving: &[f32],
        dims: VolumeDims,
        spacing: [f64; 3],
        config: &BSplineFFDConfig,
    ) -> Result<BSplineFFDResult, RegistrationError> {
        let [nz, ny, nx] = dims.as_array();
        let n = nz * ny * nx;

        // ── Input validation ──────────────────────────────────────────────
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
        if config.num_levels == 0 {
            return Err(RegistrationError::InvalidConfiguration(
                "num_levels must be >= 1".into(),
            ));
        }
        for d in 0..3 {
            if config.initial_control_spacing[d] == 0 {
                return Err(RegistrationError::InvalidConfiguration(format!(
                    "initial_control_spacing[{}] must be >= 1",
                    d
                )));
            }
        }

        // ── Initialize control grid at coarsest level ─────────────────────
        let mut ctrl_spacing = [
            config.initial_control_spacing[0] as f64,
            config.initial_control_spacing[1] as f64,
            config.initial_control_spacing[2] as f64,
        ];
        let mut ctrl_dims = init_control_grid(dims, &ctrl_spacing);
        let ctrl_n = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
        let mut cp_z = vec![0.0_f32; ctrl_n];
        let mut cp_y = vec![0.0_f32; ctrl_n];
        let mut cp_x = vec![0.0_f32; ctrl_n];

        let mut total_iters = 0usize;
        let mut final_metric = 0.0_f64;

        // Pre-allocate image-sized buffers reused across all levels and iterations
        // to eliminate 4 heap allocations per inner-loop iteration.
        let mut disp_z = vec![0.0_f32; n];
        let mut disp_y = vec![0.0_f32; n];
        let mut disp_x = vec![0.0_f32; n];
        let mut warped = vec![0.0_f32; n];
        let mut metric_scratch = MetricGradientScratch::new(dims, ctrl_dims.into());

        // ── Multi-resolution loop ─────────────────────────────────────────
        for level in 0..config.num_levels {
            tracing::info!(
                level,
                ctrl_dims = ?ctrl_dims,
                ctrl_spacing = ?ctrl_spacing,
                "BSpline FFD: starting level"
            );

            // Pre-compute basis cache once per level (dims & ctrl_spacing
            // are constant for all iterations at this level).
            let basis_cache = BasisCache::new(dims, &ctrl_spacing);

            // Re-size scratch buffers when the control grid changed at a level boundary.
            metric_scratch.resize(dims, ctrl_dims.into());

            let mut prev_metric = f64::NEG_INFINITY;

            for iter in 0..config.max_iterations_per_level {
                // 1. Evaluate dense displacement from current control points.
                evaluate_bspline_displacement_fast_into(
                    &cp_z,
                    &cp_y,
                    &cp_x,
                    &ctrl_dims,
                    dims,
                    &basis_cache,
                    &mut disp_z,
                    &mut disp_y,
                    &mut disp_x,
                );

                // 2. Warp moving image.
                warp_image_into(moving, dims, &disp_z, &disp_y, &disp_x, &mut warped);

                // 3. Compute NCC metric.
                let ncc = compute_ncc(fixed, &warped);

                // 4. Convergence check.
                let rel_change = if prev_metric.is_finite() && prev_metric.abs() > NCC_SIGMA_GUARD {
                    ((ncc - prev_metric) / prev_metric.abs()).abs()
                } else {
                    f64::INFINITY
                };

                if rel_change < config.convergence_threshold && iter > 0 {
                    tracing::debug!(iter, ncc, "BSpline FFD: converged");
                    total_iters += iter + 1;
                    final_metric = ncc;
                    break;
                }
                prev_metric = ncc;
                final_metric = ncc;

                // 5. Compute metric gradient w.r.t. control points (zero-alloc path).
                compute_metric_gradient_fast_into(
                    fixed,
                    &warped,
                    ctrl_dims.into(),
                    dims,
                    spacing,
                    &basis_cache,
                    &mut metric_scratch,
                );

                // 6. Compute bending-energy gradients w.r.t. control points.
                let be_grad =
                    bending_energy_gradient(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);

                // 7. Gradient descent update.
                let lr = config.learning_rate as f32;
                let lambda = config.regularization_weight as f32;
                let cn = ctrl_dims[0] * ctrl_dims[1] * ctrl_dims[2];
                for i in 0..cn {
                    // Ascend NCC (maximize), descend bending energy (minimize).
                    cp_z[i] += lr * (metric_scratch.grad_z[i] - lambda * be_grad.z[i]);
                    cp_y[i] += lr * (metric_scratch.grad_y[i] - lambda * be_grad.y[i]);
                    cp_x[i] += lr * (metric_scratch.grad_x[i] - lambda * be_grad.x[i]);
                }

                if iter == config.max_iterations_per_level - 1 {
                    total_iters += config.max_iterations_per_level;
                }
            }

            // Refine control grid for next level (except at the last level).
            if level + 1 < config.num_levels {
                let (new_z, new_y, new_x, new_dims, new_spacing) =
                    refine_control_grid(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing);
                cp_z = new_z;
                cp_y = new_y;
                cp_x = new_x;
                ctrl_dims = new_dims;
                ctrl_spacing = new_spacing;
            }
        }

        // ── Final warp ───────────────────────────────────────────────────
        let disp =
            evaluate_bspline_displacement(&cp_z, &cp_y, &cp_x, &ctrl_dims, &ctrl_spacing, dims);
        let warped_moving = warp_image(
            moving,
            dims,
            &disp.z,
            &disp.y,
            &disp.x,
            WarpInterpolation::Trilinear,
        );

        Ok(BSplineFFDResult {
            control_points: (cp_z, cp_y, cp_x),
            control_grid_dims: ctrl_dims.into(),
            control_spacing: ctrl_spacing,
            warped_moving,
            final_metric,
            num_iterations: total_iters,
        })
    }
}
