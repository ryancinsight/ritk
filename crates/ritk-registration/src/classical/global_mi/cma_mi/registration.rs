// ─── Registration ─────────────────────────────────────────────────────────────

use ritk_image::tensor::AutodiffBackend;
use ritk_image::tensor::{Tensor, TensorData};
use ritk_core::image::Image;
use ritk_transform::RigidTransform;

use super::super::registration::GlobalMiRegistration;
use super::super::transforms::rigid_matrix_to_homogeneous;
use super::config::{CmaMiConfig, InitStrategy};
use super::helpers::run_cma_level;
use super::result::CmaMiResult;

/// CMA-ES → RSGD cascade rigid registration pipeline.
///
/// Performs a two-phase search:
/// 1. **Global search** via CMA-ES on a coarse multi-resolution level (or a
///    coarse-to-fine cascade when `config.pyramid_schedule` is non-empty) to
///    obtain a basin-of-attraction estimate robust to local MI maxima.
/// 2. **Local refinement** (optional) via [`GlobalMiRegistration`] RSGD,
///    warm-started from the CMA-ES solution.
///
/// # Performance: Autodiff Stripping
///
/// CMA-ES is derivative-free — it never calls `.backward()`. Evaluating MI
/// on `Image<Autodiff<B>, D>` would silently build an autodiff graph on every
/// objective evaluation, wasting 2–5× CPU time on tape bookkeeping. This
/// implementation converts pyramid images to `Image<B::InnerBackend, 3>` and
/// evaluates `MutualInformation<B::InnerBackend>` inside the CMA-ES loop.
pub struct CmaMiRegistration;

impl CmaMiRegistration {
    /// Execute CMA-ES + optional RSGD cascade rigid registration.
    ///
    /// Thin wrapper around [`register_rigid_with_mask`](Self::register_rigid_with_mask)
    /// with no brain mask (uniform stochastic sampling over all voxels).
    pub fn register_rigid<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_rotation: [f64; 3],
        initial_translation: Option<[f64; 3]>,
        config: &CmaMiConfig,
    ) -> (RigidTransform<B, 3>, CmaMiResult) {
        Self::register_rigid_with_mask(
            fixed,
            moving,
            initial_rotation,
            initial_translation,
            config,
            None,
        )
    }

    /// Execute CMA-ES + optional RSGD cascade rigid registration with an optional
    /// brain mask.
    ///
    /// # Arguments
    ///
    /// * `fixed` — Reference image (any modality).
    /// * `moving` — Image to be aligned (may differ in modality from `fixed`).
    /// * `initial_rotation` — Starting Euler angles `[alpha, beta, gamma]` in
    ///   radians (ZYX convention). Pass `[0,0,0]` for no prior knowledge.
    /// * `initial_translation` — Starting translation `[tz, ty, tx]` in mm
    ///   (RITK `[z, y, x]` order). `None` → center-of-mass estimate when
    ///   `config.init_strategy == InitStrategy::CenterOfMass`.
    /// * `config` — Pipeline configuration; see [`CmaMiConfig`].
    /// * `fixed_mask` — Optional binary brain mask in fixed-image space. When
    ///   `Some`, only voxels where `mask > 0.5` contribute to MI estimation at
    ///   each pyramid level (ANTs/ITK strategy). The mask is downsampled to
    ///   match each pyramid level using the same shrink factors as the images
    ///   (no smoothing to preserve binary character). When `None`, uniform
    ///   stochastic sampling over all voxels is used (the default behaviour).
    ///
    /// # Returns
    ///
    /// `(transform, result)` where `transform` is the final [`RigidTransform`]
    /// and `result` contains diagnostics.
    pub fn register_rigid_with_mask<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_rotation: [f64; 3],
        initial_translation: Option<[f64; 3]>,
        config: &CmaMiConfig,
        fixed_mask: Option<&Image<B, 3>>,
    ) -> (RigidTransform<B, 3>, CmaMiResult) {
        let t_init: [f64; 3] = match initial_translation {
            Some(t) => t,
            None => {
                if config.init_strategy == InitStrategy::CenterOfMass {
                    super::super::center_of_mass::translation_from_centers_of_mass(fixed, moving)
                } else {
                    [0.0; 3]
                }
            }
        };

        tracing::info!(
            "CmaMiRegistration: t_init = [{:.2}, {:.2}, {:.2}] mm (CoM init = {})",
            t_init[0],
            t_init[1],
            t_init[2],
            initial_translation.is_none() && config.init_strategy == InitStrategy::CenterOfMass,
        );

        // ── Normalised parameter space ────────────────────────────────────────
        // x = [α_n, β_n, γ_n, tz_n, ty_n, tx_n]; each component ∈ [−1, 1]
        let rot_scale = config.rotation_range_rad;
        let trans_scale = config.translation_range_mm;
        let center_arr = [0.0f32; 3];

        let x0: [f64; 6] = [
            initial_rotation[0] / rot_scale,
            initial_rotation[1] / rot_scale,
            initial_rotation[2] / rot_scale,
            t_init[0] / trans_scale,
            t_init[1] / trans_scale,
            t_init[2] / trans_scale,
        ];

        if let Some(mask) = fixed_mask {
            let mask_shape = mask.shape();
            let fixed_shape = fixed.shape();
            assert_eq!(
                mask_shape, fixed_shape,
                "fixed_mask shape {:?} must equal fixed image shape {:?}",
                mask_shape, fixed_shape
            );
        }

        // ── Phase 1: CMA-ES global search ────────────────────────────────────
        let cma_result = if config.pyramid_schedule.is_empty() {
            // ── Single-level path ─────────────────────────────────────────────
            let per_axis = config.shrink_per_axis.unwrap_or([config.coarse_shrink; 3]);
            run_cma_level(
                fixed,
                moving,
                config,
                &per_axis,
                config.coarse_sigma_mm,
                config.cma_config.sigma0,
                config.cma_config.max_generations,
                config.cma_config.lambda,
                config.ipop_restarts,
                rot_scale,
                trans_scale,
                center_arr,
                &x0,
                fixed_mask,
            )
        } else {
            // ── Multi-scale cascade path ──────────────────────────────────────
            let mut current_x: Vec<f64> = x0.to_vec();
            let mut last_result: Option<crate::optimizer::CmaEsResult> = None;

            for (level_idx, level) in config.pyramid_schedule.iter().enumerate() {
                let per_axis = level.shrink_per_axis.unwrap_or([level.shrink; 3]);

                tracing::info!(
                    "CmaMiRegistration: cascade level {} — \
                     shrink={:?}, sigma_mm={:.1}, sigma0={:.3}, max_gen={}",
                    level_idx,
                    per_axis,
                    level.sigma_mm.get(),
                    level.cma_sigma0,
                    level.max_generations,
                );

                let level_result = run_cma_level(
                    fixed,
                    moving,
                    config,
                    &per_axis,
                    level.sigma_mm,
                    level.cma_sigma0,
                    level.max_generations,
                    level.lambda,
                    level.ipop_restarts,
                    rot_scale,
                    trans_scale,
                    center_arr,
                    &current_x,
                    fixed_mask,
                );

                tracing::info!(
                    "CmaMiRegistration: cascade level {} done — \
                     gens={}, best_f={:.6e}, stop={:?}",
                    level_idx,
                    level_result.generations,
                    level_result.best_f,
                    level_result.stop_reason,
                );

                // Seed the next pyramid level with this level's solution.  Clone
                // rather than `mem::take` so `level_result` retains its `best_x`:
                // the final iteration's result is read back below as `cma_result`
                // to reconstruct the transform.  Taking here would leave the
                // stored result with an empty `best_x` and panic on `best[0]`.
                current_x = level_result.best_x.clone();
                last_result = Some(level_result);
            }

            last_result.expect("pyramid_schedule was non-empty but produced no result")
        };

        tracing::info!(
            "CmaMiRegistration: CMA-ES finished — \
             generations={}, best_f={:.6e}, stop={:?}, sigma={:.3e}",
            cma_result.generations,
            cma_result.best_f,
            cma_result.stop_reason,
            cma_result.final_sigma,
        );

        // ── Reconstruct best CMA-ES transform (autodiff backend) ─────────────
        let device = fixed.data().device();
        let best = &cma_result.best_x;

        let b_alpha = (best[0].clamp(-1.0, 1.0) * rot_scale) as f32;
        let b_beta = (best[1].clamp(-1.0, 1.0) * rot_scale) as f32;
        let b_gamma = (best[2].clamp(-1.0, 1.0) * rot_scale) as f32;
        let b_tz = (best[3].clamp(-1.0, 1.0) * trans_scale) as f32;
        let b_ty = (best[4].clamp(-1.0, 1.0) * trans_scale) as f32;
        let b_tx = (best[5].clamp(-1.0, 1.0) * trans_scale) as f32;

        let b_rotation =
            Tensor::<B, 1>::from_data(TensorData::from([b_alpha, b_beta, b_gamma]), &device);
        let b_translation =
            Tensor::<B, 1>::from_data(TensorData::from([b_tz, b_ty, b_tx]), &device);
        let b_center = Tensor::<B, 1>::zeros([3], &device);

        let cma_transform = RigidTransform::<B, 3>::new(b_translation, b_rotation, b_center);

        // ── Optional RSGD refinement (uses autodiff backend) ──────────────────
        let (final_transform, rsgd_iterations, rsgd_loss_history) = if let Some(ref rsgd_cfg) =
            config.rsgd_refine
        {
            tracing::info!(
                "CmaMiRegistration: starting RSGD refinement ({} levels, {} bins)",
                rsgd_cfg.num_levels,
                rsgd_cfg.num_mi_bins,
            );

            let (rsgd_transform, rsgd_result) =
                GlobalMiRegistration::register_rigid_full(fixed, moving, cma_transform, rsgd_cfg);

            let iters: usize = rsgd_result.iterations_per_level.iter().sum();

            tracing::info!(
                "CmaMiRegistration: RSGD complete — {} iters, final MI = {:.6e}",
                iters,
                rsgd_result.final_mi,
            );

            (rsgd_transform, iters, rsgd_result.loss_history)
        } else {
            (
                cma_transform,
                0_usize,
                Vec::with_capacity(config.cma_config.max_generations),
            )
        };

        // ── Assemble result ───────────────────────────────────────────────────
        let final_mi = -cma_result.best_f; // negate: CMA-ES minimises −MI
        let result = CmaMiResult {
            matrix: rigid_matrix_to_homogeneous(&final_transform),
            final_mi,
            cma_generations: cma_result.generations,
            cma_stop_reason: cma_result.stop_reason,
            cma_final_sigma: cma_result.final_sigma,
            rsgd_iterations,
            rsgd_loss_history,
            cma_best_params: cma_result.best_x.clone(),
        };

        (final_transform, result)
    }
}
