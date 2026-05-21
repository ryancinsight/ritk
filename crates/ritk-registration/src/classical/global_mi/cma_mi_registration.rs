//! CMA-ES → RSGD cascade registration using Mattes Mutual Information.
//!
//! # Motivation: The Local-Maxima Problem in MI-Based Registration
//!
//! Multi-modal image registration by mutual information maximisation is
//! fundamentally a non-convex optimisation problem. The MI landscape
//! `I(θ) : ℝ⁶ → ℝ` over rigid-body parameters exhibits numerous local
//! maxima caused by:
//!
//! - **Aliasing**: when voxel grids of the two modalities are commensurate
//!   along a given axis, a false MI peak appears at every integer-voxel offset.
//! - **Symmetry**: anatomically symmetric structures (e.g. bilateral brain
//!   hemispheres) produce mirror-image MI peaks.
//! - **Parzen window width**: wide bins smooth the landscape and help escape
//!   local optima, but the optimal landscape is still non-convex at coarse
//!   resolution.
//!
//! Gradient-based methods such as RSGD / Adam are susceptible to local-maxima
//! trapping and are highly sensitive to initialisation.  The [`GlobalMiRegistration`]
//! pipeline (RSGD) therefore benefits from a good starting point.
//!
//! # Solution: CMA-ES Global Search + RSGD Local Refinement
//!
//! This module implements a two-phase cascade:
//!
//! ```text
//! Phase 0: Center-of-mass initialisation (zeroth-order translation estimate)
//! Phase 1: CMA-ES global search on a coarse pyramid level
//!          ├─ Derivative-free evolution strategy — can cross MI valleys
//!          ├─ Normalised parameter space  [−1, 1]⁶  for uniform scaling
//!          └─ Minimises −MI(A, B∘T; θ) over 6-DOF rigid parameters
//! Phase 2 (optional): RSGD fine refinement on the full-resolution image
//!                      using the CMA-ES solution as warm start
//! ```
//!
//! # Why CMA-ES?
//!
//! The (μ/μ_w, λ)-CMA-ES (Hansen & Ostermeier 2001) is an evolutionary
//! strategy that adapts a full covariance matrix to the curvature of the
//! objective. Key properties for registration:
//!
//! - **No gradients required**: suitable for metrics whose autodiff graph is
//!   expensive to compute at full resolution.
//! - **Invariant to rotation of parameter space**: affine parameter rescaling
//!   does not affect convergence speed.
//! - **Population-based**: evaluates `λ` candidates per generation, naturally
//!   exploring multiple modes of the MI landscape simultaneously.
//! - **Self-adaptive step size**: the step-size control (CSA / cumulative
//!   path-length control) prevents premature convergence.
//!
//! IPOP-CMA-ES (Auger & Hansen 2005) with increasing population offers
//! stronger global guarantees but is not implemented here; the coarse-pyramid
//! strategy (large `coarse_shrink`) achieves a similar smoothing effect at
//! lower computational cost.
//!
//! # References
//!
//! - Hansen, N., & Ostermeier, A. (2001). Completely derandomized self-adaptation
//!   in evolution strategies. *Evol. Comput.* 9(2):159–195.
//!   DOI: 10.1162/106365601750190398
//! - Auger, A., & Hansen, N. (2005). A restart CMA evolution strategy with
//!   increasing population size. *CEC 2005*, vol. 2, pp. 1769–1776.
//!   DOI: 10.1109/CEC.2005.1554902
//! - Klein, S., et al. (2007). Evaluation of optimization methods for
//!   nonrigid medical image registration using mutual information and
//!   B-splines. *IEEE Trans. Image Process.* 16(12):2879–2890.
//!   DOI: 10.1109/TIP.2007.909412

use burn::tensor::backend::AutodiffBackend;
use burn::tensor::{Tensor, TensorData};
use ritk_core::filter::pyramid::MultiResolutionPyramid;
use ritk_core::image::Image;
use ritk_core::transform::RigidTransform;

use super::config::GlobalMiConfig;
use super::registration::GlobalMiRegistration;
use super::transforms::{estimate_intensity_range, rigid_matrix_to_homogeneous};
use crate::metric::{Metric, MutualInformation};
use crate::optimizer::{CmaEsConfig, CmaEsOptimizer, StopReason};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the CMA-ES → RSGD cascade registration pipeline.
///
/// Tune `coarse_shrink` first: a factor of 8 on a typical 256³ brain image
/// yields 32³ pyramid levels (~32k voxels), keeping each MI evaluation
/// under ~1 ms on CPU.  Increase `cma_config.max_generations` if the search
/// stalls before convergence.
#[derive(Debug, Clone)]
pub struct CmaMiConfig {
    /// CMA-ES solver settings.
    ///
    /// Defaults are tuned for 6-DOF rigid registration on a coarse pyramid:
    /// `sigma0 = 0.3` (30% of normalised range), `max_generations = 400`.
    pub cma_config: CmaEsConfig,

    /// Isotropic shrink factor applied to both images for the CMA-ES search
    /// level. Larger values speed up each MI evaluation at the cost of spatial
    /// detail. Default: **8** (e.g. 256³ → 32³).
    pub coarse_shrink: usize,

    /// Gaussian smoothing sigma applied before downsampling, in physical units
    /// (mm). Reduces aliasing at the coarse level. Default: **4.0 mm**.
    pub coarse_sigma_mm: f64,

    /// Number of histogram bins for Mattes MI estimation. Default: **32**.
    pub num_mi_bins: usize,

    /// Fraction of voxels randomly sampled per MI evaluation ∈ (0, 1].
    /// Sparse sampling dramatically reduces wall time at the cost of
    /// gradient variance. Default: **0.15** (15%).
    pub sampling_percentage: f32,

    /// Half-range for translation parameters in mm. The CMA-ES searches
    /// `t ∈ [−range, +range]` after normalisation. Default: **60.0 mm**.
    pub translation_range_mm: f64,

    /// Half-range for rotation parameters in radians. Default: **π/4**.
    pub rotation_range_rad: f64,

    /// Optional RSGD fine-refinement configuration. When `Some`, a
    /// [`GlobalMiRegistration`] run is started from the CMA-ES solution.
    /// Default: **None** (CMA-ES result used directly).
    pub rsgd_refine: Option<GlobalMiConfig>,

    /// When `true` and `initial_translation` is `None`, automatically compute
    /// a center-of-mass translation to pre-align the images. Default: **true**.
    pub use_com_init: bool,
}

impl Default for CmaMiConfig {
    fn default() -> Self {
        Self {
            cma_config: CmaEsConfig {
                sigma0: 0.3,
                lambda: 0,
                max_generations: 400,
                sigma_tol: 1e-8,
                ftol: 1e-12,
                seed: 0xcafe_babe_dead_beef,
                record_history: false,
            },
            coarse_shrink: 8,
            coarse_sigma_mm: 4.0,
            num_mi_bins: 32,
            sampling_percentage: 0.15,
            translation_range_mm: 60.0,
            rotation_range_rad: std::f64::consts::FRAC_PI_4,
            rsgd_refine: None,
            use_com_init: true,
        }
    }
}

// ─── Result ───────────────────────────────────────────────────────────────────

/// Result produced by [`CmaMiRegistration::register_rigid`].
#[derive(Debug, Clone)]
pub struct CmaMiResult {
    /// 4×4 homogeneous matrix of the final transform (row-major, f64).
    pub matrix: [f64; 16],

    /// Final Mattes MI value (positive; negated from the CMA-ES loss).
    /// Note: this reflects the CMA-ES coarse-level MI, not the full-resolution
    /// value, even when RSGD refinement is applied.
    pub final_mi: f64,

    /// Number of CMA-ES generations executed.
    pub cma_generations: usize,

    /// Reason the CMA-ES loop terminated.
    pub cma_stop_reason: StopReason,

    /// CMA-ES final step-size σ.
    pub cma_final_sigma: f64,

    /// Total RSGD iterations across all resolution levels (0 if no refinement).
    pub rsgd_iterations: usize,

    /// Per-iteration loss history from RSGD refinement (empty if no refinement).
    pub rsgd_loss_history: Vec<f64>,
}

// ─── Registration ─────────────────────────────────────────────────────────────

/// CMA-ES → RSGD cascade rigid registration pipeline.
///
/// Performs a two-phase search:
/// 1. **Global search** via CMA-ES on a coarse multi-resolution level to
///    obtain a basin-of-attraction estimate that is robust to local MI maxima.
/// 2. **Local refinement** (optional) via the existing [`GlobalMiRegistration`]
///    RSGD pipeline, warm-started from the CMA-ES solution.
pub struct CmaMiRegistration;

impl CmaMiRegistration {
    /// Execute CMA-ES + optional RSGD cascade rigid registration.
    ///
    /// # Arguments
    ///
    /// * `fixed` — Reference image (any modality).
    /// * `moving` — Image to be aligned (may differ in modality from `fixed`).
    /// * `initial_rotation` — Starting Euler angles `[alpha, beta, gamma]` in
    ///   radians (ZYX convention matching [`RigidTransform`]).  Pass `[0,0,0]`
    ///   when no prior orientation knowledge is available.
    /// * `initial_translation` — Starting translation `[tz, ty, tx]` in mm
    ///   (RITK `[z, y, x]` order).  Pass `None` to let the pipeline compute a
    ///   center-of-mass estimate automatically (requires `use_com_init = true`
    ///   in `config`).
    /// * `config` — Pipeline configuration; see [`CmaMiConfig`].
    ///
    /// # Returns
    ///
    /// `(transform, result)` where `transform` is the final [`RigidTransform`]
    /// and `result` contains diagnostics.
    ///
    /// # Algorithm
    ///
    /// ```text
    /// Phase 0 — Initial translation
    ///   If initial_translation is None and use_com_init:
    ///     t_init = translation_from_centers_of_mass(fixed, moving)
    ///   Else:
    ///     t_init = initial_translation.unwrap_or([0; 3])
    ///
    /// Phase 1 — CMA-ES global search
    ///   Build 1-level pyramid at coarse_shrink / coarse_sigma_mm
    ///   Normalise parameters to [−1, 1]⁶
    ///   Minimise obj(θ) = −MI_Mattes(fixed_coarse, moving_coarse ∘ T(θ))
    ///   via (μ/μ_w, λ)-CMA-ES
    ///
    /// Phase 2 — Optional RSGD refinement
    ///   If rsgd_refine is Some(cfg):
    ///     Warm-start GlobalMiRegistration with CMA-ES best transform
    /// ```
    pub fn register_rigid<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_rotation: [f64; 3],
        initial_translation: Option<[f64; 3]>,
        config: &CmaMiConfig,
    ) -> (RigidTransform<B, 3>, CmaMiResult) {
        // ── Phase 0: Initial translation ─────────────────────────────────────
        let t_init: [f64; 3] = match initial_translation {
            Some(t) => t,
            None => {
                if config.use_com_init {
                    super::center_of_mass::translation_from_centers_of_mass(fixed, moving)
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
            initial_translation.is_none() && config.use_com_init,
        );

        // ── Phase 1: Coarse pyramid ───────────────────────────────────────────
        let shrink_factors = vec![vec![config.coarse_shrink; 3]];
        let smoothing_sigmas = vec![vec![config.coarse_sigma_mm; 3]];

        let fixed_pyramid = MultiResolutionPyramid::new(fixed, &shrink_factors, &smoothing_sigmas);
        let moving_pyramid =
            MultiResolutionPyramid::new(moving, &shrink_factors, &smoothing_sigmas);

        // Clone level images so they can be moved into the closure.
        let fixed_c = fixed_pyramid.get_level(0).clone();
        let moving_c = moving_pyramid.get_level(0).clone();

        tracing::info!(
            "CmaMiRegistration: coarse level — fixed {:?}, moving {:?}",
            fixed_c.shape(),
            moving_c.shape(),
        );

        // ── Intensity range & metric ──────────────────────────────────────────
        let (min_f, max_f) = estimate_intensity_range(&fixed_c);
        let (min_m, max_m) = estimate_intensity_range(&moving_c);
        let min_int = min_f.min(min_m);
        let max_int = max_f.max(max_m);

        let metric = MutualInformation::<B>::new_mattes(config.num_mi_bins, min_int, max_int)
            .with_sampling(config.sampling_percentage);

        // ── Normalised initial point ──────────────────────────────────────────
        // x0 = [alpha_n, beta_n, gamma_n, tz_n, ty_n, tx_n]
        // Each component is in [-1, 1] when the true parameter is within range.
        let rot_scale = config.rotation_range_rad;
        let trans_scale = config.translation_range_mm;

        let x0 = [
            initial_rotation[0] / rot_scale,
            initial_rotation[1] / rot_scale,
            initial_rotation[2] / rot_scale,
            t_init[0] / trans_scale,
            t_init[1] / trans_scale,
            t_init[2] / trans_scale,
        ];

        // ── CMA-ES objective closure ──────────────────────────────────────────
        // Captures fixed_c, moving_c, metric by move; re-creates RigidTransform
        // from de-normalised parameters on every evaluation.
        let device = fixed.data().device();
        let device_clone = device.clone();
        let center_arr = [0.0f32; 3];

        let obj = move |params: &[f64]| -> f64 {
            let alpha = (params[0] * rot_scale) as f32;
            let beta = (params[1] * rot_scale) as f32;
            let gamma = (params[2] * rot_scale) as f32;
            let tz = (params[3] * trans_scale) as f32;
            let ty = (params[4] * trans_scale) as f32;
            let tx = (params[5] * trans_scale) as f32;

            let rotation =
                Tensor::<B, 1>::from_data(TensorData::from([alpha, beta, gamma]), &device_clone);
            let translation =
                Tensor::<B, 1>::from_data(TensorData::from([tz, ty, tx]), &device_clone);
            let center = Tensor::<B, 1>::from_data(TensorData::from(center_arr), &device_clone);

            let transform = RigidTransform::<B, 3>::new(translation, rotation, center);

            let loss = metric.forward(&fixed_c, &moving_c, &transform);
            let loss_data = loss.into_data();
            loss_data.as_slice::<f32>().unwrap()[0] as f64
        };

        // ── Run CMA-ES ────────────────────────────────────────────────────────
        tracing::info!(
            "CmaMiRegistration: starting CMA-ES \
             (max_gen={}, sigma0={:.3}, lambda={})",
            config.cma_config.max_generations,
            config.cma_config.sigma0,
            config.cma_config.lambda,
        );

        let cma_result = CmaEsOptimizer::new(config.cma_config.clone()).run(obj, &x0);

        tracing::info!(
            "CmaMiRegistration: CMA-ES finished — \
             generations={}, best_f={:.6e}, stop={:?}, sigma={:.3e}",
            cma_result.generations,
            cma_result.best_f,
            cma_result.stop_reason,
            cma_result.final_sigma,
        );

        // ── Reconstruct best CMA-ES transform ────────────────────────────────
        let best = &cma_result.best_x;
        let b_alpha = (best[0] * rot_scale) as f32;
        let b_beta = (best[1] * rot_scale) as f32;
        let b_gamma = (best[2] * rot_scale) as f32;
        let b_tz = (best[3] * trans_scale) as f32;
        let b_ty = (best[4] * trans_scale) as f32;
        let b_tx = (best[5] * trans_scale) as f32;

        let b_rotation =
            Tensor::<B, 1>::from_data(TensorData::from([b_alpha, b_beta, b_gamma]), &device);
        let b_translation =
            Tensor::<B, 1>::from_data(TensorData::from([b_tz, b_ty, b_tx]), &device);
        let b_center = Tensor::<B, 1>::zeros([3], &device);
        let cma_transform = RigidTransform::<B, 3>::new(b_translation, b_rotation, b_center);

        // ── Optional RSGD refinement ──────────────────────────────────────────
        let (final_transform, rsgd_iterations, rsgd_loss_history) = if let Some(ref rsgd_cfg) =
            config.rsgd_refine
        {
            tracing::info!(
                "CmaMiRegistration: starting RSGD refinement \
                     ({} levels, {} bins)",
                rsgd_cfg.num_levels,
                rsgd_cfg.num_mi_bins,
            );

            let (rsgd_transform, rsgd_result) =
                GlobalMiRegistration::register_rigid_full(fixed, moving, cma_transform, rsgd_cfg);

            let iters: usize = rsgd_result.iterations_per_level.iter().sum();
            tracing::info!(
                "CmaMiRegistration: RSGD complete — \
                     {} total iterations, final MI = {:.6e}",
                iters,
                rsgd_result.final_mi,
            );

            (rsgd_transform, iters, rsgd_result.loss_history)
        } else {
            (cma_transform, 0_usize, Vec::new())
        };

        // ── Assemble result ───────────────────────────────────────────────────
        let matrix = rigid_matrix_to_homogeneous(&final_transform);
        // CMA-ES minimises −MI, so negate to recover the MI value.
        let final_mi = -cma_result.best_f;

        let result = CmaMiResult {
            matrix,
            final_mi,
            cma_generations: cma_result.generations,
            cma_stop_reason: cma_result.stop_reason,
            cma_final_sigma: cma_result.final_sigma,
            rsgd_iterations,
            rsgd_loss_history,
        };

        (final_transform, result)
    }
}
