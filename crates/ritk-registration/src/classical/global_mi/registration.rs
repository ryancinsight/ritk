//! GlobalMiRegistration struct and multi-resolution optimization loop.

use super::config::{GlobalMiConfig, GlobalMiTransformType};
use super::result::GlobalMiResult;
use super::transforms::{
    affine_matrix_to_homogeneous, compute_image_center, estimate_intensity_range,
    rigid_matrix_to_homogeneous, translation_matrix_to_homogeneous,
};
use crate::metric::{Metric, MutualInformation};
use crate::optimizer::{ConvergenceReason, Optimizer, RegularStepGradientDescent};
use burn::module::AutodiffModule;
use burn::optim::GradientsParams;
use burn::tensor::backend::AutodiffBackend;
use ritk_core::filter::pyramid::MultiResolutionPyramid;
use ritk_core::image::Image;
use ritk_core::transform::{
    AffineTransform, Resampleable, RigidTransform, Transform, TranslationTransform,
};

/// Multi-resolution Mattes MI + RSGD global registration pipeline.
///
/// Follows ITK's `ImageRegistrationMethod` architecture: multi-resolution
/// pyramid (coarse → fine) with per-level Mattes MI metric and RSGD optimizer.
pub struct GlobalMiRegistration;

impl GlobalMiRegistration {
    // ─── Typed Entry Points ───────────────────────────────────────────────

    /// Execute multi-resolution Mattes MI + RSGD rigid registration
    /// with proper 4×4 matrix extraction from the final transform.
    pub fn register_rigid_full<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_transform: RigidTransform<B, 3>,
        config: &GlobalMiConfig,
    ) -> (RigidTransform<B, 3>, GlobalMiResult<3>) {
        config.validate().expect("GlobalMiConfig validation failed");
        let (final_transform, mut result) =
            Self::execute_multires(fixed, moving, initial_transform, config);
        result.matrix = rigid_matrix_to_homogeneous(&final_transform);
        (final_transform, result)
    }

    /// Execute multi-resolution Mattes MI + RSGD affine registration
    /// with proper 4×4 matrix extraction from the final transform.
    pub fn register_affine_full<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_transform: AffineTransform<B, 3>,
        config: &GlobalMiConfig,
    ) -> (AffineTransform<B, 3>, GlobalMiResult<3>) {
        config.validate().expect("GlobalMiConfig validation failed");
        let (final_transform, mut result) =
            Self::execute_multires(fixed, moving, initial_transform, config);
        result.matrix = affine_matrix_to_homogeneous(&final_transform);
        (final_transform, result)
    }

    /// Execute multi-resolution Mattes MI + RSGD translation registration
    /// with proper 4×4 matrix extraction from the final transform.
    pub fn register_translation_full<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_transform: TranslationTransform<B, 3>,
        config: &GlobalMiConfig,
    ) -> (TranslationTransform<B, 3>, GlobalMiResult<3>) {
        config.validate().expect("GlobalMiConfig validation failed");
        let (final_transform, mut result) =
            Self::execute_multires(fixed, moving, initial_transform, config);
        result.matrix = translation_matrix_to_homogeneous(&final_transform);
        (final_transform, result)
    }

    // Kept for backwards compatibility with callers using the non-full variants.
    /// Execute rigid registration (delegates to `register_rigid_full`).
    pub fn register_rigid<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_transform: RigidTransform<B, 3>,
        config: &GlobalMiConfig,
    ) -> (RigidTransform<B, 3>, GlobalMiResult<3>) {
        Self::register_rigid_full(fixed, moving, initial_transform, config)
    }

    /// Execute affine registration (delegates to `register_affine_full`).
    pub fn register_affine<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_transform: AffineTransform<B, 3>,
        config: &GlobalMiConfig,
    ) -> (AffineTransform<B, 3>, GlobalMiResult<3>) {
        Self::register_affine_full(fixed, moving, initial_transform, config)
    }

    /// Execute translation registration (delegates to `register_translation_full`).
    pub fn register_translation<B: AutodiffBackend>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        initial_transform: TranslationTransform<B, 3>,
        config: &GlobalMiConfig,
    ) -> (TranslationTransform<B, 3>, GlobalMiResult<3>) {
        Self::register_translation_full(fixed, moving, initial_transform, config)
    }

    // ─── Matrix Extraction Delegates ─────────────────────────────────────

    /// Compute the 4×4 homogeneous matrix from a rigid transform result.
    pub fn rigid_result_matrix<B: AutodiffBackend>(transform: &RigidTransform<B, 3>) -> [f64; 16] {
        rigid_matrix_to_homogeneous(transform)
    }

    /// Compute the 4×4 homogeneous matrix from an affine transform result.
    pub fn affine_result_matrix<B: AutodiffBackend>(
        transform: &AffineTransform<B, 3>,
    ) -> [f64; 16] {
        affine_matrix_to_homogeneous(transform)
    }

    /// Compute the 4×4 homogeneous matrix from a translation transform result.
    pub fn translation_result_matrix<B: AutodiffBackend>(
        transform: &TranslationTransform<B, 3>,
    ) -> [f64; 16] {
        translation_matrix_to_homogeneous(transform)
    }

    /// Compute the physical center of an image.
    pub fn image_center<B: burn::tensor::backend::Backend, const D: usize>(
        image: &Image<B, D>,
    ) -> [f64; 3] {
        compute_image_center(image)
    }

    // ─── Generic Multi-Resolution Loop ───────────────────────────────────

    /// Core multi-resolution loop for any transform type T.
    ///
    /// Implements the ITK ImageRegistrationMethod workflow:
    /// 1. Build multi-resolution pyramids for fixed and moving images.
    /// 2. For each level (coarse → fine):
    ///    a. Get downsampled fixed/moving images
    ///    b. Resample transform to current level
    ///    c. Estimate intensity range for MI binning
    ///    d. Create Mattes MI metric with sparse sampling
    ///    e. Create RSGD optimizer with level-specific config
    ///    f. Run RSGD loop: forward → set_loss → backward → step
    /// 3. Build result from final transform parameters.
    fn execute_multires<B, T>(
        fixed: &Image<B, 3>,
        moving: &Image<B, 3>,
        mut transform: T,
        config: &GlobalMiConfig,
    ) -> (T, GlobalMiResult<3>)
    where
        B: AutodiffBackend,
        T: Transform<B, 3> + AutodiffModule<B> + Resampleable<B, 3> + Clone + 'static,
    {
        let shrink_factors: Vec<Vec<usize>> =
            config.shrink_factors.iter().map(|&f| vec![f; 3]).collect();
        let smoothing_sigmas: Vec<Vec<f64>> = config
            .smoothing_sigmas
            .iter()
            .map(|&s| vec![s; 3])
            .collect();

        let fixed_pyramid = MultiResolutionPyramid::new(fixed, &shrink_factors, &smoothing_sigmas);
        let moving_pyramid =
            MultiResolutionPyramid::new(moving, &shrink_factors, &smoothing_sigmas);

        let num_levels = config.num_levels;
        let mut convergence_history: Vec<ConvergenceReason> = Vec::with_capacity(num_levels);
        let mut iterations_per_level: Vec<usize> = Vec::with_capacity(num_levels);
        let mut final_loss_history: Vec<f64> = Vec::new();
        let mut final_loss_val: f64 = f64::INFINITY;

        tracing::info!(
            "GlobalMiRegistration: starting {}-level {} registration",
            num_levels,
            match config.transform_type {
                GlobalMiTransformType::Translation => "translation",
                GlobalMiTransformType::Rigid => "rigid",
                GlobalMiTransformType::Affine => "affine",
            }
        );

        for level in 0..num_levels {
            let fixed_level = fixed_pyramid.get_level(level);
            let moving_level = moving_pyramid.get_level(level);

            tracing::info!(
                "Level {}/{}: fixed shape {:?}, moving shape {:?}",
                level + 1,
                num_levels,
                fixed_level.shape(),
                moving_level.shape(),
            );

            transform = transform.resample(
                fixed_level.shape(),
                *fixed_level.origin(),
                *fixed_level.spacing(),
                *fixed_level.direction(),
            );

            let (min_fixed, max_fixed) = estimate_intensity_range(fixed_level);
            let (min_moving, max_moving) = estimate_intensity_range(moving_level);
            let min_intensity = min_fixed.min(min_moving);
            let max_intensity = max_fixed.max(max_moving);

            tracing::debug!(
                "Level {}: intensity range [{:.2}, {:.2}]",
                level + 1,
                min_intensity,
                max_intensity,
            );

            let metric = MutualInformation::<B>::new_mattes(
                config.num_mi_bins,
                min_intensity,
                max_intensity,
            )
            .with_sampling(config.sampling_percentage);

            let rsgd_config = config.rsgd_configs[level].clone();
            let rsgd_initial_step = rsgd_config.initial_step_length;
            let rsgd_max_iters = rsgd_config.maximum_iterations;
            let mut optimizer: RegularStepGradientDescent<T, B> =
                RegularStepGradientDescent::new(rsgd_config);

            tracing::info!(
                "Level {}: RSGD initial_step={:.4}, max_iters={}",
                level + 1,
                rsgd_initial_step,
                rsgd_max_iters,
            );

            let mut level_loss_history: Vec<f64> = Vec::new();
            let mut level_iterations: usize = 0;

            loop {
                if optimizer.converged() {
                    break;
                }

                let loss = metric.forward(fixed_level, moving_level, &transform);
                let loss_data = loss.to_data();
                let loss_val = loss_data.as_slice::<f32>().unwrap()[0] as f64;
                level_loss_history.push(loss_val);

                optimizer.set_loss(loss_val);

                let grads = loss.backward();
                let grads_params = GradientsParams::from_grads(grads, &transform);
                transform = optimizer.step(transform, grads_params);

                level_iterations += 1;
            }

            let convergence = optimizer
                .convergence_reason()
                .unwrap_or(ConvergenceReason::MaximumIterations);
            convergence_history.push(convergence);
            iterations_per_level.push(level_iterations);

            tracing::info!(
                "Level {}/{} converged: {:?}, iterations: {}, final loss: {:.6e}",
                level + 1,
                num_levels,
                convergence,
                level_iterations,
                level_loss_history.last().copied().unwrap_or(f64::NAN),
            );

            if level == num_levels - 1 {
                final_loss_val = level_loss_history.last().copied().unwrap_or(f64::NAN);
                final_loss_history = level_loss_history;
            }
        }

        let final_mi = -final_loss_val;

        // extract_homogeneous_matrix is a placeholder; typed entry points override matrix.
        let matrix = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0,
        ];

        let all_converged = convergence_history
            .iter()
            .all(|reason| *reason != ConvergenceReason::MaximumIterations);

        tracing::info!(
            "GlobalMiRegistration: complete, final MI = {:.6e}, converged = {}",
            final_mi,
            all_converged,
        );

        let result = GlobalMiResult {
            matrix,
            final_mi,
            convergence_history,
            iterations_per_level,
            loss_history: final_loss_history,
            converged: all_converged,
        };

        (transform, result)
    }
}
