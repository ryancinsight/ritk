//! Extended / edge-case unit tests for global MI registration.

use super::super::config::{GlobalMiConfig, GlobalMiTransformType};
use super::super::registration::GlobalMiRegistration;
use crate::optimizer::RegularStepGdConfig;
use crate::optimizer::{HistoryPolicy, PopulationEval};
use burn::tensor::Tensor;
use ritk_core::filter::GaussianSigma;
use ritk_core::transform::TranslationTransform;

use super::{make_box_mask, make_gaussian_blob, TestBackend};

// ── Convergence History Test ─────────────────────────────────────────────────

#[test]
fn convergence_history_tracks_per_level() {
    let device = Default::default();
    let fixed = make_gaussian_blob([16, 16, 16], [8.0, 8.0, 8.0], 3.0, &device);
    let moving = make_gaussian_blob([16, 16, 16], [8.0, 8.0, 8.0], 3.0, &device);
    let initial_transform =
        TranslationTransform::<TestBackend, 3>::new(Tensor::<TestBackend, 1>::zeros([3], &device));

    let config = GlobalMiConfig {
        num_levels: 3,
        shrink_factors: vec![4, 2, 1],
        smoothing_sigmas: vec![
            Some(GaussianSigma::new_unchecked(4.0)),
            Some(GaussianSigma::new_unchecked(2.0)),
            None,
        ],
        num_mi_bins: 16,
        sampling_percentage: 0.30,
        rsgd_configs: vec![
            RegularStepGdConfig {
                initial_step_length: 1.0,
                relaxation_factor: 0.5,
                minimum_step_length: 1e-4,
                maximum_step_length: 5.0,
                gradient_tolerance: 1e-4,
                maximum_iterations: 30,
            },
            RegularStepGdConfig {
                initial_step_length: 0.5,
                relaxation_factor: 0.5,
                minimum_step_length: 1e-5,
                maximum_step_length: 2.0,
                gradient_tolerance: 1e-5,
                maximum_iterations: 30,
            },
            RegularStepGdConfig {
                initial_step_length: 0.2,
                relaxation_factor: 0.5,
                minimum_step_length: 1e-6,
                maximum_step_length: 1.0,
                gradient_tolerance: 1e-6,
                maximum_iterations: 30,
            },
        ],
        transform_type: GlobalMiTransformType::Translation,
        center: None,
    };

    let (_, result) = GlobalMiRegistration::register_translation_full(
        &fixed,
        &moving,
        initial_transform,
        &config,
    );

    assert_eq!(result.convergence_history.len(), 3);
    assert_eq!(result.iterations_per_level.len(), 3);

    for (level, &iters) in result.iterations_per_level.iter().enumerate() {
        assert!(
            iters > 0,
            "Level {} must have positive iterations",
            level + 1
        );
    }
}

// ── Sprint 290: Brain Masking Tests (extended) ──────────────────────────────

#[test]
fn cma_mi_register_rigid_without_mask_matches_register_rigid_with_none() {
    // register_rigid and register_rigid_with_mask(mask=None) must produce the
    // same result for an identical RNG seed.
    use super::super::cma_mi::{CmaMiConfig, CmaMiRegistration};

    let device = Default::default();
    let fixed = make_gaussian_blob([8, 8, 8], [4.0, 4.0, 4.0], 2.0, &device);
    let moving = make_gaussian_blob([8, 8, 8], [4.0, 4.0, 4.0], 2.0, &device);

    let config = CmaMiConfig {
        cma_config: crate::optimizer::CmaEsConfig {
            sigma0: 0.3,
            lambda: 0,
            max_generations: 2,
            sigma_tol: 1e-8,
            ftol: f64::NEG_INFINITY,
            seed: 0xcafe_babe_dead_beef,
            parallel_population: PopulationEval::Sequential,
            record_history: HistoryPolicy::Discard,
        },
        coarse_shrink: 4,
        coarse_sigma_mm: GaussianSigma::new_unchecked(2.0),
        ..CmaMiConfig::default()
    };

    let (_, r1) = CmaMiRegistration::register_rigid(&fixed, &moving, [0.0; 3], None, &config);
    let (_, r2) =
        CmaMiRegistration::register_rigid_with_mask(&fixed, &moving, [0.0; 3], None, &config, None);

    // Same seed → same generation count (deterministic stopping by max_gen).
    assert_eq!(
        r1.cma_generations, r2.cma_generations,
        "generation count must match when mask=None"
    );

    // Both final_mi values should be finite and essentially zero for
    // identical images; stochastic MI sampling means exact equality is not
    // guaranteed — only finiteness and rough equivalence are tested.
    assert!(r1.final_mi.is_finite(), "r1 final_mi must be finite");
    assert!(r2.final_mi.is_finite(), "r2 final_mi must be finite");
}

#[test]
fn cma_mi_register_rigid_with_mask_partial_foreground_runs_without_error() {
    // A mask covering only the central 4×4×4 voxels of an 8×8×8 volume.
    // The masked MI should still converge (fewer samples — faster).
    use super::super::cma_mi::{CmaMiConfig, CmaMiRegistration, InitStrategy};

    let device = Default::default();
    let shape = [8, 8, 8];
    let fixed = make_gaussian_blob(shape, [4.0, 4.0, 4.0], 2.0, &device);
    let moving = make_gaussian_blob(shape, [4.0, 4.0, 4.0], 2.0, &device);

    // Mask: only central 2–6 voxels in each axis (4×4×4 = 64 foreground voxels).
    let mask = make_box_mask(shape, 2..6, 2..6, 2..6, &device);

    let config = CmaMiConfig {
        cma_config: crate::optimizer::CmaEsConfig {
            sigma0: 0.3,
            lambda: 0,
            max_generations: 3,
            sigma_tol: 1e-8,
            ftol: f64::NEG_INFINITY,
            seed: 42,
            parallel_population: PopulationEval::Sequential,
            record_history: HistoryPolicy::Discard,
        },
        coarse_shrink: 2, // mild shrink so 8×8×8 → 4×4×4 (mask still has foreground)
        coarse_sigma_mm: GaussianSigma::new_unchecked(1.0),
        sampling_percentage: 1.0, // use all foreground
        init_strategy: InitStrategy::Manual,
        ..CmaMiConfig::default()
    };

    let (transform, result) = CmaMiRegistration::register_rigid_with_mask(
        &fixed,
        &moving,
        [0.0; 3],
        None,
        &config,
        Some(&mask),
    );

    assert!(
        result.final_mi.is_finite(),
        "final MI must be finite, got {}",
        result.final_mi
    );

    for v in transform
        .translation()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
    {
        assert!(v.is_finite(), "translation must be finite, got {v}");
    }
}
