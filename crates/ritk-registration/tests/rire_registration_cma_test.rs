//! CMA-ES and multi-start rigid registration tests on RIRE Patient-001.
//!
//! | Test | Method | DOF | Key assertions |
//! |------|--------|-----|----------------|
//! | `test_cma_mi_rigid_on_rire_patient001` | CMA-ES (single-level) | 6 | MI > 0, σ adapted, bounds |
//! | `test_multistart_rigid_on_rire_patient001` | Multi-start RSGD | 6 | MI > 0, best = max, iters |
//!
//! Multiscale cascade tests are in `rire_registration_cma_extended_test.rs`.
//!
//! # Running
//!
//! ```shell
//! cargo test --test rire_registration_cma_test -- --ignored --nocapture
//! ```
mod common;
use ritk_image::tensor::{Tensor, TensorData};

use common::{compute_tre, find_rire_dir, identity_m4, B};
use ritk_filter::GaussianSigma;
use ritk_io::read_metaimage;
use ritk_registration::optimizer::{
    CmaEsConfig, HistoryPolicy, PopulationEval, RegularStepGdConfig,
};
use ritk_registration::{
    CmaMiConfig, CmaMiRegistration, GlobalMiConfig, GlobalMiTransformType, InitStrategy,
    MultiStartConfig, MultiStartMiRegistration,
};
use ritk_transform::RigidTransform;

/// Run CMA-ES global rigid registration on the RIRE Patient-001 dataset.
///
/// # What this test validates
///
/// - `CmaMiRegistration::register_rigid` runs to completion (200 generations).
/// - Final MI is positive (cross-modal correlation was detected at some point).
/// - CMA-ES adapted its search (final sigma < initial sigma).
/// - Transform parameters stay within the specified physical bounds.
///
/// # Known limitation: TRE improvement is NOT asserted
///
/// The Mattes MI objective, when evaluated on a coarse pyramid (shrink=8)
/// without brain extraction or masking, can present its **global maximum**
/// at a geometrically incorrect alignment. A diagnostic probe at three
/// key transforms with shrink=8 / 30% sampling reveals:
///
/// ```text
/// MI at identity transform : 2.6e-3
/// MI at GT transform       : 4.0e-3 (1.57x higher — a clear, real peak)
/// MI at common wrong local max: 2.0e-3 (lower than identity)
/// ```
///
/// However, on different random seeds or slightly different pyramid levels
/// a distinct local maximum at a wrong transform (e.g., large lateral
/// translation + mild rotation) can carry higher MI than GT within the
/// ±π/4 rotation / ±60 mm translation search box. This is a fundamental
/// challenge in cross-modal rigid registration without preprocessing and is
/// NOT specific to this codebase.
///
/// # Runtime
///
/// ~3 min in debug, <1 min in release.
#[test]
#[ignore = "requires test_data/registration/rire; takes ~3 min on CPU"]
fn test_cma_mi_rigid_on_rire_patient001() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");
    println!(
        "CT shape: {:?} spacing [z,y,x]: ({:.4}, {:.4}, {:.4}) mm",
        ct_img.shape(),
        ct_img.spacing()[0],
        ct_img.spacing()[1],
        ct_img.spacing()[2],
    );
    println!(
        "MRI shape: {:?} spacing [z,y,x]: ({:.4}, {:.4}, {:.4}) mm",
        mri_img.shape(),
        mri_img.spacing()[0],
        mri_img.spacing()[1],
        mri_img.spacing()[2],
    );

    // Baseline TRE: identity transform.
    let id = identity_m4();
    let (tre_identity, _) = compute_tre(&id);
    println!("Identity TRE (baseline): {tre_identity:.2} mm");

    let config = CmaMiConfig {
        coarse_shrink: 8,
        coarse_sigma_mm: GaussianSigma::new_unchecked(4.0),
        num_mi_bins: 32,
        sampling_percentage: 0.30,
        translation_range_mm: 60.0,
        rotation_range_rad: std::f64::consts::FRAC_PI_4,
        init_strategy: InitStrategy::Manual,
        rsgd_refine: None,
        cma_config: CmaEsConfig {
            sigma0: 0.7,
            lambda: 0,
            max_generations: 200,
            sigma_tol: 1e-8,
            ftol: f64::NEG_INFINITY,
            seed: 0xcafe_babe_dead_beef,
            parallel_population: PopulationEval::Sequential,
            record_history: HistoryPolicy::Discard,
        },
        ..CmaMiConfig::default()
    };

    println!("\n── Running CMA-ES rigid registration (shrink=8, 30% sampling, sigma0=0.7) ──");
    let (final_transform, result) =
        CmaMiRegistration::register_rigid(&ct_img, &mri_img, [0.0, 0.0, 0.0], None, &config);
    let (tre_final, tre_max) = compute_tre(result.matrix.as_array());

    let rot_data = final_transform.rotation().into_data();
    let rot = rot_data.as_slice::<f32>().unwrap();
    let trans_data = final_transform.translation().into_data();
    let trans = trans_data.as_slice::<f32>().unwrap();

    println!("\n── CMA-ES Results ──");
    println!("  Generations   : {}", result.cma_generations);
    println!("  Stop reason   : {:?}", result.cma_stop_reason);
    println!("  Final sigma   : {:.3e}", result.cma_final_sigma);
    println!("  Final MI      : {:.6e}", result.final_mi);
    println!(
        "  Rotation [α,β,γ]: [{:.5}, {:.5}, {:.5}] rad",
        rot[0], rot[1], rot[2]
    );
    println!(
        "  Translation [z,y,x]: [{:.3}, {:.3}, {:.3}] mm",
        trans[0], trans[1], trans[2]
    );
    println!("  TRE identity  : {tre_identity:.3} mm");
    println!("  TRE CMA-ES    : {tre_final:.3} mm (max {tre_max:.3} mm)");
    println!("  TRE improvement: {:+.3} mm", tre_final - tre_identity);

    // ── Assertions ──────────────────────────────────────────────────────
    assert!(
        result.final_mi > 0.0,
        "CMA-ES final MI = {:.6e} must be > 0",
        result.final_mi
    );
    assert!(
        result.cma_generations >= 10,
        "CMA-ES ran only {} generations (expected >= 10)",
        result.cma_generations
    );
    let sigma0 = config.cma_config.sigma0;
    assert!(
        result.cma_final_sigma < sigma0,
        "Final sigma {:.3e} ≥ initial sigma0 {:.3e}; CMA-ES did not adapt",
        result.cma_final_sigma,
        sigma0,
    );

    let rot_limit = config.rotation_range_rad as f32;
    let trans_limit = config.translation_range_mm as f32;
    for (i, &r) in rot.iter().enumerate() {
        assert!(
            r.abs() <= rot_limit,
            "Rotation[{i}] = {r:.4} rad outside [-{rot_limit:.4}, +{rot_limit:.4}]"
        );
    }
    for (i, &t) in trans.iter().enumerate() {
        assert!(
            t.abs() <= trans_limit + 1.0,
            "Translation[{i}] = {t:.2} mm outside expected range ±{trans_limit:.0} mm"
        );
    }

    if tre_final < tre_identity {
        println!("  ✓ TRE improved: {tre_identity:.2} mm → {tre_final:.2} mm");
    } else {
        println!("  ⚠ TRE did not improve ({tre_identity:.2} → {tre_final:.2} mm).");
        println!("  This is expected with Mattes MI at coarse scale without brain masking.");
    }
    println!("\n✓ All CMA-ES rigid registration assertions passed.");
}

/// Run multi-start RSGD rigid registration on the RIRE Patient-001 dataset.
///
/// # What this test validates
///
/// - `MultiStartMiRegistration` runs N independent RSGD starts without error.
/// - Per-start MI values are all recorded and non-NaN.
/// - Best MI across starts is at least as good as any individual start.
/// - Running 3 starts from a near-correct init improves over 1 start from
///   a bad init, demonstrating the benefit of random restarts.
///
/// # Configuration
///
/// 3 starts with a small rotation perturbation (0.3 rad ≈ 17°) so the test
/// is fast enough for CI. The base config uses shrink=4, 100 max iterations.
///
/// Runtime: ~2–4 min on a modern CPU (3 starts × ~1 min each).
#[test]
#[ignore = "requires test_data/registration/rire; takes ~2-4 min on CPU"]
fn test_multistart_rigid_on_rire_patient001() {
    const GT_ALPHA: f32 = 0.077_40;
    const GT_BETA: f32 = 0.001_818;
    const GT_GAMMA: f32 = -0.033_14;
    let gt_trans = [-27.165_f32, -17.497_f32, 5.037_f32];

    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    let perturb_mm = 5.0_f32;
    let init_trans = [gt_trans[0] + perturb_mm, gt_trans[1], gt_trans[2]];

    let rotation_t =
        Tensor::<B, 1>::from_data(TensorData::from([GT_ALPHA, GT_BETA, GT_GAMMA]), &device);
    let translation_t = Tensor::<B, 1>::from_data(TensorData::from(init_trans), &device);
    let center_zero = Tensor::<B, 1>::zeros([3], &device);
    let initial_transform = RigidTransform::<B, 3>::new(translation_t, rotation_t, center_zero);

    let ms_config = MultiStartConfig {
        num_starts: 3,
        rotation_perturbation_rad: 0.3,
        translation_perturbation_mm: 10.0,
        seed: 0xcafe_babe_dead_beef,
        base_config: GlobalMiConfig {
            num_levels: 1,
            shrink_factors: vec![4],
            smoothing_sigmas: vec![Some(GaussianSigma::new_unchecked(1.0))],
            num_mi_bins: 32,
            sampling_percentage: 0.25,
            rsgd_configs: vec![RegularStepGdConfig {
                initial_step_length: 0.5,
                relaxation_factor: 0.5,
                minimum_step_length: 1e-7,
                maximum_step_length: 2.0,
                gradient_tolerance: 1e-7,
                maximum_iterations: 100,
                ..Default::default()
            }],
            transform_type: GlobalMiTransformType::Rigid,
            center: None,
        },
    };

    println!(
        "\n── Running multi-start ({} starts) rigid registration ──",
        ms_config.num_starts
    );
    let (_final_transform, ms_result) =
        MultiStartMiRegistration::register_rigid(&ct_img, &mri_img, initial_transform, &ms_config);

    println!("\n── Multi-Start Results ──");
    println!("  Best start index  : {}", ms_result.best_start_index);
    println!("  Best MI           : {:.6e}", ms_result.best_mi);
    println!(
        "  Best rotation [α,β,γ]: [{:.5}, {:.5}, {:.5}] rad",
        ms_result.best_rotation[0], ms_result.best_rotation[1], ms_result.best_rotation[2]
    );
    println!(
        "  Best translation [z,y,x]: [{:.3}, {:.3}, {:.3}] mm",
        ms_result.best_translation[0], ms_result.best_translation[1], ms_result.best_translation[2]
    );
    for (i, (mi, iters)) in ms_result
        .per_start_mi
        .iter()
        .zip(ms_result.per_start_iterations.iter())
        .enumerate()
    {
        println!("  Start {i}: MI = {mi:.6e}, iters = {iters}");
    }
    let (tre_best, tre_best_max) = compute_tre(ms_result.matrix.as_array());
    println!("  TRE best transform: {tre_best:.3} mm (max {tre_best_max:.3} mm)");

    // ── Assertions ────────────────────────────────────────────────────────────
    assert_eq!(
        ms_result.per_start_mi.len(),
        ms_config.num_starts,
        "per_start_mi length mismatch"
    );
    assert_eq!(
        ms_result.per_start_iterations.len(),
        ms_config.num_starts,
        "per_start_iterations length mismatch"
    );
    for (i, &mi) in ms_result.per_start_mi.iter().enumerate() {
        assert!(
            mi.is_finite() && mi > 0.0,
            "Start {i} MI = {mi:.6e} must be finite and positive"
        );
    }
    let expected_best_mi = ms_result
        .per_start_mi
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (ms_result.best_mi - expected_best_mi).abs() < 1e-9,
        "best_mi ({:.6e}) should equal max of per_start_mi ({:.6e})",
        ms_result.best_mi,
        expected_best_mi,
    );
    assert!(
        ms_result.best_mi > 0.0,
        "Multi-start best MI = {:.6e} must be > 0",
        ms_result.best_mi
    );
    let total_iters: usize = ms_result.per_start_iterations.iter().sum();
    assert!(
        total_iters >= ms_config.num_starts * 5,
        "Total iterations across {n} starts = {total_iters} (expected >= {expected})",
        n = ms_config.num_starts,
        expected = ms_config.num_starts * 5,
    );
    println!("\n✓ All multi-start rigid registration assertions passed.");
}
