//! GlobalMI translation-only near-ground-truth registration test on RIRE Patient-001.
//!
//! Companion to `rire_registration_rigid_test.rs`. Contains the near-GT
//! translation refinement test that asserts TRE < 5 mm.
//!
//! # Running
//!
//! ```shell
//! cargo test --test rire_registration_rigid_extended_test -- --ignored --nocapture
//! ```
mod common;

use coeus_core::SequentialBackend;
use ritk_image::tensor::{Tensor, TensorData};

use common::{compute_tre, find_rire_dir, identity_m4, B};
use ritk_filter::GaussianSigma;
use ritk_io::read_metaimage;
use ritk_registration::optimizer::RegularStepGdConfig;
use ritk_registration::{GlobalMiConfig, GlobalMiRegistration, GlobalMiTransformType};

/// Run 3-DOF translation-only `GlobalMiRegistration` on the RIRE Patient-001
/// data, starting from **near the ground truth** (GT translation + 3 mm
/// z-perturbation).
///
/// Unlike `test_global_mi_translation_only_on_rire_patient001` which starts from
/// the identity transform, this test begins within the convergence basin by
/// perturbing the ground-truth translation by +3 mm in z. It verifies that
/// **local refinement** from a close initialisation converges to TRE < 5 mm.
///
/// # Configuration
///
/// Same as the cold-start translation test: single level, shrink 4, 200
/// iterations. Initial translation = GT + [+3 mm, 0, 0] in RITK [z,y,x].
///
/// # Assertions
///
/// | Assertion | Rationale |
/// |-----------|------------|
/// | `final_mi > 0` | Cross-modal alignment found at near-GT position. |
/// | `loss_history.len() >= 2` | At least 2 MI evaluations were recorded. |
/// | `final_loss <= initial_loss` | Optimizer moved toward higher MI. |
/// | `TRE < 5.0 mm` | Local refinement from near-GT converges to a good solution. |
#[test]
#[ignore = "requires test_data/registration/rire; takes ~3-5 min on CPU"]
fn test_global_mi_translation_near_gt_rire_patient001() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    let device: <SequentialBackend as ritk_image::tensor::Backend>::Device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");
    println!(
        "CT shape: {:?} spacing (z,y,x): ({:.4}, {:.4}, {:.4}) mm",
        ct_img.shape(),
        ct_img.spacing()[0],
        ct_img.spacing()[1],
        ct_img.spacing()[2],
    );

    // Same config as the cold-start translation test.
    let config = GlobalMiConfig {
        num_levels: 1,
        shrink_factors: vec![4],
        smoothing_sigmas: vec![Some(GaussianSigma::new_unchecked(1.0))],
        num_mi_bins: 32,
        sampling_percentage: 0.30,
        rsgd_configs: vec![RegularStepGdConfig {
            initial_step_length: 5.0,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-8,
            maximum_step_length: 15.0,
            gradient_tolerance: 1e-8,
            maximum_iterations: 200,
            ..Default::default()
        }],
        transform_type: GlobalMiTransformType::Translation,
        center: None,
    };

    // GT translation in RITK [z,y,x] order: [-27.165, -17.497, 5.037].
    // Perturb by +3 mm in z to create a near-GT starting point.
    const GT_TZ: f32 = -27.165;
    const GT_TY: f32 = -17.497;
    const GT_TX: f32 = 5.037;

    let initial_translation =
        Tensor::<B, 1>::from_data(TensorData::from([GT_TZ + 3.0_f32, GT_TY, GT_TX]), &device);
    let initial_t = ritk_transform::TranslationTransform::<B, 3>::new(initial_translation);

    // Sanity-check: initial TRE should be ~3 mm.
    let mut m_init = identity_m4();
    m_init[3] = (GT_TZ + 3.0) as f64; // z translation
    m_init[7] = GT_TY as f64; // y translation
    m_init[11] = GT_TX as f64; // x translation
    let (tre_init, _) = compute_tre(&m_init);
    println!("Initial (near-GT + 3 mm z) TRE: {tre_init:.2} mm (expect ~3 mm)");

    println!("\n── Running 3-DOF translation registration from near-GT (shrink [4], 200 iters) ──");
    let (final_t, result) =
        GlobalMiRegistration::register_translation_full(&ct_img, &mri_img, initial_t, &config);

    let t_data = final_t.translation().to_data();
    let t = t_data.as_slice::<f32>().unwrap();
    println!(
        "  Estimated translation (z,y,x): [{:.2}, {:.2}, {:.2}] mm",
        t[0], t[1], t[2]
    );
    println!(
        "  GT translation (z,y,x): [{:.2}, {:.2}, {:.2}] mm",
        GT_TZ, GT_TY, GT_TX
    );

    let mut m = identity_m4();
    m[3] = t[0] as f64; // z translation
    m[7] = t[1] as f64; // y translation
    m[11] = t[2] as f64; // x translation
    let (tre_after, tre_max_after) = compute_tre(&m);

    let initial_loss = result.loss_history.first().copied().unwrap_or(f64::NAN);
    let final_loss = result.loss_history.last().copied().unwrap_or(f64::NAN);

    println!("\n── Results ─────────────────────────────────────────────────");
    println!("  Final MI      : {:.6}", result.final_mi);
    println!("  Iterations    : {:?}", result.iterations_per_level);
    println!("  Loss first→last: {initial_loss:.6e} → {final_loss:.6e}");
    println!(
        "  TRE near-GT   : {tre_init:.2} mm → after: {tre_after:.2} mm (max: {tre_max_after:.2} mm)"
    );

    // ── Assertions ────────────────────────────────────────────────────────────
    // 1. MI must be positive.
    assert!(
        result.final_mi > 0.0,
        "Expected final_mi > 0 after near-GT translation registration, got {:.6}",
        result.final_mi
    );
    // 2. At least 2 MI evaluations so we can compare first vs last loss.
    assert!(
        result.loss_history.len() >= 2,
        "Need at least 2 loss history entries, got {}",
        result.loss_history.len()
    );
    // 3. Loss must decrease (MI must improve over the optimisation run).
    assert!(
        final_loss <= initial_loss,
        "MI loss did not decrease: initial = {initial_loss:.6e}, final = {final_loss:.6e}"
    );
    // 4. TRE must be < 5 mm after local refinement from near-GT.
    assert!(
        tre_after < 5.0,
        "TRE after near-GT local refinement too large: {tre_after:.2} mm (expected < 5.0 mm)"
    );
    println!("\n✓ All near-GT translation-registration assertions passed.");
}
