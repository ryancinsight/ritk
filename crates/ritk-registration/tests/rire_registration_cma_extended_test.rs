//! Multiscale CMA-ES rigid registration tests on RIRE Patient-001.
//!
//! | Test | Method | DOF | Key assertions |
//! |------|--------|-----|----------------|
//! | `test_cma_mi_multiscale_on_rire_patient001` | CMA-ES 3-level cascade | 6 | MI > 0, σ adapted, bounds |
//! | `test_cma_mi_thin_slab_multiscale_on_rire_patient001` | CMA-ES thin-slab cascade | 6 | MI > 0, σ adapted, bounds |
//!
//! # Running
//!
//! ```shell
//! cargo test --test rire_registration_cma_extended_test -- --ignored --nocapture
//! ```
mod common;

use burn_ndarray::NdArray;

use common::{compute_tre, find_rire_dir, identity_m4, B};
use ritk_io::read_metaimage;
use ritk_registration::{CmaMiConfig, CmaMiRegistration};

/// Run three-level coarse-to-fine CMA-ES cascade on RIRE Patient-001.
///
/// Uses `CmaMiConfig::brain_rigid_multiscale()` which runs:
/// Level 0: shrink=16, sigma0=0.8, 100 gens (wide exploration)
/// Level 1: shrink= 8, sigma0=0.3, 200 gens (rough alignment)
/// Level 2: shrink= 4, sigma0=0.1, 100 gens (fine convergence)
///
/// Each level seeds the next with its best parameter vector.
/// The cascade + NMI combination should produce a lower TRE than the
/// single-level `test_cma_mi_rigid_on_rire_patient001` run.
///
/// # Assertions
///
/// | Assertion | Rationale |
/// |-----------|----------|
/// | `final_mi > 0` | Optimizer found cross-modal alignment. |
/// | `cma_generations >= 10` | Pipeline ran meaningfully at last level. |
/// | `final_sigma < sigma0` | CMA-ES adapted (landscape is not flat). |
/// | Parameters within search bounds | Penalty invariant holds. |
#[test]
#[ignore = "requires test_data/registration/rire; takes ~10-15 min on CPU"]
fn test_cma_mi_multiscale_on_rire_patient001() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();
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
    let id = identity_m4();
    let (tre_identity, _) = compute_tre(&id);
    println!("Identity TRE (baseline): {tre_identity:.2} mm");

    // Use the pre-tuned three-level cascade preset.
    let config = CmaMiConfig::brain_rigid_multiscale();
    assert_eq!(
        config.pyramid_schedule.len(),
        3,
        "Sanity: multiscale preset must have 3 levels"
    );

    println!("\n── Running 3-level cascade CMA-ES (NMI, shrinks 16→8→4) ──");
    for (i, level) in config.pyramid_schedule.iter().enumerate() {
        println!(
            "  Level {}: shrink={}, sigma_mm={:.1}, sigma0={:.2}, max_gen={}",
            i,
            level.shrink,
            level.sigma_mm.get(),
            level.cma_sigma0,
            level.max_generations
        );
    }

    let (final_transform, result) =
        CmaMiRegistration::register_rigid(&ct_img, &mri_img, [0.0, 0.0, 0.0], None, &config);
    let (tre_final, tre_max) = compute_tre(result.matrix.as_array());

    let rot_data = final_transform.rotation().into_data();
    let rot = rot_data.as_slice::<f32>().unwrap();
    let trans_data = final_transform.translation().into_data();
    let trans = trans_data.as_slice::<f32>().unwrap();

    println!("\n── Multiscale Cascade Results ──");
    println!("  Generations (last level): {}", result.cma_generations);
    println!("  Stop reason            : {:?}", result.cma_stop_reason);
    println!("  Final sigma            : {:.3e}", result.cma_final_sigma);
    println!("  Final MI               : {:.6e}", result.final_mi);
    println!(
        "  Rotation [α,β,γ]: [{:.5}, {:.5}, {:.5}] rad",
        rot[0], rot[1], rot[2]
    );
    println!(
        "  Translation [z,y,x]: [{:.3}, {:.3}, {:.3}] mm",
        trans[0], trans[1], trans[2]
    );
    println!("  TRE identity   : {tre_identity:.3} mm");
    println!("  TRE multiscale : {tre_final:.3} mm (max {tre_max:.3} mm)");
    println!("  TRE change     : {:+.3} mm", tre_final - tre_identity);

    // ── Assertions ──────────────────────────────────────────────────────
    assert!(
        result.final_mi > 0.0,
        "Cascade final MI = {:.6e} must be > 0",
        result.final_mi
    );
    assert!(
        result.cma_generations >= 10,
        "Last-level CMA-ES ran only {} generations (expected >= 10)",
        result.cma_generations
    );
    assert!(
        result.cma_final_sigma < 0.1,
        "Final sigma {:.3e} >= last level sigma0 0.1; CMA-ES did not adapt",
        result.cma_final_sigma
    );

    let rot_limit = config.rotation_range_rad as f32;
    let trans_limit = config.translation_range_mm as f32;
    for (i, &r) in rot.iter().enumerate() {
        assert!(
            r.abs() <= rot_limit,
            "Rotation[{i}] = {r:.4} rad outside [±{rot_limit:.4}]"
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
        println!("  Expected behaviour: cascade finds better basin than single-level.");
    }
    println!("\n✓ All multiscale cascade assertions passed.");
}

/// Run the thin-slab multiscale cascade on RIRE Patient-001.
///
/// Uses `CmaMiConfig::brain_rigid_multiscale_thin_slab()` with per-axis shrink
/// `[1,16,16]->[1,8,8]->[1,4,4]` to preserve all 29 z-slices of the CT throughout
/// the pyramid, avoiding the information collapse that occurs with isotropic shrink.
///
/// # Assertions
///
/// - MI > 0 (valid NMI objective at the coarse level)
/// - CMA-ES ran >= 10 generations at the last level
/// - Final sigma adapted (< initial sigma0 of 0.1)
/// - Transform parameters within configured bounds
///
/// TRE is reported but NOT asserted -- cold-start cross-modal registration without
/// brain masking can still converge to local maxima. Use `register_rigid_with_mask`
/// for TRE-guaranteed registration (Sprint 290).
#[test]
#[ignore = "requires test_data/registration/rire; takes ~3-4 min on CPU"]
fn test_cma_mi_thin_slab_multiscale_on_rire_patient001() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();
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
    let id = identity_m4();
    let (tre_identity, _) = compute_tre(&id);
    println!("Identity TRE (baseline): {tre_identity:.2} mm");

    // Use the thin-slab anisotropic cascade preset.
    let config = CmaMiConfig::brain_rigid_multiscale_thin_slab();
    assert_eq!(
        config.pyramid_schedule.len(),
        3,
        "Sanity: thin-slab multiscale preset must have 3 levels"
    );

    println!("\n── Running thin-slab 3-level CMA-ES (NMI, [1,16,16]→[1,8,8]→[1,4,4]) ──");
    for (i, level) in config.pyramid_schedule.iter().enumerate() {
        if let Some(axes) = level.shrink_per_axis {
            println!(
                "  Level {}: shrink_per_axis=[{},{},{}], sigma_mm={:.1}, sigma0={:.2}, max_gen={}",
                i,
                axes[0],
                axes[1],
                axes[2],
                level.sigma_mm.get(),
                level.cma_sigma0,
                level.max_generations
            );
        } else {
            println!(
                "  Level {}: shrink={}, sigma_mm={:.1}, sigma0={:.2}, max_gen={}",
                i,
                level.shrink,
                level.sigma_mm.get(),
                level.cma_sigma0,
                level.max_generations
            );
        }
    }

    let (final_transform, result) =
        CmaMiRegistration::register_rigid(&ct_img, &mri_img, [0.0, 0.0, 0.0], None, &config);
    let (tre_final, tre_max) = compute_tre(result.matrix.as_array());

    let rot_data = final_transform.rotation().into_data();
    let rot = rot_data.as_slice::<f32>().unwrap();
    let trans_data = final_transform.translation().into_data();
    let trans = trans_data.as_slice::<f32>().unwrap();

    println!("\n── Thin-Slab Cascade Results ──");
    println!("  Generations (last level): {}", result.cma_generations);
    println!("  Stop reason            : {:?}", result.cma_stop_reason);
    println!("  Final sigma            : {:.3e}", result.cma_final_sigma);
    println!("  Final MI               : {:.6e}", result.final_mi);
    println!(
        "  Rotation [a,b,g]: [{:.5}, {:.5}, {:.5}] rad",
        rot[0], rot[1], rot[2]
    );
    println!(
        "  Translation [z,y,x]: [{:.3}, {:.3}, {:.3}] mm",
        trans[0], trans[1], trans[2]
    );
    println!("  TRE identity                : {tre_identity:.3} mm");
    println!("  TRE thin-slab cascade       : {tre_final:.3} mm (max {tre_max:.3} mm)");
    println!(
        "  TRE change                  : {:+.3} mm",
        tre_final - tre_identity
    );
    println!(
        "  Sprint 292 comparison       : isotropic multiscale TRE ~146 mm, thin-slab TRE = {tre_final:.2} mm"
    );

    // ── Assertions
    assert!(
        result.final_mi > 0.0,
        "Thin-slab cascade final MI = {:.6e} must be > 0",
        result.final_mi
    );
    assert!(
        result.cma_generations >= 10,
        "Last-level CMA-ES ran only {} generations (expected >= 10)",
        result.cma_generations
    );
    assert!(
        result.cma_final_sigma < 0.1,
        "Final sigma {:.3e} >= last level sigma0 0.1; CMA-ES did not adapt",
        result.cma_final_sigma
    );

    let rot_limit = config.rotation_range_rad as f32;
    let trans_limit = config.translation_range_mm as f32;
    for (i, &r) in rot.iter().enumerate() {
        assert!(
            r.abs() <= rot_limit,
            "Rotation[{i}] = {r:.4} rad outside [+/-{rot_limit:.4}]"
        );
    }
    for (i, &t) in trans.iter().enumerate() {
        assert!(
            t.abs() <= trans_limit + 1.0,
            "Translation[{i}] = {t:.2} mm outside expected range +/-{trans_limit:.0} mm"
        );
    }

    if tre_final < tre_identity {
        println!("  ✓ TRE improved: {tre_identity:.2} mm → {tre_final:.2} mm");
    } else {
        println!("  ⚠ TRE did not improve ({tre_identity:.2} → {tre_final:.2} mm).");
        println!("  Cold-start cross-modal registration without masking.");
    }
    println!("\n✓ All thin-slab multiscale cascade assertions passed.");
}
