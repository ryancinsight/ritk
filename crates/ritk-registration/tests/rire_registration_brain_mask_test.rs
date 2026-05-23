//! RIRE brain mask generation and validation on real cross-modal data.
//!
//! Closes the highest-priority open gap: brain masking infrastructure was built
//! (Sprint 290) but never validated on real RIRE data. This is the first test
//! that:
//!
//! 1. Generates a brain mask from the CT volume using threshold + morphology.
//! 2. Runs thin-slab multiscale CMA-ES WITHOUT mask (baseline).
//! 3. Runs the same config WITH mask.
//! 4. Validates that masked registration produces lower TRE than identity and
//!    is not worse than unmasked registration.
//!
//! # Running
//!
//! ```shell
//! cargo test --test rire_registration_brain_mask_test -- --ignored --nocapture
//! ```
//!
//! Runtime: ~10–15 min on CPU (masked and unmasked runs each ~3–7 min).
mod common;

use burn::tensor::backend::Backend;
use burn_ndarray::NdArray;

use common::{compute_tre, find_rire_dir, identity_m4, B};
use ritk_core::filter::{
    BinaryDilateFilter, BinaryErodeFilter, BinaryFillholeFilter, BinaryThresholdImageFilter,
};
use ritk_core::segmentation::ConnectedComponentsFilter;
use ritk_core::Image;
use ritk_io::read_metaimage;
use ritk_registration::{CmaMiConfig, CmaMiRegistration};

/// Generate a brain mask from a CT volume using threshold + morphology.
///
/// Pipeline:
/// 1. Threshold CT intensity to soft-tissue range [0, 100] HU.
/// 2. Erode (radius 2) — break thin connections (skull, meninges, neck muscle).
/// 3. Label connected components (26-connectivity).
/// 4. Keep the largest component (brain).
/// 5. Dilate (radius 2) — restore eroded brain boundary.
/// 6. Fill internal holes.
fn create_ct_brain_mask<B: Backend>(ct: &Image<B, 3>) -> Image<B, 3> {
    let threshold = BinaryThresholdImageFilter::new(0.0, 100.0, 1.0, 0.0);
    let mask = threshold.apply(ct).expect("threshold failed");

    let erode = BinaryErodeFilter::new(2);
    let eroded = erode.apply(&mask).expect("erosion failed");

    let cc = ConnectedComponentsFilter::with_connectivity(26);
    let (label_img, stats) = cc.apply(&eroded);

    let largest = stats
        .iter()
        .max_by_key(|s| s.voxel_count)
        .expect("no connected components in eroded mask");

    let lv = largest.label as f32;
    let largest_only =
        BinaryThresholdImageFilter::new(lv, lv, 1.0, 0.0)
            .apply(&label_img)
            .expect("largest-component threshold failed");

    let dilate = BinaryDilateFilter::new(2);
    let dilated = dilate.apply(&largest_only).expect("dilation failed");

    let fill = BinaryFillholeFilter::new();
    fill.apply(&dilated).expect("hole-fill failed")
}

#[test]
#[ignore = "requires test_data/registration/rire; takes ~10-15 min on CPU"]
fn test_brain_masked_registration_tre_on_rire_patient001() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");

    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
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

    // ── Generate brain mask ───────────────────────────────────────────────────
    println!("\n── Generating brain mask from CT ──");
    let brain_mask = create_ct_brain_mask(&ct_img);
    let total_voxels: usize = brain_mask.shape().iter().product();
    let fg_voxels = brain_mask.data_vec().iter().filter(|&&v| v > 0.5).count();
    println!(
        "  Shape: {:?}, foreground: {}/{} ({:.1}%)",
        brain_mask.shape(),
        fg_voxels,
        total_voxels,
        100.0 * fg_voxels as f64 / total_voxels as f64
    );
    assert!(
        fg_voxels > 0,
        "Brain mask is empty — threshold [0,100] HU may be wrong for this CT"
    );
    assert!(
        fg_voxels as f64 > 0.01 * total_voxels as f64,
        "Brain mask covers only {:.1}% of volume — likely too restrictive",
        100.0 * fg_voxels as f64 / total_voxels as f64
    );

    // ── Registration config (thin-slab multiscale) ────────────────────────────
    let config = CmaMiConfig::brain_rigid_multiscale_thin_slab();
    let id = identity_m4();
    let (tre_identity, _) = compute_tre(&id);
    println!("\nIdentity TRE (baseline): {tre_identity:.2} mm");

    // ── Without mask ─────────────────────────────────────────────────────────
    println!("\n── Registration WITHOUT brain mask ──");
    let t0 = std::time::Instant::now();
    let (_tfm_a, res_a) =
        CmaMiRegistration::register_rigid(&ct_img, &mri_img, [0.0; 3], None, &config);
    let dt_a = t0.elapsed();
    let (tre_a, tre_a_max) = compute_tre(&res_a.matrix);
    println!(
        "  Time: {:.1}s, MI: {:.6e}, TRE: {tre_a:.3} mm (max {tre_a_max:.3})",
        dt_a.as_secs_f64(),
        res_a.final_mi
    );

    // ── With mask ────────────────────────────────────────────────────────────
    println!("\n── Registration WITH brain mask ──");
    let t1 = std::time::Instant::now();
    let (_tfm_b, res_b) = CmaMiRegistration::register_rigid_with_mask(
        &ct_img,
        &mri_img,
        [0.0; 3],
        None,
        &config,
        Some(&brain_mask),
    );
    let dt_b = t1.elapsed();
    let (tre_b, tre_b_max) = compute_tre(&res_b.matrix);
    println!(
        "  Time: {:.1}s, MI: {:.6e}, TRE: {tre_b:.3} mm (max {tre_b_max:.3})",
        dt_b.as_secs_f64(),
        res_b.final_mi
    );

    // ── Comparison ───────────────────────────────────────────────────────────
    println!("\n── TRE Comparison ──");
    println!("  Identity           : {tre_identity:.2} mm");
    println!("  Without mask       : {tre_a:.3} mm (max {tre_a_max:.3})");
    println!("  With mask          : {tre_b:.3} mm (max {tre_b_max:.3})");
    println!("  Δ(masked - id)     : {:.3} mm", tre_b - tre_identity);
    println!("  Δ(masked - unmask) : {:.3} mm", tre_b - tre_a);

    // ── Assertions ───────────────────────────────────────────────────────────
    assert!(
        res_b.final_mi > 0.0,
        "Masked final MI = {:.6e} must be > 0",
        res_b.final_mi
    );

    assert!(
        tre_b < tre_identity,
        "Masked TRE {tre_b:.3} mm did not improve over identity {tre_identity:.3} mm"
    );

    assert!(
        tre_b <= tre_a + 1.0,
        "Masked TRE {tre_b:.3} mm > unmasked TRE {tre_a:.3} mm (tolerance: 1 mm)"
    );

    println!("\n✓ Brain mask validation passed on RIRE Patient-001.");
}
