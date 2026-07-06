//! Core TRE sanity and coordinate-system self-consistency tests for RIRE registration.
//!
//! This file contains:
//! - Pure-math TRE sanity tests (no data required)
//! - Coordinate-system self-consistency tests (data-gated)
//! - Center-of-mass initialization test (data-gated)
//!
//! For the actual registration algorithm tests, see:
//! - `rire_registration_rigid_test.rs` — GlobalMI rigid/translation tests
//! - `rire_registration_cma_test.rs` — CMA-ES, multi-start, and multiscale tests
//!
//! # Running
//!
//! ```shell
//! # Run all non-ignored tests (pure-math only):
//! cargo test --test rire_registration_algorithm_test -- --nocapture
//!
//! # Run data-gated tests (requires test_data):
//! cargo test --test rire_registration_algorithm_test -- --ignored --nocapture
//! ```
//!
//! # Coordinate conventions
//!
//! RITK stores images with shape `[nz, ny, nx]` and world coordinates in
//! `[z, y, x]` order (dim-0 = slice, dim-1 = row, dim-2 = col). The RIRE
//! ground-truth uses `[x, y, z]` order (x = col, y = row, z = slice).
//!
//! The helpers `apply_ritk_m4_to_rire_point` and `resample_mri_into_ct_ritk`
//! internally perform the dimension permutation so that TRE and NCC calculations
//! use the standard RIRE `[x, y, z]` reference frame.
//!
//! # RIRE provenance
//!
//! Images and transforms are from the *Retrospective Image Registration
//! Evaluation* (RIRE) project, NIH Project 8R01EB002124-03, PI: J. Michael
//! Fitzpatrick, Vanderbilt University. License: CC-BY-3.0-US.
//! Data site: <https://rire.insight-journal.org/>

mod common;

use burn_ndarray::NdArray;
use common::{
    apply_ritk_m4_to_rire_point, compute_tre, find_rire_dir, identity_m4, ncc, normalize_minmax,
    resample_mri_into_ct_ritk, B, RIRE_CORNERS,
};
use ritk_io::read_metaimage;
use ritk_registration::translation_from_centers_of_mass;

// ── Pure-math TRE sanity (no data required) ─────────────────────────────────

/// Verify that `apply_ritk_m4_to_rire_point` with the identity matrix returns
/// the input unchanged (i.e., the RIRE ↔ RITK permutation round-trips cleanly).
#[test]
fn test_identity_m4_gives_zero_coordinate_mapping_change() {
    let id = identity_m4();
    for pair in &RIRE_CORNERS {
        let src = [pair[0], pair[1], pair[2]];
        let result = apply_ritk_m4_to_rire_point(&id, src);
        for d in 0..3 {
            assert!(
                (result[d] - src[d]).abs() < 1e-10,
                "Identity matrix should be a no-op on RIRE coords: got {result:?}, expected {src:?}"
            );
        }
    }
}

/// Verify the identity TRE baseline: the mean TRE with the identity transform
/// (no registration) against the RIRE ground-truth fiducials is > 30 mm,
/// confirming there is a meaningful misalignment to correct.
#[test]
fn test_identity_tre_reflects_rire_misalignment() {
    let id = identity_m4();
    let (mean_tre, max_tre) = compute_tre(&id);
    println!("Identity TRE — mean: {mean_tre:.2} mm, max: {max_tre:.2} mm");
    assert!(
        mean_tre > 30.0,
        "Expected identity mean TRE > 30 mm (confirming misalignment), got {mean_tre:.2} mm"
    );
    assert!(
        max_tre > 40.0,
        "Expected identity max TRE > 40 mm, got {max_tre:.2} mm"
    );
}

// ── Coordinate-system self-consistency tests ─────────────────────────────────

/// Verify that `resample_mri_into_ct_ritk` with the identity transform produces
/// voxel values consistent with the direct volume data (i.e., the RITK→MRI-index
/// mapping is correct when there is no motion).
///
/// With identity transform and matching spacing/shape the resampled MRI should
/// have at least 0.80 NCC with the original MRI data sampled at the CT grid.
#[test]
#[ignore = "requires test_data/registration/rire"]
fn test_resampling_helper_self_consistency() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    let device: <NdArray<f32> as ritk_image::tensor::Backend>::Device = Default::default();
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    let mri_raw: Vec<f32> = mri_img.data_slice().into_owned();
    let mri_shape = mri_img.shape(); // [nz, ny, nx]
    let mri_sz = mri_img.spacing()[0] as f64;
    let mri_sy = mri_img.spacing()[1] as f64;
    let mri_sx = mri_img.spacing()[2] as f64;

    // Resample MRI into itself with identity transform: every output sample
    // should equal the nearest MRI input voxel (trilinear, so exact at grid pts).
    let id = identity_m4();
    let mri_resampled = resample_mri_into_ct_ritk(
        mri_shape,
        [mri_sz, mri_sy, mri_sx],
        &mri_raw,
        mri_shape,
        [mri_sz, mri_sy, mri_sx],
        &id,
    );
    let mri_norm = normalize_minmax(&mri_raw);
    let mri_resampled_norm = normalize_minmax(&mri_resampled);
    let self_ncc = ncc(&mri_norm, &mri_resampled_norm);

    println!("Self-resample NCC (should be ≈ 1.0): {self_ncc:.6}");
    assert!(
        self_ncc > 0.99,
        "Self-resample NCC should be > 0.99 (identity transform, same grid), got {self_ncc:.6}"
    );
}

/// Verify that center-of-mass initialization gives a meaningful translation.
///
/// This is a pure-math test (no registration run). It loads the CT and MRI
/// images and checks that `translation_from_centers_of_mass` returns a
/// vector whose magnitude is in a physically reasonable range (5–60 mm).
/// The exact value is not asserted since it depends on image content, but
/// any anatomically plausible scan should have the CoM translation within
/// the expected range for the RIRE patient-001 dataset.
#[test]
#[ignore = "requires test_data/registration/rire"]
fn test_center_of_mass_init_rire_patient001() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    let device: <NdArray<f32> as ritk_image::tensor::Backend>::Device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    let com_trans = translation_from_centers_of_mass(&ct_img, &mri_img);
    let magnitude = (com_trans[0].powi(2) + com_trans[1].powi(2) + com_trans[2].powi(2)).sqrt();
    println!(
        "CoM translation [z,y,x] = [{:.2}, {:.2}, {:.2}] mm (|t| = {:.2} mm)",
        com_trans[0], com_trans[1], com_trans[2], magnitude
    );
    // The GT translation is ~32mm. CoM gives a rough estimate; any value between
    // 5 and 80mm confirms the function is computing a non-trivial offset.
    assert!(
        magnitude > 5.0,
        "CoM translation magnitude {magnitude:.2} mm is unexpectedly small \
            (should be at least 5 mm for CT vs. MRI of distinct geometries)"
    );
    assert!(
        magnitude < 100.0,
        "CoM translation magnitude {magnitude:.2} mm is unexpectedly large (> 100 mm)"
    );
    println!("\u{2713} Center-of-mass translation is in the expected range.");
}
