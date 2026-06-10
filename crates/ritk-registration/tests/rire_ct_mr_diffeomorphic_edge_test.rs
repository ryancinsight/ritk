//! RIRE CT/MR T1 diffeomorphic edge-case integration tests.
//!
//! These tests validate inverse-transform recovery and ground-truth alignment
//! improvements for the CT↔MRI T1 pair. They require the RIRE test data under
//! `test_data/registration/rire/` and can be run with:
//!
//! ```shell
//! cargo test --test rire_ct_mr_diffeomorphic_edge_test -- --ignored
//! ```
//!
//! # RIRE provenance
//!
//! Images and the standard transformation were provided as part of the project
//! *Retrospective Image Registration Evaluation* (RIRE), National Institutes
//! of Health, Project Number 8R01EB002124-03, Principal Investigator
//! J. Michael Fitzpatrick, Vanderbilt University, Nashville TN.
//! Data site: <https://rire.insight-journal.org/>
//! License: Creative Commons Attribution 3.0 United States.

use burn_ndarray::NdArray;

mod common;

use common::{
    downsample_stride, find_rire_dir, ncc, normalize_minmax, resample_mri_into_ct_space, GT_ROT,
    GT_TRANS,
};
use ritk_io::read_metaimage;

type B = NdArray<f32>;

/// # Specification
///
/// Resampling MRI T1 onto the CT grid using the fiducial ground-truth transform
/// must produce better NCC than using a perturbed transform (+50 mm in x).
///
/// The test:
/// 1. Loads CT and MRI T1 volumes.
/// 2. Downsamples CT with stride 4 → effective spacing (2.614, 2.614, 16.0) mm.
/// 3. Resamples the full-resolution MRI into the CT-downsampled grid using
///    the GT transform → `ncc_gt_aligned`.
/// 4. Resamples the full-resolution MRI using GT + [+50 mm, 0, 0] perturbation
///    (guaranteed to misalign the brains) → `ncc_perturbed`.
/// 5. Asserts `ncc_gt_aligned > ncc_perturbed` and `ncc_gt_aligned > -0.5`.
///
/// The 50 mm x-shift is larger than the full MRI FOV displacement introduced
/// by the GT translation, so it reliably degrades alignment relative to GT.
#[test]
#[ignore = "requires test_data/registration/rire/"]
fn test_rire_gt_transform_improves_ct_mri_alignment() {
    let Some(rire_dir) = find_rire_dir() else {
        eprintln!(
            "RIRE data directory not found; \
 skipping test_rire_gt_transform_improves_ct_mri_alignment"
        );
        return;
    };

    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");

    for p in &[&ct_path, &mri_path] {
        if !p.exists() {
            eprintln!(
                "Required file {} not found; \
 skipping test_rire_gt_transform_improves_ct_mri_alignment",
                p.display()
            );
            return;
        }
    }

    let device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("CT .mha must load");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("MRI T1 .mha must load");

    // ── Spacing in (x/col, y/row, z/slice) order ──────────────────────────
    // image.spacing(): [0]=z, [1]=y, [2]=x
    let ct_sz = ct_img.spacing()[0]; // z=4.0
    let ct_sy = ct_img.spacing()[1]; // y=0.653595
    let ct_sx = ct_img.spacing()[2]; // x=0.653595

    let mri_sz = mri_img.spacing()[0]; // z=4.0
    let mri_sy = mri_img.spacing()[1]; // y=1.25
    let mri_sx = mri_img.spacing()[2]; // x=1.25

    let ct_shape = ct_img.shape(); // [29, 512, 512]
    let mri_shape = mri_img.shape(); // [26, 256, 256]
    let ct_data: Vec<f32> = ct_img.data_slice().into_owned();
    let mri_data: Vec<f32> = mri_img.data_slice().into_owned();

    // ── Downsample CT with stride=4 ────────────────────────────────────────
    let stride = 4_usize;
    let (ct_ds_data, ct_ds_shape) = downsample_stride(&ct_data, ct_shape, stride);

    // Effective CT downsampled spacing (x, y, z order):
    let ct_ds_spacing_xyz = [
        ct_sx * stride as f64, // x/col
        ct_sy * stride as f64, // y/row
        ct_sz * stride as f64, // z/slice
    ];

    // Full-res MRI spacing (x, y, z order):
    let mri_spacing_xyz = [mri_sx, mri_sy, mri_sz];

    // ── GT-aligned resampling ─────────────────────────────────────────────
    let aligned_mri = resample_mri_into_ct_space(
        &ct_ds_data,
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &GT_ROT,
        &GT_TRANS,
    );

    // ── Perturbed resampling (+50 mm in x) ────────────────────────────────
    let t_perturbed = [GT_TRANS[0] + 50.0, GT_TRANS[1], GT_TRANS[2]];
    let perturbed_mri = resample_mri_into_ct_space(
        &ct_ds_data,
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &GT_ROT,
        &t_perturbed,
    );

    // ── NCC comparison ────────────────────────────────────────────────────
    let ct_norm = normalize_minmax(&ct_ds_data);
    let aligned_norm = normalize_minmax(&aligned_mri);
    let perturbed_norm = normalize_minmax(&perturbed_mri);

    let ncc_gt_aligned = ncc(&ct_norm, &aligned_norm);
    let ncc_perturbed = ncc(&ct_norm, &perturbed_norm);

    eprintln!(
        "Alignment NCC — GT: {:.6}, Perturbed (+50mm x): {:.6}, Δ: {:.6}",
        ncc_gt_aligned,
        ncc_perturbed,
        ncc_gt_aligned - ncc_perturbed
    );

    assert!(
        ncc_gt_aligned > ncc_perturbed,
        "GT-aligned NCC ({:.6}) must exceed perturbed NCC ({:.6}); \
 the ground-truth transform must improve alignment over a +50 mm \
 x-shift perturbation",
        ncc_gt_aligned,
        ncc_perturbed
    );

    assert!(
        ncc_gt_aligned > -0.5,
        "GT-aligned NCC ({:.6}) must be > -0.5 (not strongly anti-correlated)",
        ncc_gt_aligned
    );
}

/// # Specification
///
/// Applying a known 5-voxel (+13 mm) column shift to the GT-aligned MRI and
/// then its exact inverse (−5 voxels) must recover the original NCC to within
/// 0.05.
///
/// Validates the fundamental invertibility property: `T ∘ T^{-1} = id` (to
/// within boundary effects from zero-padding: ~5/128 ≈ 4% of voxels along x).
///
/// The test:
/// 1. Loads CT and MRI T1.
/// 2. Downsamples CT with stride 4 → shape ≈ [8, 128, 128], effective
///    spacing (2.614, 2.614, 16.0) mm.
/// 3. Resamples full MRI into CT-downsampled grid using GT → `aligned_mri`.
/// 4. Applies a +5 voxel column shift to `aligned_mri` → `perturbed_mri`
///    (source from `ix - 5`, zero-pad left border).
/// 5. Applies the inverse shift (−5 voxels: source from `ix + 5`) to
///    `perturbed_mri` → `recovered_mri`.
/// 6. Normalizes CT, aligned, perturbed, and recovered with `normalize_minmax`.
/// 7. Assertions:
///    - `ncc_perturbed < ncc_aligned - 0.01`
///    - `ncc_recovered > ncc_perturbed`
///    - `|ncc_recovered − ncc_aligned| < 0.05`
///
/// The tolerance 0.05 accounts for edge effects from zero-padding
/// (~5/128 ≈ 4 % of x-extent voxels).
#[test]
#[ignore = "requires test_data/registration/rire/"]
fn test_rire_inverse_transform_recovers_shifted_mri() {
    let Some(rire_dir) = find_rire_dir() else {
        eprintln!(
            "RIRE data directory not found; \
 skipping test_rire_inverse_transform_recovers_shifted_mri"
        );
        return;
    };

    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");

    for p in &[&ct_path, &mri_path] {
        if !p.exists() {
            eprintln!(
                "Required file {} not found; \
 skipping test_rire_inverse_transform_recovers_shifted_mri",
                p.display()
            );
            return;
        }
    }

    let device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("CT .mha must load");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("MRI T1 .mha must load");

    // ── Spacing in (x/col, y/row, z/slice) order ──────────────────────────
    let ct_sz = ct_img.spacing()[0]; // z=4.0
    let ct_sy = ct_img.spacing()[1]; // y=0.653595
    let ct_sx = ct_img.spacing()[2]; // x=0.653595

    let mri_sz = mri_img.spacing()[0]; // z=4.0
    let mri_sy = mri_img.spacing()[1]; // y=1.25
    let mri_sx = mri_img.spacing()[2]; // x=1.25

    let ct_shape = ct_img.shape();
    let mri_shape = mri_img.shape();
    let ct_data: Vec<f32> = ct_img.data_slice().into_owned();
    let mri_data: Vec<f32> = mri_img.data_slice().into_owned();

    // ── Downsample CT with stride=4 ────────────────────────────────────────
    let stride = 4_usize;
    let (ct_ds_data, ct_ds_shape) = downsample_stride(&ct_data, ct_shape, stride);
    let [nz, ny, nx] = ct_ds_shape;

    let ct_ds_spacing_xyz = [
        ct_sx * stride as f64,
        ct_sy * stride as f64,
        ct_sz * stride as f64,
    ];
    let mri_spacing_xyz = [mri_sx, mri_sy, mri_sz];

    // ── Step 3: Resample MRI into CT-downsampled space using GT ───────────
    let aligned_mri = resample_mri_into_ct_space(
        &ct_ds_data,
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &GT_ROT,
        &GT_TRANS,
    );

    // ── Step 4: Apply +5 voxel column shift ───────────────────────────────
    // perturbed[iz, iy, ix] = aligned[iz, iy, ix - 5] if ix >= 5, else 0.0
    let shift: usize = 5;
    let mut perturbed_mri = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if ix >= shift {
                    perturbed_mri[iz * ny * nx + iy * nx + ix] =
                        aligned_mri[iz * ny * nx + iy * nx + (ix - shift)];
                }
                // else: perturbed = 0.0 (zero-padding on the left border)
            }
        }
    }

    // ── Step 5: Apply inverse shift (−5 voxels) ───────────────────────────
    // recovered[iz, iy, ix] = perturbed[iz, iy, ix + 5] if ix + 5 < nx, else 0.0
    let mut recovered_mri = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if ix + shift < nx {
                    recovered_mri[iz * ny * nx + iy * nx + ix] =
                        perturbed_mri[iz * ny * nx + iy * nx + (ix + shift)];
                }
                // else: recovered = 0.0 (zero-padding on the right border)
            }
        }
    }

    // ── Normalize and compute NCC ─────────────────────────────────────────
    let ct_norm = normalize_minmax(&ct_ds_data);
    let aligned_norm = normalize_minmax(&aligned_mri);
    let perturbed_norm = normalize_minmax(&perturbed_mri);
    let recovered_norm = normalize_minmax(&recovered_mri);

    let ncc_aligned = ncc(&ct_norm, &aligned_norm);
    let ncc_perturbed = ncc(&ct_norm, &perturbed_norm);
    let ncc_recovered = ncc(&ct_norm, &recovered_norm);

    // Count zero-padded voxels introduced by the shift round-trip.
    let boundary_zeros = nz * ny * shift; // voxels zeroed on the right after recovery
    let total_voxels = nz * ny * nx;
    let boundary_fraction = boundary_zeros as f64 / total_voxels as f64;

    eprintln!(
        "Shift recovery NCC — aligned: {:.6}, perturbed: {:.6}, recovered: {:.6}",
        ncc_aligned, ncc_perturbed, ncc_recovered
    );
    eprintln!(
        "Boundary zeros from shift round-trip: {} / {} voxels ({:.2}%)",
        boundary_zeros,
        total_voxels,
        boundary_fraction * 100.0
    );
    eprintln!(
        "|ncc_recovered - ncc_aligned| = {:.6}",
        (ncc_recovered - ncc_aligned).abs()
    );

    // Assertion 1: perturbation must degrade alignment.
    assert!(
        ncc_perturbed < ncc_aligned - 0.01,
        "5-voxel shift must degrade NCC by > 0.01: \
 ncc_aligned={:.6}, ncc_perturbed={:.6}",
        ncc_aligned,
        ncc_perturbed
    );

    // Assertion 2: inverse shift must improve over the perturbed state.
    assert!(
        ncc_recovered > ncc_perturbed,
        "Inverse shift must recover NCC above perturbed: \
 ncc_recovered={:.6}, ncc_perturbed={:.6}",
        ncc_recovered,
        ncc_perturbed
    );

    // Assertion 3: recovered NCC must be close to the original aligned NCC
    // (within 0.05, accounting for ~4% boundary zero-padding from the round-trip).
    assert!(
        (ncc_recovered - ncc_aligned).abs() < 0.05,
        "Recovered NCC ({:.6}) must be within 0.05 of aligned NCC ({:.6}); \
 got |Δ| = {:.6}. This validates T ∘ T^{{-1}} ≈ identity to within \
 boundary effects ({:.2}% padded voxels).",
        ncc_recovered,
        ncc_aligned,
        (ncc_recovered - ncc_aligned).abs(),
        boundary_fraction * 100.0
    );
}
