//! RIRE CT/MRI T1 Registration Validation
//!
//! This example walks through the complete CT↔MRI registration validation
//! workflow for the *Retrospective Image Registration Evaluation* (RIRE)
//! Patient-001 dataset. It is self-contained — all math helpers are inlined —
//! and uses only the CPU NdArray backend (no autodiff, no GPU required).
//!
//! # What RIRE is
//!
//! RIRE (https://rire.insight-journal.org/) is a classic benchmark for
//! rigid inter-modality image registration. Fiducial markers implanted in
//! cadaver skulls provide millimetre-accurate ground-truth transforms between
//! CT, MRI-T1, MRI-T2, and PET modalities. The transforms are provided as
//! ITK Euler3DTransform parameter files. We use Patient-001 CT → MRI T1 here.
//!
//! # Running
//!
//! From the repository root:
//!
//! ```shell
//! cargo run --example rire_mri_ct_registration
//! ```
//!
//! The example expects RIRE test data under `test_data/registration/rire/`.
//!
//! # Provenance
//!
//! Images and transforms provided by the RIRE project, NIH grant
//! 8R01EB002124-03, PI J. Michael Fitzpatrick, Vanderbilt University.
//! License: Creative Commons Attribution 3.0 United States.

mod constants;
mod math;
mod pipeline;
mod viz;

use std::path::PathBuf;

use constants::{B, GT_ROT, GT_TRANS};
use math::{
    downsample_stride, euler3d_from_matrix, mat3_det, mat3_mul, mat3_transpose, ncc,
    normalize_minmax, resample_mri_into_ct_space,
};
use pipeline::{perturbation_and_recovery, print_inverse_and_roundtrip, run_cma_es};
use viz::{ncc_bar, save_comparison_png};

use ritk_io::read_metaimage;

/// Shared state computed during the early pipeline steps and passed to later
/// functions to avoid re-computation and excessive parameter lists.
pub struct RireData {
    pub ct_path: PathBuf,
    pub mri_path: PathBuf,
    pub device: burn_ndarray::NdArrayDevice,
    pub ct_data: Vec<f32>,
    pub mri_data: Vec<f32>,
    pub ct_shape: [usize; 3],
    pub mri_shape: [usize; 3],
    pub ct_ds_data: Vec<f32>,
    pub ct_ds_shape: [usize; 3],
    pub ct_ds_spacing_xyz: [f64; 3],
    pub mri_spacing_xyz: [f64; 3],
    pub aligned_mri: Vec<f32>,
    pub perturbed_mri: Vec<f32>,
    pub identity_mri: Vec<f32>,
    pub ct_norm: Vec<f32>,
    pub ncc_aligned: f64,
    pub ncc_perturbed: f64,
    pub mid_z: usize,
    pub ny_vis: usize,
    pub nx_vis: usize,
    pub ct_slice: Vec<f32>,
    pub pre_slice: Vec<f32>,
    pub ncc_slice_pre: f64,
    pub ncc_slice_post: f64,
    pub output_dir: PathBuf,
}

fn main() -> anyhow::Result<()> {
    // ── 1. Header ─────────────────────────────────────────────────────────────
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║ RIRE CT / MRI-T1 Registration Validation (Patient-001) ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
    println!("The Retrospective Image Registration Evaluation (RIRE) benchmark");
    println!("provides fiducial-marker ground-truth transforms between CT, MRI,");
    println!("and PET modalities. This example validates the Patient-001 CT→T1");
    println!("transform by measuring Pearson NCC before and after alignment, then");
    println!("demonstrates that a known voxel-shift perturbation can be exactly");
    println!("inverted to recover the original alignment quality.");
    println!();

    // ── 2. Locate data files ──────────────────────────────────────────────────
    println!("── Locating RIRE data ─────────────────────────────────────────────");
    let search_paths = [
        "test_data/registration/rire",
        "../test_data/registration/rire",
        "../../test_data/registration/rire",
    ];
    let rire_dir = search_paths
        .iter()
        .map(std::path::Path::new)
        .find(|p| p.exists())
        .map(|p| p.to_path_buf());
    let rire_dir = if let Some(d) = rire_dir {
        println!(" Found RIRE data at: {}", d.display());
        d
    } else {
        anyhow::bail!(
            "RIRE data directory not found.\n\
             Searched:\n  {}\n\
             Please place training_001_ct.mha and training_001_mr_T1.mha\n\
             under test_data/registration/rire/ relative to the workspace root.",
            search_paths.join("\n  ")
        );
    };
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    for p in [&ct_path, &mri_path] {
        if !p.exists() {
            anyhow::bail!("Required file not found: {}", p.display());
        }
    }
    println!();

    // ── 3. Load images and report metadata ───────────────────────────────────
    println!("── Loading images ─────────────────────────────────────────────────");
    let device = Default::default();
    print!(" Loading CT ... ");
    let ct_img = read_metaimage::<B, _>(&ct_path, &device)?;
    println!("done");
    print!(" Loading MRI T1 ... ");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device)?;
    println!("done");
    println!();

    // Report metadata for both images.
    for (label, img) in [("CT", &ct_img), ("MRI T1", &mri_img)] {
        let sh = img.shape(); // [nz, ny, nx]
        let sz = img.spacing()[0]; // z / slice
        let sy = img.spacing()[1]; // y / row
        let sx = img.spacing()[2]; // x / col
        let vox = img.data_vec();
        let vmin = vox.iter().cloned().fold(f32::INFINITY, f32::min);
        let vmax = vox.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let phys_x = sh[2] as f64 * sx;
        let phys_y = sh[1] as f64 * sy;
        let phys_z = sh[0] as f64 * sz;
        println!(" [{label}]");
        println!(
            "  Shape          : [{} × {} × {}] (nz × ny × nx)",
            sh[0], sh[1], sh[2]
        );
        println!(
            "  Spacing        : z={:.4} mm  y={:.4} mm  x={:.4} mm",
            sz, sy, sx
        );
        println!(
            "  Extent         : {:.1} × {:.1} × {:.1} mm (z × y × x)",
            phys_z, phys_y, phys_x
        );
        println!("  Intensity range : [{:.1}, {:.1}]", vmin, vmax);
        println!();
    }

    // ── 4. Ground-truth transform ─────────────────────────────────────────────
    println!("── Ground-truth CT → MRI T1 transform ────────────────────────────");
    println!(" Rotation matrix R (row-major 3×3):");
    for row in 0..3 {
        println!(
            "  [ {:+.9} {:+.9} {:+.9} ]",
            GT_ROT[row * 3],
            GT_ROT[row * 3 + 1],
            GT_ROT[row * 3 + 2]
        );
    }
    println!(
        " Translation t (mm): [{:+.6}, {:+.6}, {:+.6}]",
        GT_TRANS[0], GT_TRANS[1], GT_TRANS[2]
    );

    // Decompose into ITK Euler3D angles (convention: R = Rz·Rx·Ry).
    let (ax, ay, az) = euler3d_from_matrix(&GT_ROT);
    println!();
    println!(" Euler3D angles (ITK convention R = Rz·Rx·Ry):");
    println!("  aX = {:+.6} rad ({:+.3}°)", ax, ax.to_degrees());
    println!("  aY = {:+.6} rad ({:+.3}°)", ay, ay.to_degrees());
    println!("  aZ = {:+.6} rad ({:+.3}°)", az, az.to_degrees());

    // Verify orthogonality: R·R^T should equal I.
    let rrt = mat3_mul(&GT_ROT, &mat3_transpose(&GT_ROT));
    let ortho_err = (0..9)
        .map(|i| {
            let expected = if i % 4 == 0 { 1.0 } else { 0.0 };
            (rrt[i] - expected).abs()
        })
        .fold(0.0_f64, f64::max);
    let det = mat3_det(&GT_ROT);
    println!();
    println!(" Orthogonality check:");
    println!("  ‖R·R^T − I‖_∞ = {:.2e} (tol 1e-9)", ortho_err);
    println!("  det(R)        = {:.9} (should be +1.0)", det);
    println!();

    // ── 5. Inverse transform ──────────────────────────────────────────────────
    print_inverse_and_roundtrip();

    // ── 6. Resample MRI into CT space ─────────────────────────────────────────
    println!("── Resampling MRI into CT space (stride-4 downsample for speed) ───");
    let ct_shape = ct_img.shape();
    let mri_shape = mri_img.shape();
    let ct_data = ct_img.data_vec();
    let mri_data = mri_img.data_vec();

    // Spacing in (x/col, y/row, z/slice) order — note spacing()[0]=z, [2]=x.
    let ct_sx = ct_img.spacing()[2] as f64;
    let ct_sy = ct_img.spacing()[1] as f64;
    let ct_sz = ct_img.spacing()[0] as f64;
    let mri_sx = mri_img.spacing()[2] as f64;
    let mri_sy = mri_img.spacing()[1] as f64;
    let mri_sz = mri_img.spacing()[0] as f64;

    // Downsample CT with stride 4 → roughly [8 × 128 × 128] at ~2.6 mm/vox.
    let stride = 4_usize;
    let (ct_ds_data, ct_ds_shape) = downsample_stride(&ct_data, ct_shape, stride);
    let ct_ds_spacing_xyz = [
        ct_sx * stride as f64,
        ct_sy * stride as f64,
        ct_sz * stride as f64,
    ];
    let mri_spacing_xyz = [mri_sx, mri_sy, mri_sz];

    println!(
        " CT downsampled: {:?} @ [{:.3}, {:.3}, {:.3}] mm/vox (x,y,z)",
        ct_ds_shape, ct_ds_spacing_xyz[0], ct_ds_spacing_xyz[1], ct_ds_spacing_xyz[2]
    );
    println!(
        " Resampling {} MRI voxels via GT transform ...",
        ct_ds_shape[0] * ct_ds_shape[1] * ct_ds_shape[2]
    );

    // GT-aligned resampling.
    let aligned_mri = resample_mri_into_ct_space(
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &GT_ROT,
        &GT_TRANS,
    );
    println!(" GT-aligned resampling complete.");

    // Perturbed resampling (+50 mm in x) — guaranteed to misalign the brains.
    let t_perturbed = [GT_TRANS[0] + 50.0, GT_TRANS[1], GT_TRANS[2]];
    println!(" Resampling with +50 mm x-perturbation ...");
    let perturbed_mri = resample_mri_into_ct_space(
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &GT_ROT,
        &t_perturbed,
    );
    println!(" Perturbed resampling complete.");

    // Identity (no transform) resampling — the "pre-registration" state.
    let identity_rot = [1.0f64, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let identity_trans = [0.0f64; 3];
    println!(" Resampling with identity transform (pre-registration baseline) ...");
    let identity_mri = resample_mri_into_ct_space(
        ct_ds_shape,
        ct_ds_spacing_xyz,
        &mri_data,
        mri_shape,
        mri_spacing_xyz,
        &identity_rot,
        &identity_trans,
    );
    println!(" Identity resampling complete.");
    println!();

    // ── 7. Side-by-side visual comparison (PNG) ───────────────────────────────
    println!("── Side-by-side visual comparison (PNG output) ────────────────────");
    let [nz_vis, ny_vis, nx_vis] = ct_ds_shape;

    // Pick the middle axial slice — typically the most informative for brain.
    let mid_z = nz_vis / 2;
    let s = mid_z * ny_vis * nx_vis;
    let e = s + ny_vis * nx_vis;
    let ct_slice = ct_ds_data[s..e].to_vec();
    let pre_slice = identity_mri[s..e].to_vec();
    let post_slice = &aligned_mri[s..e];

    // NCC at this specific slice (for the terminal annotation).
    let ncc_slice_pre = ncc(&normalize_minmax(&ct_slice), &normalize_minmax(&pre_slice));
    let ncc_slice_post = ncc(&normalize_minmax(&ct_slice), &normalize_minmax(post_slice));

    let output_dir = std::path::Path::new("output");
    std::fs::create_dir_all(output_dir)?;
    let png_path = output_dir.join("rire_registration_comparison.png");
    save_comparison_png(
        &ct_slice,
        &pre_slice,
        post_slice,
        [ny_vis, nx_vis],
        &png_path,
    )?;
    println!(
        " Saved: {} ({}x{} px per panel, 4 panels)",
        png_path.display(),
        nx_vis,
        ny_vis
    );
    println!();
    println!(
        " Axial slice: z-index {} of {} → physical z ≈ {:.0} mm",
        mid_z,
        nz_vis,
        mid_z as f64 * ct_ds_spacing_xyz[2]
    );
    println!();
    println!(" Panel legend (left → right):");
    println!();
    println!(" ═══════════════════════════════════════════════════════════════════════════");
    println!(" [GREY header] Panel 1: CT (fixed image, soft-tissue window)");
    println!(" [RED  header] Panel 2: Pre-reg overlay (red=CT, green=MRI identity)");
    println!(" [YELL header] Panel 3: Transform Δ (|GT-aligned − identity| voxelwise)");
    println!(" [GRN  header] Panel 4: Post-reg overlay (red=CT, green=MRI GT-aligned)");
    println!(" ═══════════════════════════════════════════════════════════════════════════");
    println!();
    println!(" Overlay colour key:");
    println!("  Yellow/white areas = good anatomical overlap (CT ≈ MRI after norm)");
    println!("  Red-only areas     = CT signal with no matching MRI tissue");
    println!("  Green-only areas   = MRI signal with no matching CT structure");
    println!();
    println!(" NCC at mid-slice z={}:", mid_z);
    println!("  Pre (identity):    {:.4}", ncc_slice_pre);
    println!("  Post (GT-aligned): {:.4}", ncc_slice_post);
    println!(
        "  Improvement:       {:+.4}",
        ncc_slice_post - ncc_slice_pre
    );
    println!();

    // ── 8. Alignment quality (volume-wide) ───────────────────────────────────
    println!("── Alignment quality (Pearson NCC on minmax-normalised volumes) ───");
    let ct_norm = normalize_minmax(&ct_ds_data);
    let aligned_norm = normalize_minmax(&aligned_mri);
    let perturbed_norm = normalize_minmax(&perturbed_mri);
    let ncc_aligned = ncc(&ct_norm, &aligned_norm);
    let ncc_perturbed = ncc(&ct_norm, &perturbed_norm);
    let improvement = ncc_aligned - ncc_perturbed;

    println!(" NCC scale: -1 (anti-correlated) ... 0 (unrelated) ... +1 (identical)");
    println!();
    ncc_bar("CT vs GT-aligned MRI ", ncc_aligned);
    ncc_bar("CT vs +50mm MRI (bad) ", ncc_perturbed);
    println!();
    println!(" GT NCC       : {:.6}", ncc_aligned);
    println!(" Perturbed NCC : {:.6}", ncc_perturbed);
    println!(
        " Improvement   : {:+.6} (GT better by this margin)",
        improvement
    );
    if ncc_aligned > ncc_perturbed {
        println!(" ✓ GT transform improves alignment over the perturbed baseline.");
    } else {
        println!(" ✗ WARNING: GT transform did not improve NCC — check data paths.");
    }
    println!();

    // ── 9. Perturbation-and-recovery workflow ─────────────────────────────────
    let data = RireData {
        ct_path,
        mri_path,
        device,
        ct_data,
        mri_data,
        ct_shape,
        mri_shape,
        ct_ds_data,
        ct_ds_shape,
        ct_ds_spacing_xyz,
        mri_spacing_xyz,
        aligned_mri,
        perturbed_mri,
        identity_mri,
        ct_norm,
        ncc_aligned,
        ncc_perturbed,
        mid_z,
        ny_vis,
        nx_vis,
        ct_slice,
        pre_slice,
        ncc_slice_pre,
        ncc_slice_post,
        output_dir: output_dir.to_path_buf(),
    };
    let perturb = perturbation_and_recovery(&data);

    // ── 10. Summary ────────────────────────────────────────────────────────────
    println!("--- Summary ---");
    println!();
    println!(" Metric                                   NCC value");
    println!(" ─────────────────────────────────────── ──────────");
    println!(
        " CT vs GT-aligned MRI                    {:+.6}",
        data.ncc_aligned
    );
    println!(
        " CT vs +50 mm perturbed MRI              {:+.6}",
        data.ncc_perturbed
    );
    println!(
        " NCC improvement from GT alignment       {:+.6}",
        improvement
    );
    println!(
        " CT vs +5-vox shifted (aligned MRI)      {:+.6}",
        perturb.ncc_shift_perturbed
    );
    println!(
        " CT vs −5-vox recovered                  {:+.6}",
        perturb.ncc_shift_recovered
    );
    println!();
    println!(" What these numbers demonstrate:");
    println!("  1. The RIRE GT transform produces positive NCC (~0.64), confirming");
    println!("     that CT and MRI T1 are modality-correlated when aligned.");
    println!("  2. A 50 mm mis-registration degrades NCC by ~0.30, well beyond any");
    println!("     noise floor — registration error is reliably detectable with NCC.");
    println!("  3. Composing a known forward shift with its exact inverse returns NCC");
    println!("     to within 0.05 of the original, validating T ∘ T^{{-1}} ≈ id up to");
    println!("     the ~4% boundary zero-padding introduced by the round-trip shift.");
    println!();

    // ── 11. CMA-ES + Center-of-Mass Rigid Registration ───────────────────────
    let mut data = data;
    run_cma_es(&mut data)?;

    println!("Done.");
    Ok(())
}
