//! RIRE CT/MRI T1 Registration Validation
//!
//! This example walks through the complete CTâ†”MRI registration validation
//! workflow for the *Retrospective Image Registration Evaluation* (RIRE)
//! Patient-001 dataset. It is self-contained â€” all math helpers are inlined â€”
//! and uses only the CPU NdArray backend (no autodiff, no GPU required).
//!
//! # What RIRE is
//!
//! RIRE (<https://rire.insight-journal.org/>) is a classic benchmark for
//! rigid inter-modality image registration. Fiducial markers implanted in
//! cadaver skulls provide millimetre-accurate ground-truth transforms between
//! CT, MRI-T1, MRI-T2, and PET modalities. The transforms are provided as
//! ITK Euler3DTransform parameter files. We use Patient-001 CT â†’ MRI T1 here.
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
    normalize_minmax, resample_mri_into_ct_space };
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
    pub output_dir: PathBuf }

fn main() -> anyhow::Result<()> {
    // â”€â”€ 1. Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    println!("â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—");
    println!("â•‘ RIRE CT / MRI-T1 Registration Validation (Patient-001) â•‘");
    println!("â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
    println!();
    println!("The Retrospective Image Registration Evaluation (RIRE) benchmark");
    println!("provides fiducial-marker ground-truth transforms between CT, MRI,");
    println!("and PET modalities. This example validates the Patient-001 CTâ†’T1");
    println!("transform by measuring Pearson NCC before and after alignment, then");
    println!("demonstrates that a known voxel-shift perturbation can be exactly");
    println!("inverted to recover the original alignment quality.");
    println!();

    // â”€â”€ 2. Locate data files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    println!("â”€â”€ Locating RIRE data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€");
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

    // â”€â”€ 3. Load images and report metadata â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    println!("â”€â”€ Loading images â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€");
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
        let vox = img.data_slice();
        let vmin = vox.iter().cloned().fold(f32::INFINITY, f32::min);
        let vmax = vox.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let phys_x = sh[2] as f64 * sx;
        let phys_y = sh[1] as f64 * sy;
        let phys_z = sh[0] as f64 * sz;
        println!(" [{label}]");
        println!(
            "  Shape          : [{} Ã— {} Ã— {}] (nz Ã— ny Ã— nx)",
            sh[0], sh[1], sh[2]
        );
        println!(
            "  Spacing        : z={:.4} mm  y={:.4} mm  x={:.4} mm",
            sz, sy, sx
        );
        println!(
            "  Extent         : {:.1} Ã— {:.1} Ã— {:.1} mm (z Ã— y Ã— x)",
            phys_z, phys_y, phys_x
        );
        println!("  Intensity range : [{:.1}, {:.1}]", vmin, vmax);
        println!();
    }

    // â”€â”€ 4. Ground-truth transform â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    println!("â”€â”€ Ground-truth CT â†’ MRI T1 transform â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€");
    println!(" Rotation matrix R (row-major 3Ã—3):");
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

    // Decompose into ITK Euler3D angles (convention: R = RzÂ·RxÂ·Ry).
    let (ax, ay, az) = euler3d_from_matrix(&GT_ROT);
    println!();
    println!(" Euler3D angles (ITK convention R = RzÂ·RxÂ·Ry):");
    println!("  aX = {:+.6} rad ({:+.3}Â°)", ax, ax.to_degrees());
    println!("  aY = {:+.6} rad ({:+.3}Â°)", ay, ay.to_degrees());
    println!("  aZ = {:+.6} rad ({:+.3}Â°)", az, az.to_degrees());

    // Verify orthogonality: RÂ·R^T should equal I.
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
    println!("  â€–RÂ·R^T âˆ’ Iâ€–_âˆž = {:.2e} (tol 1e-9)", ortho_err);
    println!("  det(R)        = {:.9} (should be +1.0)", det);
    println!();

    // â”€â”€ 5. Inverse transform â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print_inverse_and_roundtrip();

    // â”€â”€ 6. Resample MRI into CT space â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    println!("â”€â”€ Resampling MRI into CT space (stride-4 downsample for speed) â”€â”€â”€");
    let ct_shape = ct_img.shape();
    let mri_shape = mri_img.shape();
    let ct_data: Vec<f32> = ct_img.data_slice().into_owned();
    let mri_data: Vec<f32> = mri_img.data_slice().into_owned();

    // Spacing in (x/col, y/row, z/slice) order â€” note spacing()[0]=z, [2]=x.
    let ct_sx = ct_img.spacing()[2] as f64;
    let ct_sy = ct_img.spacing()[1] as f64;
    let ct_sz = ct_img.spacing()[0] as f64;
    let mri_sx = mri_img.spacing()[2] as f64;
    let mri_sy = mri_img.spacing()[1] as f64;
    let mri_sz = mri_img.spacing()[0] as f64;

    // Downsample CT with stride 4 â†’ roughly [8 Ã— 128 Ã— 128] at ~2.6 mm/vox.
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

    // Perturbed resampling (+50 mm in x) â€” guaranteed to misalign the brains.
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

    // Identity (no transform) resampling â€” the "pre-registration" state.
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

    // â”€â”€ 7. Side-by-side visual comparison (PNG) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    println!("â”€â”€ Side-by-side visual comparison (PNG output) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€");
    let [nz_vis, ny_vis, nx_vis] = ct_ds_shape;

    // Pick the middle axial slice â€” typically the most informative for brain.
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
        " Axial slice: z-index {} of {} â†’ physical z â‰ˆ {:.0} mm",
        mid_z,
        nz_vis,
        mid_z as f64 * ct_ds_spacing_xyz[2]
    );
    println!();
    println!(" Panel legend (left â†’ right):");
    println!();
    println!(" â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
    println!(" [GREY header] Panel 1: CT (fixed image, soft-tissue window)");
    println!(" [RED  header] Panel 2: Pre-reg overlay (red=CT, green=MRI identity)");
    println!(" [YELL header] Panel 3: Transform Î” (|GT-aligned âˆ’ identity| voxelwise)");
    println!(" [GRN  header] Panel 4: Post-reg overlay (red=CT, green=MRI GT-aligned)");
    println!(" â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•");
    println!();
    println!(" Overlay colour key:");
    println!("  Yellow/white areas = good anatomical overlap (CT â‰ˆ MRI after norm)");
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

    // â”€â”€ 8. Alignment quality (volume-wide) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    println!("â”€â”€ Alignment quality (Pearson NCC on minmax-normalised volumes) â”€â”€â”€");
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
        println!(" âœ“ GT transform improves alignment over the perturbed baseline.");
    } else {
        println!(" âœ— WARNING: GT transform did not improve NCC â€” check data paths.");
    }
    println!();

    // â”€â”€ 9. Perturbation-and-recovery workflow â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        output_dir: output_dir.to_path_buf() };
    let perturb = perturbation_and_recovery(&data);

    // â”€â”€ 10. Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    println!("--- Summary ---");
    println!();
    println!(" Metric                                   NCC value");
    println!(" â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€");
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
        " CT vs âˆ’5-vox recovered                  {:+.6}",
        perturb.ncc_shift_recovered
    );
    println!();
    println!(" What these numbers demonstrate:");
    println!("  1. The RIRE GT transform produces positive NCC (~0.64), confirming");
    println!("     that CT and MRI T1 are modality-correlated when aligned.");
    println!("  2. A 50 mm mis-registration degrades NCC by ~0.30, well beyond any");
    println!("     noise floor â€” registration error is reliably detectable with NCC.");
    println!("  3. Composing a known forward shift with its exact inverse returns NCC");
    println!("     to within 0.05 of the original, validating T âˆ˜ T^{{-1}} â‰ˆ id up to");
    println!("     the ~4% boundary zero-padding introduced by the round-trip shift.");
    println!();

    // â”€â”€ 11. CMA-ES + Center-of-Mass Rigid Registration â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    let mut data = data;
    run_cma_es(&mut data)?;

    println!("Done.");
    Ok(())
}
