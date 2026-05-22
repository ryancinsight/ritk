//! High-level pipeline steps for the RIRE CT/MRI T1 registration validation.
//!
//! Each function corresponds to a numbered section of the example workflow,
//! extracting verbose printing and computation logic out of `main()`.

use super::constants::{RegB, GT_EULER_ZYX, GT_ROT, GT_TRANS, GT_TRANS_ZYX};
use super::math::{
    apply_rigid, compute_tre, identity_m4, ncc, normalize_minmax, resample_mri_into_ct_ritk,
    rigid_inverse,
};
use super::viz::{ncc_bar, save_comparison_png};
use super::RireData;

use ritk_io::read_metaimage;
use ritk_registration::{CmaMiConfig, CmaMiRegistration};

// ── Step 5: Inverse transform ───────────────────────────────────────────────

/// Print the inverse transform, verify orthogonality, and show roundtrip
/// sanity check at a probe point.
pub fn print_inverse_and_roundtrip() {
    println!("── Inverse transform T^{{-1}}(y) = R^T · y − R^T · t ────────────");
    let (r_inv, t_inv) = rigid_inverse(&GT_ROT, &GT_TRANS);
    println!(" R^T (forward → backward rotation):");
    for row in 0..3 {
        println!(
            "  [ {:+.9} {:+.9} {:+.9} ]",
            r_inv[row * 3],
            r_inv[row * 3 + 1],
            r_inv[row * 3 + 2]
        );
    }
    println!(
        " t_inv (mm): [{:+.6}, {:+.6}, {:+.6}]",
        t_inv[0], t_inv[1], t_inv[2]
    );

    // Quick roundtrip sanity-check at a probe point.
    let probe = [166.0_f64, 166.0, 56.0];
    let fwd = apply_rigid(&GT_ROT, &GT_TRANS, &probe);
    let back = apply_rigid(&r_inv, &t_inv, &fwd);
    let roundtrip_err = {
        let dx = back[0] - probe[0];
        let dy = back[1] - probe[1];
        let dz = back[2] - probe[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    };
    println!();
    println!(
        " Roundtrip sanity check at probe ({:.0},{:.0},{:.0}) mm:",
        probe[0], probe[1], probe[2]
    );
    println!(
        "  T(probe)       = ({:.6}, {:.6}, {:.6})",
        fwd[0], fwd[1], fwd[2]
    );
    println!(
        "  T^{{-1}}(T(p)) = ({:.6}, {:.6}, {:.6})",
        back[0], back[1], back[2]
    );
    println!(
        "  Roundtrip error: {:.2e} mm (sub-nanometre)",
        roundtrip_err
    );
    println!();
}

// ── Step 9: Perturbation-and-recovery ───────────────────────────────────────

/// Struct holding the results of the perturbation-and-recovery workflow.
pub struct PerturbResult {
    pub ncc_shift_perturbed: f64,
    pub ncc_shift_recovered: f64,
}

/// Apply a +5 voxel column shift and its inverse, measuring NCC at each
/// stage to validate that `T ∘ T⁻¹ ≈ id` up to boundary effects.
pub fn perturbation_and_recovery(data: &RireData) -> PerturbResult {
    println!("── Perturbation-and-recovery (5-voxel column shift) ───────────────");
    println!(" Starting from the GT-aligned MRI in CT-downsampled space ...");
    let [nz, ny, nx] = data.ct_ds_shape;
    let shift: usize = 5;

    // Step A: Apply +5 voxel column shift.
    // perturbed[iz, iy, ix] = aligned[iz, iy, ix - 5] (ix >= 5), else 0.
    let mut shift_perturbed = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if ix >= shift {
                    shift_perturbed[iz * ny * nx + iy * nx + ix] =
                        data.aligned_mri[iz * ny * nx + iy * nx + (ix - shift)];
                }
            }
        }
    }
    println!(
        " Applied +{shift} voxel column shift (≈{:.1} mm in x).",
        shift as f64 * data.ct_ds_spacing_xyz[0]
    );

    // Step B: Apply inverse shift (−5 voxels).
    // recovered[iz, iy, ix] = perturbed[iz, iy, ix + 5] (ix + 5 < nx), else 0.
    let mut recovered_mri = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                if ix + shift < nx {
                    recovered_mri[iz * ny * nx + iy * nx + ix] =
                        shift_perturbed[iz * ny * nx + iy * nx + (ix + shift)];
                }
            }
        }
    }
    println!(" Applied inverse −{shift} voxel shift to recover original alignment.");

    // Measure NCC at each stage.
    let shift_perturbed_norm = normalize_minmax(&shift_perturbed);
    let recovered_norm = normalize_minmax(&recovered_mri);
    let ncc_shift_before = data.ncc_aligned;
    let ncc_shift_perturbed = ncc(&data.ct_norm, &shift_perturbed_norm);
    let ncc_shift_recovered = ncc(&data.ct_norm, &recovered_norm);
    let boundary_zeros = nz * ny * shift;
    let boundary_pct = 100.0 * boundary_zeros as f64 / (nz * ny * nx) as f64;

    println!();
    println!(" NCC before shift       : {:.6}", ncc_shift_before);
    println!(
        " NCC after +{shift}-vox shift  : {:.6}",
        ncc_shift_perturbed
    );
    println!(
        " NCC after −{shift}-vox recovery: {:.6}",
        ncc_shift_recovered
    );
    println!(
        " Boundary zero-pad fraction: {:.1}% ({} / {} voxels)",
        boundary_pct,
        boundary_zeros,
        nz * ny * nx
    );
    println!(
        " |recovered − original| = {:.6} (tol < 0.05 for boundary effects)",
        (ncc_shift_recovered - ncc_shift_before).abs()
    );
    println!();
    ncc_bar("Before shift (GT)        ", ncc_shift_before);
    ncc_bar("After +5-vox shift       ", ncc_shift_perturbed);
    ncc_bar("After −5-vox recovery    ", ncc_shift_recovered);
    if ncc_shift_recovered > ncc_shift_perturbed {
        println!("\n ✓ Inverse shift recovers NCC above the perturbed level.");
    }
    if (ncc_shift_recovered - ncc_shift_before).abs() < 0.05 {
        println!(" ✓ Recovered NCC matches original within 0.05 (boundary tolerance).");
    }
    println!();

    PerturbResult {
        ncc_shift_perturbed,
        ncc_shift_recovered,
    }
}

// ── Step 11: CMA-ES registration ────────────────────────────────────────────

/// Run CMA-ES registration, print results, compute NCC alignment, and save
/// a comparison PNG.
pub fn run_cma_es(data: &mut RireData) -> anyhow::Result<()> {
    println!("── CMA-ES + Center-of-Mass Rigid MRI → CT Registration ────────────");
    println!();
    println!(" The new CMA-ES cascade pipeline escapes local-maxima traps that");
    println!(" gradient-based RSGD faces when starting from identity.");
    println!(" Phase 0: Center-of-Mass (CoM) translation initialisation");
    println!(" Phase 1: CMA-ES global search on a coarse pyramid (shrink=8)");
    println!();

    // ── TRE at identity (pre-registration baseline) ───────────────────────
    let id = identity_m4();
    let (tre_id, tre_id_max) = compute_tre(&id);
    println!(
        " Baseline TRE (identity transform) : {:.2} mm (max {:.2} mm)",
        tre_id, tre_id_max
    );
    println!(
        " (RIRE images are ≈{:.0} mm apart in world space before any registration)",
        tre_id
    );
    println!();

    // ── Load images with the autodiff backend ─────────────────────────────
    // NdArrayDevice is identical for NdArray<f32> and Autodiff<NdArray<f32>>,
    // so we can reuse the already-created `device` value.
    print!(" Loading images (autodiff backend) … ");
    let ct_reg = read_metaimage::<RegB, _>(&data.ct_path, &data.device)?;
    let mri_reg = read_metaimage::<RegB, _>(&data.mri_path, &data.device)?;
    println!("done.");
    println!();

    // ── CMA-ES configuration ──────────────────────────────────────────────
    let cma_config = CmaMiConfig {
        coarse_shrink: 8,
        coarse_sigma_mm: 4.0,
        num_mi_bins: 32,
        sampling_percentage: 0.20,
        translation_range_mm: 80.0,
        rotation_range_rad: std::f64::consts::FRAC_PI_4,
        use_com_init: true,
        rsgd_refine: None, // pure CMA-ES to isolate the global search
        ..CmaMiConfig::default()
    };

    println!(" Config:");
    println!("  Pyramid shrink    : {}×", cma_config.coarse_shrink);
    println!("  Max generations   : 400 (default CmaMiConfig)");
    println!("  MI bins           : {}", cma_config.num_mi_bins);
    println!(
        "  Sampling          : {:.0}%",
        cma_config.sampling_percentage * 100.0
    );
    println!(
        "  Translation range : ±{:.0} mm",
        cma_config.translation_range_mm
    );
    println!("  Rotation range    : ±π/4 rad (±45°)");
    println!("  CoM initialisation: {}", cma_config.use_com_init);
    println!();
    println!(" Running CMA-ES — expect 2–5 min on a modern CPU …");
    println!();

    let (final_transform, reg_result) = CmaMiRegistration::register_rigid(
        &ct_reg,
        &mri_reg,
        [0.0, 0.0, 0.0], // identity rotation start
        None,            // translation: computed from center-of-mass
        &cma_config,
    );

    let (tre_cma, tre_cma_max) = compute_tre(&reg_result.matrix);

    let rot_data = final_transform.rotation().into_data();
    let rot = rot_data.as_slice::<f32>().unwrap();
    let trans_data = final_transform.translation().into_data();
    let trans = trans_data.as_slice::<f32>().unwrap();

    println!("── CMA-ES Registration Results ─────────────────────────────────────");
    println!();
    println!(" Convergence:");
    println!("  Generations : {}", reg_result.cma_generations);
    println!("  Stop reason : {:?}", reg_result.cma_stop_reason);
    println!("  Final MI    : {:.6e}", reg_result.final_mi);
    println!("  Final σ     : {:.3e}", reg_result.cma_final_sigma);
    println!();
    println!(" Estimated transform (RITK ZYX convention):");
    println!(
        "  Rotation [α,β,γ]    : [{:+.5}, {:+.5}, {:+.5}] rad",
        rot[0], rot[1], rot[2]
    );
    println!(
        "  Translation [tz,ty,tx]: [{:+.2}, {:+.2}, {:+.2}] mm",
        trans[0], trans[1], trans[2]
    );
    println!();
    println!(" Ground-truth (RIRE Patient-001):");
    println!(
        "  Rotation [α,β,γ]    : [{:+.5}, {:+.5}, {:+.5}] rad",
        GT_EULER_ZYX[0], GT_EULER_ZYX[1], GT_EULER_ZYX[2]
    );
    println!(
        "  Translation [tz,ty,tx]: [{:+.2}, {:+.2}, {:+.2}] mm",
        GT_TRANS_ZYX[0], GT_TRANS_ZYX[1], GT_TRANS_ZYX[2]
    );
    println!();
    println!(" Target Registration Error (TRE) at 8 RIRE fiducial corners:");
    println!(
        "  Before (identity) : {:.2} mm (max {:.2} mm)",
        tre_id, tre_id_max
    );
    println!(
        "  After  (CMA-ES)   : {:.2} mm (max {:.2} mm)",
        tre_cma, tre_cma_max
    );
    let tre_delta = tre_id - tre_cma;
    println!(
        "  Improvement       : {:+.2} mm ({:.1}%)",
        tre_delta,
        100.0 * tre_delta / tre_id
    );
    println!();

    // ── NCC alignment quality with the CMA-ES transform ───────────────────
    // Resample MRI into the (downsampled) CT grid using the RITK 4×4 matrix.
    let ct_ds_spacing_zyx = [
        data.ct_ds_spacing_xyz[2], // sz = z (index [2] of xyz array)
        data.ct_ds_spacing_xyz[1], // sy = y
        data.ct_ds_spacing_xyz[0], // sx = x (index [0] of xyz array)
    ];
    let mri_spacing_zyx = [
        data.mri_spacing_xyz[2], // sz
        data.mri_spacing_xyz[1], // sy
        data.mri_spacing_xyz[0], // sx
    ];

    let registered_mri = resample_mri_into_ct_ritk(
        data.ct_ds_shape,
        ct_ds_spacing_zyx,
        &data.mri_data,
        data.mri_shape,
        mri_spacing_zyx,
        &reg_result.matrix,
    );
    let registered_norm = normalize_minmax(&registered_mri);
    let ncc_registered = ncc(&data.ct_norm, &registered_norm);
    let ncc_identity_vol = ncc(&data.ct_norm, &normalize_minmax(&data.identity_mri));

    println!(" NCC alignment quality (minmax-normalised, downsampled volumes):");
    ncc_bar("CT vs identity MRI (pre-reg) ", ncc_identity_vol);
    ncc_bar("CT vs CMA-ES registered MRI  ", ncc_registered);
    ncc_bar("CT vs GT-aligned MRI         ", data.ncc_aligned);
    println!();
    println!(" Pre-reg NCC : {:.6}", ncc_identity_vol);
    println!(
        " CMA-ES NCC  : {:.6} (Δ = {:+.6} vs pre-reg)",
        ncc_registered,
        ncc_registered - ncc_identity_vol
    );
    println!(" GT NCC      : {:.6}", data.ncc_aligned);
    println!();

    // ── Save CMA-ES comparison PNG ────────────────────────────────────────
    // Panels: CT (grey) | pre-reg overlay | Δ | CMA-ES-registered overlay
    let mid_s = data.mid_z * data.ny_vis * data.nx_vis;
    let reg_slice = &registered_mri[mid_s..mid_s + data.ny_vis * data.nx_vis];
    let ncc_slice_reg = ncc(
        &normalize_minmax(&data.ct_slice),
        &normalize_minmax(reg_slice),
    );
    let png_cma_path = data.output_dir.join("rire_registration_cma_es.png");
    save_comparison_png(
        &data.ct_slice,
        &data.pre_slice,
        reg_slice,
        [data.ny_vis, data.nx_vis],
        &png_cma_path,
    )?;
    println!(
        " Saved: {} (CT | pre-reg | Δ | CMA-ES-registered)",
        png_cma_path.display()
    );
    println!(
        " Mid-slice NCC: pre={:.4} CMA-ES={:.4} GT={:.4}",
        data.ncc_slice_pre, ncc_slice_reg, data.ncc_slice_post
    );
    println!();

    if tre_cma < tre_id {
        println!(
            " ✓ CMA-ES reduced TRE from {:.1} → {:.1} mm ({:.1}% improvement).",
            tre_id,
            tre_cma,
            100.0 * (tre_id - tre_cma) / tre_id
        );
        if ncc_registered > ncc_identity_vol {
            println!(
                " ✓ NCC improved from {:.4} → {:.4} (CT/MRI cross-modal alignment better).",
                ncc_identity_vol, ncc_registered
            );
        }
    } else {
        println!(" ✗ CMA-ES did not reduce TRE on this run.");
        println!(" Try increasing max_generations or translation_range_mm.");
    }
    println!();

    Ok(())
}
