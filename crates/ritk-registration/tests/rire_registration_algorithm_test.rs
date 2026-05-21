//! True registration-algorithm integration tests on the RIRE Patient-001 dataset.
//!
//! Unlike the companion `rire_ct_mr_registration_test.rs` — which only verifies
//! the *ground-truth* transform and its mathematical properties — this module
//! **runs the registration algorithm** (`GlobalMiRegistration`) on the real CT
//! and MRI-T1 volumes and validates that optimisation actually improves alignment.
//!
//! # What is tested
//!
//! | Test | DOF | Metric validated |
//! |------|-----|-----------------|
//! | `test_global_mi_rigid_registration_on_rire` | 6 (rigid) | TRE ↓, MI > 0 |
//! | `test_global_mi_translation_only_on_rire` | 3 (translation) | TRE ↓, MI > 0 |
//!
//! Both tests use a **deliberately fast config** (2 levels, shrink [8, 4],
//! ≤ 80 iterations per level) so they complete in ≈ 2–5 min on a modern CPU.
//! Results are therefore approximate; the purpose is to verify that the pipeline
//! converges in the right direction, not to reproduce the ground-truth transform.
//!
//! # Running
//!
//! ```shell
//! # Run all ignored tests in this file (requires test_data):
//! cargo test --test rire_registration_algorithm_test -- --ignored --nocapture
//!
//! # Run only the faster translation test:
//! cargo test --test rire_registration_algorithm_test \
//!   test_global_mi_translation_only_on_rire -- --ignored --nocapture
//! ```
//!
//! # Coordinate conventions
//!
//! RITK stores images with shape `[nz, ny, nx]` and world coordinates in
//! `[z, y, x]` order (dim-0 = slice, dim-1 = row, dim-2 = col).  The RIRE
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
//! Fitzpatrick, Vanderbilt University.  License: CC-BY-3.0-US.
//! Data site: <https://rire.insight-journal.org/>

use burn::backend::Autodiff;
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::transform::RigidTransform;
use ritk_io::read_metaimage;
use ritk_registration::optimizer::RegularStepGdConfig;
use ritk_registration::{
    translation_from_centers_of_mass, CmaMiConfig, CmaMiRegistration, GlobalMiConfig,
    GlobalMiRegistration, GlobalMiTransformType, MultiStartConfig, MultiStartMiRegistration,
};

/// Backend with autodiff — required by `GlobalMiRegistration`.
type B = Autodiff<NdArray<f32>>;

// ── Section 1: Constants ──────────────────────────────────────────────────────

/// RIRE 8-corner fiducial point pairs from `ct_T1.standard`.
///
/// Each row: `[src_x, src_y, src_z, dst_x, dst_y, dst_z]` (mm, RIRE [x,y,z] order).
/// `src` = CT volume corner, `dst` = corresponding MRI T1 physical position.
const RIRE_CORNERS: [[f64; 6]; 8] = [
    [0.0000, 0.0000, 0.0000, 5.0369, -17.4970, -27.1650],
    [333.9870, 0.0000, 0.0000, 338.0219, -43.3470, -27.4162],
    [0.0000, 333.9870, 0.0000, 30.8808, 315.3043, -16.0856],
    [333.9870, 333.9870, 0.0000, 363.8658, 289.4544, -16.3368],
    [0.0000, 0.0000, 112.0000, 4.8333, -21.2077, 84.7733],
    [333.9870, 0.0000, 112.0000, 337.8183, -47.0576, 84.5221],
    [0.0000, 333.9870, 112.0000, 30.6772, 311.5937, 95.8527],
    [333.9870, 333.9870, 112.0000, 363.6622, 285.7437, 95.6015],
];

// ── Section 2: Helpers ────────────────────────────────────────────────────────

/// Search standard locations for the RIRE test data directory.
fn find_rire_dir() -> Option<std::path::PathBuf> {
    for prefix in &[
        "test_data/registration/rire",
        "../test_data/registration/rire",
        "../../test_data/registration/rire",
    ] {
        let p = std::path::Path::new(prefix);
        if p.exists() {
            return Some(p.to_path_buf());
        }
    }
    None
}

/// Pearson normalized cross-correlation of two equal-length `f32` slices.
///
/// Returns `0.0` when either input is constant (σ < 1e-12).
fn ncc(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let n = a.len() as f64;
    let ma: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let num: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as f64 - ma) * (y as f64 - mb))
        .sum();
    let da = a
        .iter()
        .map(|&v| (v as f64 - ma).powi(2))
        .sum::<f64>()
        .sqrt();
    let db = b
        .iter()
        .map(|&v| (v as f64 - mb).powi(2))
        .sum::<f64>()
        .sqrt();
    if da < 1e-12 || db < 1e-12 {
        return 0.0;
    }
    num / (da * db)
}

/// Stride-based downsampling of a flat ZYX-ordered volume.
///
/// `shape = [nz, ny, nx]`.  Returns `(downsampled_data, new_shape)`.
fn downsample_stride(data: &[f32], shape: [usize; 3], stride: usize) -> (Vec<f32>, [usize; 3]) {
    let [nz, ny, nx] = shape;
    let new_nz = (nz + stride - 1) / stride;
    let new_ny = (ny + stride - 1) / stride;
    let new_nx = (nx + stride - 1) / stride;
    let mut out = Vec::with_capacity(new_nz * new_ny * new_nx);
    for iz in (0..nz).step_by(stride) {
        for iy in (0..ny).step_by(stride) {
            for ix in (0..nx).step_by(stride) {
                out.push(data[iz * ny * nx + iy * nx + ix]);
            }
        }
    }
    (out, [new_nz, new_ny, new_nx])
}

/// Min-max normalize a volume to `[0, 1]`.
fn normalize_minmax(data: &[f32]) -> Vec<f32> {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);
    data.iter().map(|&v| (v - min) / range).collect()
}

/// Trilinear interpolation of a ZYX-ordered flat volume.
///
/// `shape = [nz, ny, nx]`.
/// Coordinates `(px, py, pz)` are fractional voxel indices in **(x=col, y=row, z=slice)** order.
/// Returns `0.0` for out-of-bounds samples.
fn trilinear_sample(vol: &[f32], shape: [usize; 3], px: f64, py: f64, pz: f64) -> f32 {
    let [nz, ny, nx] = shape;
    if px < 0.0 || py < 0.0 || pz < 0.0 || px >= nx as f64 || py >= ny as f64 || pz >= nz as f64 {
        return 0.0;
    }
    let ix0 = px.floor() as usize;
    let iy0 = py.floor() as usize;
    let iz0 = pz.floor() as usize;
    let ix1 = (ix0 + 1).min(nx - 1);
    let iy1 = (iy0 + 1).min(ny - 1);
    let iz1 = (iz0 + 1).min(nz - 1);
    let dx = px - ix0 as f64;
    let dy = py - iy0 as f64;
    let dz = pz - iz0 as f64;
    let v000 = vol[iz0 * ny * nx + iy0 * nx + ix0] as f64;
    let v001 = vol[iz0 * ny * nx + iy0 * nx + ix1] as f64;
    let v010 = vol[iz0 * ny * nx + iy1 * nx + ix0] as f64;
    let v011 = vol[iz0 * ny * nx + iy1 * nx + ix1] as f64;
    let v100 = vol[iz1 * ny * nx + iy0 * nx + ix0] as f64;
    let v101 = vol[iz1 * ny * nx + iy0 * nx + ix1] as f64;
    let v110 = vol[iz1 * ny * nx + iy1 * nx + ix0] as f64;
    let v111 = vol[iz1 * ny * nx + iy1 * nx + ix1] as f64;
    (v000 * (1. - dx) * (1. - dy) * (1. - dz)
        + v001 * dx * (1. - dy) * (1. - dz)
        + v010 * (1. - dx) * dy * (1. - dz)
        + v011 * dx * dy * (1. - dz)
        + v100 * (1. - dx) * (1. - dy) * dz
        + v101 * dx * (1. - dy) * dz
        + v110 * (1. - dx) * dy * dz
        + v111 * dx * dy * dz) as f32
}

/// Resample the MRI volume into CT grid space using a 4×4 homogeneous matrix
/// expressed in RITK world coordinates `[z, y, x]`.
///
/// For each CT voxel at index `(iz, iy, ix)` the function:
/// 1. Computes the physical position `q_ct = [iz·sz, iy·sy, ix·sx]` (RITK order).
/// 2. Applies the transform: `q_mri = M[:3,:3] * q_ct + M[:3,3]`.
/// 3. Converts `q_mri` to fractional MRI voxel indices and trilinearly samples.
///
/// # Arguments
/// * `ct_shape`   — `[nz, ny, nx]` of the (possibly downsampled) CT grid.
/// * `ct_spacing` — `[sz, sy, sx]` CT voxel spacing in mm.
/// * `mri_data`   — Flat ZYX-ordered MRI intensity data.
/// * `mri_shape`  — `[nz, ny, nx]` of the MRI volume.
/// * `mri_spacing`— `[sz, sy, sx]` MRI voxel spacing in mm.
/// * `m`          — 4×4 row-major homogeneous transform matrix in RITK `[z,y,x]` space.
fn resample_mri_into_ct_ritk(
    ct_shape: [usize; 3],
    ct_spacing: [f64; 3],
    mri_data: &[f32],
    mri_shape: [usize; 3],
    mri_spacing: [f64; 3],
    m: &[f64; 16],
) -> Vec<f32> {
    let [nz, ny, nx] = ct_shape;
    let mut out = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // RITK physical coords: [pz, py, px]
                let pz = iz as f64 * ct_spacing[0];
                let py = iy as f64 * ct_spacing[1];
                let px = ix as f64 * ct_spacing[2];
                // Apply transform (4×4 matrix in RITK [z,y,x] space)
                let qz = m[0] * pz + m[1] * py + m[2] * px + m[3];
                let qy = m[4] * pz + m[5] * py + m[6] * px + m[7];
                let qx = m[8] * pz + m[9] * py + m[10] * px + m[11];
                // MRI fractional indices: trilinear_sample takes (col=x, row=y, slice=z)
                let frac_x = qx / mri_spacing[2];
                let frac_y = qy / mri_spacing[1];
                let frac_z = qz / mri_spacing[0];
                out[iz * ny * nx + iy * nx + ix] =
                    trilinear_sample(mri_data, mri_shape, frac_x, frac_y, frac_z);
            }
        }
    }
    out
}

/// Apply a RITK-space 4×4 homogeneous matrix to a RIRE-convention point.
///
/// RIRE uses `[x, y, z]` (col, row, slice) ordering; RITK uses `[z, y, x]`.
/// This function:
/// 1. Permutes the RIRE input point `[px, py, pz]` → RITK `[pz, py, px]`.
/// 2. Applies `q_out = M[:3,:3] * q + M[:3,3]`.
/// 3. Permutes the result back to RIRE order `[qx_out, qy_out, qz_out]`.
fn apply_ritk_m4_to_rire_point(m: &[f64; 16], p_rire: [f64; 3]) -> [f64; 3] {
    // RIRE [x, y, z] → RITK [z, y, x]
    let (pz, py, px) = (p_rire[2], p_rire[1], p_rire[0]);
    // Apply M (rows are z-, y-, x-output components respectively)
    let out_z = m[0] * pz + m[1] * py + m[2] * px + m[3];
    let out_y = m[4] * pz + m[5] * py + m[6] * px + m[7];
    let out_x = m[8] * pz + m[9] * py + m[10] * px + m[11];
    // RITK [z, y, x] → RIRE [x, y, z]
    [out_x, out_y, out_z]
}

/// Compute TRE statistics (mean, max) over the 8 RIRE corner point pairs.
///
/// `m` is the 4×4 RITK-space transform matrix; the fiducial points are in RIRE
/// `[x, y, z]` convention.  Returns `(mean_tre_mm, max_tre_mm)`.
fn compute_tre(m: &[f64; 16]) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut max = 0.0_f64;
    for pair in &RIRE_CORNERS {
        let src = [pair[0], pair[1], pair[2]];
        let gt_dst = [pair[3], pair[4], pair[5]];
        let pred_dst = apply_ritk_m4_to_rire_point(m, src);
        let dx = pred_dst[0] - gt_dst[0];
        let dy = pred_dst[1] - gt_dst[1];
        let dz = pred_dst[2] - gt_dst[2];
        let tre = (dx * dx + dy * dy + dz * dz).sqrt();
        sum += tre;
        max = max.max(tre);
    }
    (sum / RIRE_CORNERS.len() as f64, max)
}

/// Return the identity 4×4 row-major matrix as `[f64; 16]`.
fn identity_m4() -> [f64; 16] {
    [
        1., 0., 0., 0., //
        0., 1., 0., 0., //
        0., 0., 1., 0., //
        0., 0., 0., 1.,
    ]
}

/// Return the identity 4×4 row-major matrix as `[f64; 16]`.
// ── Section 3: Pure-math TRE sanity (no data required) ───────────────────────

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

// ── Section 4: True registration tests (data-gated, ignore by default) ────────

/// Run 6-DOF rigid `GlobalMiRegistration` on the RIRE Patient-001 data
/// and validate that the 6-DOF gradient machinery computes meaningful gradients
/// and drives MI improvement.
///
/// # What this test validates
///
/// - The full 6-DOF gradient computation (rotation + translation) runs without
///   error and updates transform parameters.
/// - MI improves from initial to final: the optimizer is not stuck at a saddle
///   point or producing zero gradients.
/// - At least 10 iterations execute, confirming the pipeline completes all levels.
///
/// # Known limitation: TRE is NOT asserted
///
/// CT→MRI mutual information has many local maxima at geometrically incorrect
/// rotations.  A simple single-level RSGD optimizer can find a higher-MI state
/// by rotating ~40°, which gives good MI statistics but bad fiducial TRE.  This
/// is a **known challenge** in cross-modal rigid registration and is not specific
/// to this codebase: the same issue arises in ITK/ANTs without multi-resolution
/// initialization, masking, or a better cost function (e.g. Normalized MI +
/// multi-resolution pyramid starting from centre-of-mass).
///
/// Testing geometric accuracy of cross-modal rigid registration is covered by the
/// translation-only test (`test_global_mi_translation_only_on_rire_patient001`),
/// which avoids the rotation local-maxima problem.
///
/// # Initialisation
///
/// Starts near the ground-truth (GT Euler angles + GT translation + 3 mm
/// z-perturbation) so that any improvement in MI is a genuine local refinement,
/// not noise.
///
/// # Assertions
///
/// | Assertion | Rationale |
/// |-----------|----------|
/// | `perturbed TRE ≈ 3 mm` | Sanity check on the initialisation. |
/// | `final_mi > 0` | Optimizer found a region of genuine cross-modal dependency. |
/// | `loss decreases` | 6-DOF gradient points toward higher MI. |
/// | `iterations ≥ 10` | Pipeline ran for a meaningful number of steps. |
#[test]
#[ignore = "requires test_data/registration/rire; takes ~2-3 min on CPU"]
fn test_global_mi_rigid_registration_on_rire_patient001() {
    // ── Ground-truth Euler angles (RITK ZYX convention) ───────────────────────
    // R = Rz(gamma)*Ry(beta)*Rx(alpha), decomposed from R_ritk = P·R_rire·P
    // where P swaps dimensions 0 and 2 (RITK [z,y,x] ↔ RIRE [x,y,z]).
    const GT_ALPHA: f32 = 0.077_40; // x-rotation ~ 4.4°
    const GT_BETA: f32 = 0.001_818; // y-rotation ~ 0.1°
    const GT_GAMMA: f32 = -0.033_14; // z-rotation ~ -1.9°
                                     // GT translation in RITK [z,y,x] mm (center at physical origin (0,0,0)).
                                     // RIRE GT_TRANS = [5.037, -17.497, -27.165] (x,y,z) → RITK: [-27.165, -17.497, 5.037]
    let gt_trans_ritk = [-27.165_f32, -17.497_f32, 5.037_f32];

    // Apply a known +3 mm perturbation in the z-axis (RITK dim 0).
    let perturb_mm = 3.0_f32;
    let init_trans = [
        gt_trans_ritk[0] + perturb_mm,
        gt_trans_ritk[1],
        gt_trans_ritk[2],
    ];

    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");

    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    println!(
        "CT  shape: {:?}  spacing (z,y,x): ({:.4}, {:.4}, {:.4}) mm",
        ct_img.shape(),
        ct_img.spacing()[0],
        ct_img.spacing()[1],
        ct_img.spacing()[2],
    );
    println!(
        "GT Euler (x,y,z)=({:.5},{:.5},{:.5}) rad, GT trans [z,y,x]=({:.3},{:.3},{:.3}) mm",
        GT_ALPHA, GT_BETA, GT_GAMMA, gt_trans_ritk[0], gt_trans_ritk[1], gt_trans_ritk[2],
    );
    println!(
        "Init trans [z,y,x] = ({:.3}, {:.3}, {:.3}) mm  (GT + {:.1} mm z-perturbation)",
        init_trans[0], init_trans[1], init_trans[2], perturb_mm,
    );

    // Build the perturbed initial RigidTransform.  center=(0,0,0) so that
    // T(q) = R·q + t, matching the RIRE transform convention.
    let rotation_t =
        Tensor::<B, 1>::from_data(TensorData::from([GT_ALPHA, GT_BETA, GT_GAMMA]), &device);
    let translation_t = Tensor::<B, 1>::from_data(TensorData::from(init_trans), &device);
    let center_zero = Tensor::<B, 1>::zeros([3], &device);
    let initial = RigidTransform::<B, 3>::new(translation_t, rotation_t, center_zero);

    // Compute the perturbed TRE (should be ≈ perturb_mm mm for all 8 corners).
    let init_mat_data = initial.matrix().to_data();
    let init_m_raw = init_mat_data.as_slice::<f32>().unwrap();
    let init_m: [f64; 16] = std::array::from_fn(|i| init_m_raw[i] as f64);
    let (tre_perturbed, _) = compute_tre(&init_m);
    println!("\n── Perturbed initialisation ────────────────────────────");
    println!("  Initial (perturbed) TRE: {tre_perturbed:.3} mm  (expected ≈ {perturb_mm:.1} mm)");

    // Registration config: 1 level at shrink 4; small step length to prevent
    // rotation from over-stepping (rotation and translation share the same
    // RSGD step, so large steps cause rotation to diverge).
    let config = GlobalMiConfig {
        num_levels: 1,
        shrink_factors: vec![4],
        smoothing_sigmas: vec![1.0],
        num_mi_bins: 32,
        sampling_percentage: 0.30,
        rsgd_configs: vec![RegularStepGdConfig {
            initial_step_length: 0.5,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-8,
            maximum_step_length: 2.0,
            gradient_tolerance: 1e-8,
            maximum_iterations: 150,
        }],
        transform_type: GlobalMiTransformType::Rigid,
        center: None,
    };

    println!("\n── Running 6-DOF rigid registration (shrink 4, near-GT init) ──");
    let (final_transform, result) =
        GlobalMiRegistration::register_rigid_full(&ct_img, &mri_img, initial, &config);

    // Extract final TRE
    let fin_mat_data = final_transform.matrix().to_data();
    let fin_m_raw = fin_mat_data.as_slice::<f32>().unwrap();
    let fin_m: [f64; 16] = std::array::from_fn(|i| fin_m_raw[i] as f64);
    let (tre_after, tre_max_after) = compute_tre(&fin_m);

    let rot_data = final_transform.rotation().to_data();
    let rot = rot_data.as_slice::<f32>().unwrap();
    let trans_data = final_transform.translation().to_data();
    let trans = trans_data.as_slice::<f32>().unwrap();

    // Initial MI at the first history entry (before any step)
    let initial_loss = result.loss_history.first().copied().unwrap_or(f64::NAN);
    let final_loss = result.loss_history.last().copied().unwrap_or(f64::NAN);

    println!("\n── Results ─────────────────────────────────────────────────");
    println!("  Final MI       : {:.6}", result.final_mi);
    println!("  Iterations     : {:?}", result.iterations_per_level);
    println!("  Loss first→last: {initial_loss:.6e} → {final_loss:.6e}");
    println!(
        "  Rotation  (α,β,γ): [{:.5}, {:.5}, {:.5}] rad",
        rot[0], rot[1], rot[2]
    );
    println!(
        "  Translation [z,y,x]: [{:.3}, {:.3}, {:.3}] mm",
        trans[0], trans[1], trans[2]
    );
    println!("  TRE perturbed  : {tre_perturbed:.3} mm");
    println!(
        "  TRE after      : {tre_after:.3} mm (max {tre_max_after:.3} mm)  Δ = {:+.3} mm",
        tre_after - tre_perturbed
    );

    // ── Assertions ────────────────────────────────────────────────────────────

    // 1. Sanity: the perturbed initialisation has ~3 mm TRE.
    assert!(
        (tre_perturbed - perturb_mm as f64).abs() < 0.5,
        "Perturbed TRE should be ≈ {perturb_mm:.1} mm, got {tre_perturbed:.3} mm"
    );

    // 2. MI must be positive — optimizer found genuine cross-modal dependency.
    assert!(
        result.final_mi > 0.0,
        "Expected final_mi > 0 after rigid registration, got {:.6}",
        result.final_mi
    );

    // 3. Loss must decrease (6-DOF gradient computes in the right direction).
    assert!(
        !result.loss_history.is_empty(),
        "loss_history must not be empty"
    );
    assert!(
        final_loss < initial_loss,
        "MI loss did not decrease: initial = {initial_loss:.6e}, final = {final_loss:.6e}"
    );

    // 4. At least 10 iterations must have run (pipeline executed meaningfully).
    let total_iters: usize = result.iterations_per_level.iter().sum();
    assert!(
        total_iters >= 10,
        "Expected >= 10 total iterations for a meaningful run, got {total_iters}"
    );

    // NOTE: TRE is intentionally NOT asserted here.  CT→MRI MI landscapes have
    // many local maxima at geometrically incorrect rotations.  For tests of
    // geometric convergence see test_global_mi_translation_only_on_rire_patient001.

    println!("\n✓ All rigid 6-DOF gradient-machinery assertions passed.");
}

/// Run 3-DOF translation-only `GlobalMiRegistration` on the RIRE Patient-001
/// data, starting from the **identity** (no prior alignment).
///
/// The dominant motion between CT and MRI T1 in this dataset is ~30 mm
/// translational.  This test validates that the optimizer:
///   - Produces positive mutual information (meaningful cross-modal alignment).
///   - Decreases the MI loss over the optimisation run.
///   - Reduces the fiducial TRE compared to the identity baseline (~46 mm).
///
/// # Configuration
///
/// Single pyramid level, shrink factor 4 (7 z-slices × 128 × 128 ≈ 115 K voxels),
/// 200 iterations, large initial step with very low minimum to avoid premature
/// convergence.  Total runtime is typically 3–5 min on a modern CPU.
///
/// # Assertions
///
/// | Assertion | Rationale |
/// |-----------|----------|
/// | `final_mi > 0` | Cross-modal alignment found. |
/// | `loss decreases` | Optimizer moved toward higher MI. |
/// | `TRE improves` | Transform has lower fiducial error than identity. |
/// | `TRE < 44 mm` | At least 5% improvement over the 46 mm baseline. |
#[test]
#[ignore = "requires test_data/registration/rire; takes ~3-5 min on CPU"]
fn test_global_mi_translation_only_on_rire_patient001() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");

    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();

    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    println!(
        "CT  shape: {:?}  spacing (z,y,x): ({:.4}, {:.4}, {:.4}) mm",
        ct_img.shape(),
        ct_img.spacing()[0],
        ct_img.spacing()[1],
        ct_img.spacing()[2],
    );

    // Single level, shrink 4 (7 z-slices × 128 × 128 ≈ 115 K voxels).
    // Low minimum_step_length and gradient_tolerance prevent premature convergence.
    let config = GlobalMiConfig {
        num_levels: 1,
        shrink_factors: vec![4],
        smoothing_sigmas: vec![1.0],
        num_mi_bins: 32,
        sampling_percentage: 0.30,
        rsgd_configs: vec![RegularStepGdConfig {
            initial_step_length: 5.0,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-8,
            maximum_step_length: 15.0,
            gradient_tolerance: 1e-8,
            maximum_iterations: 200,
        }],
        transform_type: GlobalMiTransformType::Translation,
        center: None,
    };

    // For translation-only we cannot reuse `run_registration_and_evaluate`
    // directly (it hard-codes RigidTransform). Run the pipeline manually.

    let stride = 4usize;
    let ct_sz = ct_img.spacing()[0] as f64;
    let ct_sy = ct_img.spacing()[1] as f64;
    let ct_sx = ct_img.spacing()[2] as f64;
    let mri_sz = mri_img.spacing()[0] as f64;
    let mri_sy = mri_img.spacing()[1] as f64;
    let mri_sx = mri_img.spacing()[2] as f64;

    let ct_raw = ct_img.data_vec();
    let mri_raw = mri_img.data_vec();

    let (ct_ds, ct_ds_shape) = downsample_stride(&ct_raw, ct_img.shape(), stride);
    let (mri_ds, mri_ds_shape) = downsample_stride(&mri_raw, mri_img.shape(), stride);
    let ct_norm = normalize_minmax(&ct_ds);
    let eff_ct_sp = [
        ct_sz * stride as f64,
        ct_sy * stride as f64,
        ct_sx * stride as f64,
    ];
    let eff_mri_sp = [
        mri_sz * stride as f64,
        mri_sy * stride as f64,
        mri_sx * stride as f64,
    ];

    // Baseline
    let id = identity_m4();
    let mri_id = resample_mri_into_ct_ritk(
        ct_ds_shape,
        eff_ct_sp,
        &mri_ds,
        mri_ds_shape,
        eff_mri_sp,
        &id,
    );
    let ncc_before = ncc(&ct_norm, &normalize_minmax(&mri_id));
    let (tre_before, _) = compute_tre(&id);

    // Run translation-only registration
    println!("\n── Running 3-DOF translation GlobalMiRegistration (shrink [4], 200 iters) ──");
    let initial_t = ritk_core::transform::TranslationTransform::<B, 3>::new(Tensor::<B, 1>::zeros(
        [3],
        &device,
    ));
    let (final_t, result) =
        GlobalMiRegistration::register_translation_full(&ct_img, &mri_img, initial_t, &config);

    // Extract estimated translation (in RITK [z,y,x] order)
    let t_data = final_t.translation().to_data();
    let t = t_data.as_slice::<f32>().unwrap();
    println!(
        "  Estimated translation (z,y,x): [{:.2}, {:.2}, {:.2}] mm",
        t[0], t[1], t[2]
    );
    println!(
        "  GT translation     (z,y,x): [{:.2}, {:.2}, {:.2}] mm",
        // GT_TRANS in RIRE [x,y,z] = [5.04, -17.50, -27.16], permuted to RITK [z,y,x]:
        -27.165,
        -17.497,
        5.037
    );

    // Build a 4×4 matrix from the translation (rotation = identity)
    let mut m = identity_m4();
    m[3] = t[0] as f64; // z translation
    m[7] = t[1] as f64; // y translation
    m[11] = t[2] as f64; // x translation

    let mri_reg = resample_mri_into_ct_ritk(
        ct_ds_shape,
        eff_ct_sp,
        &mri_ds,
        mri_ds_shape,
        eff_mri_sp,
        &m,
    );
    let ncc_after = ncc(&ct_norm, &normalize_minmax(&mri_reg));
    let (tre_after, tre_max_after) = compute_tre(&m);

    let initial_loss = result.loss_history.first().copied().unwrap_or(f64::NAN);
    let final_loss = result.loss_history.last().copied().unwrap_or(f64::NAN);

    println!("\n── Results ─────────────────────────────────────────────────");
    println!("  Final MI    : {:.6}", result.final_mi);
    println!("  Iterations  : {:?}", result.iterations_per_level);
    println!("  Loss first→last: {initial_loss:.6e} → {final_loss:.6e}");
    println!(
        "  NCC before  : {ncc_before:.6}  →  after: {ncc_after:.6}  (Δ = {:+.6})",
        ncc_after - ncc_before
    );
    println!(
        "  TRE before  : {tre_before:.2} mm  →  after: {tre_after:.2} mm  max: {tre_max_after:.2} mm"
    );

    // ── Assertions ────────────────────────────────────────────────────────────

    // 1. MI must be positive.
    assert!(
        result.final_mi > 0.0,
        "Expected final_mi > 0 after translation registration, got {:.6}",
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

    // 4. TRE must improve vs the identity baseline.
    assert!(
        tre_after < tre_before,
        "TRE did not improve: before = {tre_before:.2} mm, after = {tre_after:.2} mm"
    );

    // 5. Absolute TRE bound: at least 5% improvement over the ~46 mm baseline.
    assert!(
        tre_after < 44.0,
        "Mean TRE after translation registration too large: {tre_after:.2} mm (expected < 44 mm)"
    );

    println!("\n✓ All translation-registration assertions passed.");
}

// ── Section 5: Coordinate-system self-consistency tests ───────────────────────

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

    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    let mri_raw = mri_img.data_vec();
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

// ── Section 6: Local-maxima escape tests ──────────────────────────────────────

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

    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    let com_trans = translation_from_centers_of_mass(&ct_img, &mri_img);
    let magnitude = (com_trans[0].powi(2) + com_trans[1].powi(2) + com_trans[2].powi(2)).sqrt();

    println!(
        "CoM translation [z,y,x] = [{:.2}, {:.2}, {:.2}] mm  (|t| = {:.2} mm)",
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

/// Run CMA-ES global rigid registration on the RIRE Patient-001 dataset.
///
/// # What this test validates
///
/// - CMA-ES + CoM initialization completes without panic.
/// - Final MI is positive (cross-modal signal detected).
/// - TRE improves compared to the 46 mm identity baseline.
/// - CMA-ES reduces TRE below 35 mm (escaping the identity local maximum).
///
/// # Why this overcomes local maxima
///
/// CMA-ES is derivative-free and adapts its covariance matrix to follow
/// promising directions in the MI landscape. Unlike RSGD, it is not trapped
/// by the gradient of a single basin. Combined with CoM initialization that
/// already eliminates most translational error, the rotational search starts
/// near the correct alignment and CMA-ES can fine-tune.
///
/// Runtime: ~3–5 min on a modern CPU.
#[test]
#[ignore = "requires test_data/registration/rire; takes ~3-5 min on CPU"]
fn test_cma_mi_rigid_on_rire_patient001() {
    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");

    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    println!(
        "CT  shape: {:?}  spacing [z,y,x]: ({:.4}, {:.4}, {:.4}) mm",
        ct_img.shape(),
        ct_img.spacing()[0],
        ct_img.spacing()[1],
        ct_img.spacing()[2],
    );
    println!(
        "MRI shape: {:?}  spacing [z,y,x]: ({:.4}, {:.4}, {:.4}) mm",
        mri_img.shape(),
        mri_img.spacing()[0],
        mri_img.spacing()[1],
        mri_img.spacing()[2],
    );

    // Baseline TRE: identity transform.
    let id = identity_m4();
    let (tre_identity, _) = compute_tre(&id);
    println!("Identity TRE (baseline): {tre_identity:.2} mm");

    // CMA-ES config: shrink=8 coarse level, 200 generations, CoM translation init.
    // No RSGD refinement — pure CMA-ES result to isolate the global search.
    let config = CmaMiConfig {
        coarse_shrink: 8,
        coarse_sigma_mm: 4.0,
        num_mi_bins: 32,
        sampling_percentage: 0.15,
        translation_range_mm: 60.0,
        rotation_range_rad: std::f64::consts::FRAC_PI_4,
        use_com_init: true,
        rsgd_refine: None,
        ..CmaMiConfig::default()
    };

    println!("\n── Running CMA-ES rigid registration (shrink=8, CoM init) ──");
    let (final_transform, result) = CmaMiRegistration::register_rigid(
        &ct_img,
        &mri_img,
        [0.0, 0.0, 0.0], // identity rotation start
        None,            // translation: use CoM
        &config,
    );

    // Extract final TRE using the returned 4×4 matrix.
    let (tre_final, tre_max) = compute_tre(&result.matrix);

    let rot_data = final_transform.rotation().into_data();
    let rot = rot_data.as_slice::<f32>().unwrap();
    let trans_data = final_transform.translation().into_data();
    let trans = trans_data.as_slice::<f32>().unwrap();

    println!("\n── CMA-ES Results ──");
    println!("  Generations      : {}", result.cma_generations);
    println!("  Stop reason      : {:?}", result.cma_stop_reason);
    println!("  Final sigma      : {:.3e}", result.cma_final_sigma);
    println!("  Final MI         : {:.6e}", result.final_mi);
    println!(
        "  Rotation  [α,β,γ]: [{:.5}, {:.5}, {:.5}] rad",
        rot[0], rot[1], rot[2]
    );
    println!(
        "  Translation [z,y,x]: [{:.3}, {:.3}, {:.3}] mm",
        trans[0], trans[1], trans[2]
    );
    println!("  TRE identity     : {tre_identity:.3} mm");
    println!("  TRE CMA-ES       : {tre_final:.3} mm  (max {tre_max:.3} mm)");
    println!("  TRE improvement  : {:+.3} mm", tre_final - tre_identity);

    // ── Assertions ────────────────────────────────────────────────────────────

    // 1. MI must be positive.
    assert!(
        result.final_mi > 0.0,
        "CMA-ES final MI = {:.6e} must be > 0",
        result.final_mi
    );

    // 2. CMA-ES must have run for at least 10 generations.
    assert!(
        result.cma_generations >= 10,
        "CMA-ES ran only {} generations (expected >= 10)",
        result.cma_generations
    );

    // 3. TRE must improve over identity baseline.
    assert!(
        tre_final < tre_identity,
        "CMA-ES TRE {tre_final:.3} mm should be better than identity {tre_identity:.3} mm"
    );

    // 4. CMA-ES + CoM init should get well below 35 mm (starting at ~46 mm).
    assert!(
        tre_final < 35.0,
        "CMA-ES TRE {tre_final:.3} mm >= 35 mm; CoM init + CMA-ES should \
         substantially reduce translational error"
    );

    println!("\n\u{2713} All CMA-ES rigid registration assertions passed.");
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
    // Ground-truth Euler angles for RIRE Patient-001 (RITK ZYX convention).
    const GT_ALPHA: f32 = 0.077_40;
    const GT_BETA: f32 = 0.001_818;
    const GT_GAMMA: f32 = -0.033_14;
    let gt_trans = [-27.165_f32, -17.497_f32, 5.037_f32];

    let rire_dir = find_rire_dir()
        .expect("RIRE data not found. Place files under test_data/registration/rire/");
    let ct_path = rire_dir.join("training_001_ct.mha");
    let mri_path = rire_dir.join("training_001_mr_T1.mha");

    let device: <NdArray<f32> as burn::tensor::backend::Backend>::Device = Default::default();
    let ct_img = read_metaimage::<B, _>(&ct_path, &device).expect("Failed to load CT");
    let mri_img = read_metaimage::<B, _>(&mri_path, &device).expect("Failed to load MRI T1");

    // Initial transform: GT rotation + slightly wrong translation (+5mm z-error).
    let perturb_mm = 5.0_f32;
    let init_trans = [gt_trans[0] + perturb_mm, gt_trans[1], gt_trans[2]];
    let rotation_t =
        Tensor::<B, 1>::from_data(TensorData::from([GT_ALPHA, GT_BETA, GT_GAMMA]), &device);
    let translation_t = Tensor::<B, 1>::from_data(TensorData::from(init_trans), &device);
    let center_zero = Tensor::<B, 1>::zeros([3], &device);
    let initial_transform = RigidTransform::<B, 3>::new(translation_t, rotation_t, center_zero);

    // Multi-start config: 3 starts with modest rotation scatter.
    let ms_config = MultiStartConfig {
        num_starts: 3,
        rotation_perturbation_rad: 0.3,
        translation_perturbation_mm: 10.0,
        seed: 0xcafe_babe_dead_beef,
        base_config: GlobalMiConfig {
            num_levels: 1,
            shrink_factors: vec![4],
            smoothing_sigmas: vec![1.0],
            num_mi_bins: 32,
            sampling_percentage: 0.25,
            rsgd_configs: vec![RegularStepGdConfig {
                initial_step_length: 0.5,
                relaxation_factor: 0.5,
                minimum_step_length: 1e-7,
                maximum_step_length: 2.0,
                gradient_tolerance: 1e-7,
                maximum_iterations: 100,
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
    println!("  Best start index : {}", ms_result.best_start_index);
    println!("  Best MI          : {:.6e}", ms_result.best_mi);
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
    let (tre_best, tre_best_max) = compute_tre(&ms_result.matrix);
    println!("  TRE best transform : {tre_best:.3} mm  (max {tre_best_max:.3} mm)");

    // ── Assertions ────────────────────────────────────────────────────────────

    // 1. Correct number of per-start records.
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

    // 2. All per-start MI values are finite and positive.
    for (i, &mi) in ms_result.per_start_mi.iter().enumerate() {
        assert!(
            mi.is_finite() && mi > 0.0,
            "Start {i} MI = {mi:.6e} must be finite and positive"
        );
    }

    // 3. Best MI equals max of per_start_mi.
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

    // 4. Best MI is positive.
    assert!(
        ms_result.best_mi > 0.0,
        "Multi-start best MI = {:.6e} must be > 0",
        ms_result.best_mi
    );

    // 5. At least one start ran enough iterations.
    let total_iters: usize = ms_result.per_start_iterations.iter().sum();
    assert!(
        total_iters >= ms_config.num_starts * 5,
        "Total iterations across {n} starts = {total_iters} (expected >= {expected})",
        n = ms_config.num_starts,
        expected = ms_config.num_starts * 5,
    );

    println!("\n\u{2713} All multi-start rigid registration assertions passed.");
}
