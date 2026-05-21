//! RIRE CT/MR T1 registration integration tests.
//!
//! # Test structure
//!
//! Tests are split into two groups:
//!
//! ## Group 1 — Pure math tests (always run, no `#[ignore]`)
//!
//! These tests exercise the rigid-transform math helpers (apply, inverse,
//! orthogonality) and the 8-corner fiducial verification from the RIRE
//! standard file `ct_T1.standard`.  They require no external data and run in
//! milliseconds.
//!
//! ## Group 2 — Image integration tests (`#[ignore = "requires test_data/…"]`)
//!
//! These tests load the actual `.mha` volumes, check spatial metadata, and
//! verify that the ground-truth transform improves CT↔MRI alignment measured
//! by Pearson NCC.  They require the RIRE test data under
//! `test_data/registration/rire/` and can be run with:
//!
//! ```shell
//! cargo test --test rire_ct_mr_registration_test -- --ignored
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
//!
//! # Coordinate conventions
//!
//! Both RIRE volumes have identity direction cosines and origin (0,0,0).
//! Physical coordinates follow the RIRE convention:
//!
//! - x-axis: along image columns (`ix * spacing_x`)
//! - y-axis: along image rows    (`iy * spacing_y`)
//! - z-axis: along slices        (`iz * spacing_z`)
//!
//! The ground-truth transform maps CT physical coordinates to MRI T1 physical
//! coordinates: `T(p) = R · p + t`.  Center of rotation is (0,0,0).
//!
//! RITK tensor convention: shape `[nz, ny, nx]`, flat index
//! `iz * ny * nx + iy * nx + ix`.  `image.spacing()[0]` = z (slice),
//! `[1]` = y (row), `[2]` = x (col).

use burn_ndarray::NdArray;
use ritk_io::read_metaimage;
use std::f64::consts::PI;

type B = NdArray<f32>;

// ── Section 2: Constants ──────────────────────────────────────────────────────

/// Ground-truth rotation matrix R (row-major, 3×3) for the CT → MRI T1
/// transform.  Derived from the ITK Euler3DTransform parameters in
/// `training_001_ct_to_mr_T1_ground_truth.tfm` using the ITK convention
/// `R = Rz(aZ) · Rx(aX) · Ry(aY)`.
///
/// Row 0: [ 0.997000003,  0.077380155, -0.001818059 ]
/// Row 1: [-0.077397855,  0.996449628, -0.033131713 ]
/// Row 2: [-0.000752132,  0.033173032,  0.999449341 ]
const GT_ROT: [f64; 9] = [
    0.997000003,
    0.077380155,
    -0.001818059,
    -0.077397855,
    0.996449628,
    -0.033131713,
    -0.000752132,
    0.033173032,
    0.999449341,
];

/// Ground-truth translation vector t (mm) for the CT → MRI T1 transform.
///
/// `t = [tx, ty, tz] = [5.03685847, -17.49694636, -27.16499259]`
const GT_TRANS: [f64; 3] = [5.03685847, -17.49694636, -27.16499259];

/// RIRE 8-corner point pairs from `ct_T1.standard`.
///
/// Each row: `[src_x, src_y, src_z, dst_x, dst_y, dst_z]` (all mm).
/// Source points are at the eight corners of the CT physical volume
/// (x ∈ {0, 333.987}, y ∈ {0, 333.987}, z ∈ {0, 112}).
/// Destination points are the corresponding MRI T1 physical coordinates
/// as determined by fiducial-marker-based ground-truth registration.
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

// ── Section 3: Pure math helpers ──────────────────────────────────────────────

/// Apply a rigid transform `T(p) = R · p + t` to a 3-D point.
///
/// `r` is a row-major 3×3 rotation matrix stored as a flat 9-element array:
/// `r[i*3 + j]` is the element at row `i`, column `j`.
fn apply_rigid(r: &[f64; 9], t: &[f64; 3], p: &[f64; 3]) -> [f64; 3] {
    [
        r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
        r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
        r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2],
    ]
}

/// Transpose a row-major 3×3 matrix.
///
/// Output element at `(i, j)` = input element at `(j, i)`.
fn mat3_transpose(m: &[f64; 9]) -> [f64; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Multiply two row-major 3×3 matrices: returns `a · b`.
fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut out = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i * 3 + k] * b[k * 3 + j];
            }
            out[i * 3 + j] = s;
        }
    }
    out
}

/// Compute the determinant of a row-major 3×3 matrix using cofactor expansion
/// along the first row.
fn mat3_det(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Compute the inverse of a rigid (rotation + translation) transform.
///
/// For a rigid transform `T(p) = R · p + t`:
/// - `R^{-1} = R^T`  (rotation matrices are orthogonal)
/// - `t^{-1} = −R^T · t`
///
/// Returns `(R^T, −R^T · t)`.
fn rigid_inverse(r: &[f64; 9], t: &[f64; 3]) -> ([f64; 9], [f64; 3]) {
    let r_inv = mat3_transpose(r);
    // t_inv = −R^T · t  (apply R^T to t, then negate)
    let t_inv = [
        -(r_inv[0] * t[0] + r_inv[1] * t[1] + r_inv[2] * t[2]),
        -(r_inv[3] * t[0] + r_inv[4] * t[1] + r_inv[5] * t[2]),
        -(r_inv[6] * t[0] + r_inv[7] * t[1] + r_inv[8] * t[2]),
    ];
    (r_inv, t_inv)
}

/// Pearson normalized cross-correlation of two equal-length `f32` slices.
///
/// Returns `0.0` when either input has standard deviation below `1e-12`
/// (degenerate / constant signal).
fn ncc(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "ncc: inputs must have equal length");
    let n = a.len() as f64;
    let ma: f64 = a.iter().map(|&v| v as f64).sum::<f64>() / n;
    let mb: f64 = b.iter().map(|&v| v as f64).sum::<f64>() / n;
    let num: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x as f64 - ma) * (y as f64 - mb))
        .sum();
    let da: f64 = a
        .iter()
        .map(|&v| (v as f64 - ma).powi(2))
        .sum::<f64>()
        .sqrt();
    let db: f64 = b
        .iter()
        .map(|&v| (v as f64 - mb).powi(2))
        .sum::<f64>()
        .sqrt();
    if da < 1e-12 || db < 1e-12 {
        return 0.0;
    }
    num / (da * db)
}

/// Downsample a flat ZYX-ordered volume by taking every `stride`-th voxel
/// along each axis.
///
/// `shape` is `[nz, ny, nx]`.
///
/// # Returns
/// `(downsampled_flat_vec, [new_nz, new_ny, new_nx])`
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

/// Minmax-normalize a volume to `[0, 1]`.
///
/// When `max == min` the range is clamped to `1e-8` to avoid division by zero,
/// producing an all-zero output.
fn normalize_minmax(data: &[f32]) -> Vec<f32> {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);
    data.iter().map(|&v| (v - min) / range).collect()
}

/// Trilinear interpolation of a ZYX-ordered volume at fractional voxel
/// coordinates `(px, py, pz)` given in **(x = col, y = row, z = slice)**
/// order.
///
/// The volume is indexed as `vol[iz * ny * nx + iy * nx + ix]`.
///
/// Returns `0.0` for any sample whose integer floor indices fall outside the
/// valid voxel range `[0, n-1]`.
fn trilinear_sample(vol: &[f32], shape: [usize; 3], px: f64, py: f64, pz: f64) -> f32 {
    let [nz, ny, nx] = shape;

    // Out-of-bounds: any coordinate strictly outside the voxel grid.
    if px < 0.0 || py < 0.0 || pz < 0.0 || px >= nx as f64 || py >= ny as f64 || pz >= nz as f64 {
        return 0.0;
    }

    let ix0 = px.floor() as usize;
    let iy0 = py.floor() as usize;
    let iz0 = pz.floor() as usize;

    // Clamp the upper neighbor to the last valid index (handles the px == nx-1 edge case).
    let ix1 = (ix0 + 1).min(nx - 1);
    let iy1 = (iy0 + 1).min(ny - 1);
    let iz1 = (iz0 + 1).min(nz - 1);

    let dx = px - ix0 as f64;
    let dy = py - iy0 as f64;
    let dz = pz - iz0 as f64;

    // Fetch the 8 surrounding voxel values.
    let v000 = vol[iz0 * ny * nx + iy0 * nx + ix0] as f64;
    let v001 = vol[iz0 * ny * nx + iy0 * nx + ix1] as f64;
    let v010 = vol[iz0 * ny * nx + iy1 * nx + ix0] as f64;
    let v011 = vol[iz0 * ny * nx + iy1 * nx + ix1] as f64;
    let v100 = vol[iz1 * ny * nx + iy0 * nx + ix0] as f64;
    let v101 = vol[iz1 * ny * nx + iy0 * nx + ix1] as f64;
    let v110 = vol[iz1 * ny * nx + iy1 * nx + ix0] as f64;
    let v111 = vol[iz1 * ny * nx + iy1 * nx + ix1] as f64;

    let val = v000 * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        + v001 * dx * (1.0 - dy) * (1.0 - dz)
        + v010 * (1.0 - dx) * dy * (1.0 - dz)
        + v011 * dx * dy * (1.0 - dz)
        + v100 * (1.0 - dx) * (1.0 - dy) * dz
        + v101 * dx * (1.0 - dy) * dz
        + v110 * (1.0 - dx) * dy * dz
        + v111 * dx * dy * dz;

    val as f32
}

/// Search for the RIRE data directory starting from the current working
/// directory and walking up two levels.
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

// ── Section 4: Data helper ────────────────────────────────────────────────────

/// Resample `mri_data` (a ZYX-ordered flat volume) into the coordinate frame
/// of a CT volume, producing one output sample per CT voxel.
///
/// # Arguments
///
/// * `_ct_data`        — CT voxel data (not read; present for API symmetry).
/// * `ct_shape`        — CT volume shape `[nz, ny, nx]`.
/// * `ct_spacing_xyz`  — CT voxel spacing in **(x/col, y/row, z/slice)** order (mm).
/// * `mri_data`        — Source MRI voxel data in ZYX order.
/// * `mri_shape`       — MRI volume shape `[nz, ny, nx]`.
/// * `mri_spacing_xyz` — MRI voxel spacing in **(x/col, y/row, z/slice)** order (mm).
/// * `r_ct_to_mri`     — Rotation matrix mapping CT physical coords → MRI physical coords.
/// * `t_ct_to_mri`     — Translation vector (mm) for the same mapping.
///
/// # Algorithm
///
/// For each CT voxel `(iz, iy, ix)`:
/// 1. Compute physical CT coords:
///    `px_ct = ix * ct_spacing_xyz[0]`,
///    `py_ct = iy * ct_spacing_xyz[1]`,
///    `pz_ct = iz * ct_spacing_xyz[2]`.
/// 2. Apply the rigid transform: `p_mri = R · p_ct + t`.
/// 3. Convert to MRI fractional voxel coords:
///    `vx = p_mri[0] / mri_spacing_xyz[0]`, etc.
/// 4. Trilinearly sample the MRI volume at `(vx, vy, vz)`.
///
/// Returns a flat `Vec<f32>` in ZYX order with `nz * ny * nx` elements.
fn resample_mri_into_ct_space(
    _ct_data: &[f32],
    ct_shape: [usize; 3],
    ct_spacing_xyz: [f64; 3],
    mri_data: &[f32],
    mri_shape: [usize; 3],
    mri_spacing_xyz: [f64; 3],
    r_ct_to_mri: &[f64; 9],
    t_ct_to_mri: &[f64; 3],
) -> Vec<f32> {
    let [nz, ny, nx] = ct_shape;
    let mut out = vec![0.0_f32; nz * ny * nx];
    for iz in 0..nz {
        for iy in 0..ny {
            for ix in 0..nx {
                // Physical coordinates in CT space (x = col, y = row, z = slice).
                let px_ct = ix as f64 * ct_spacing_xyz[0];
                let py_ct = iy as f64 * ct_spacing_xyz[1];
                let pz_ct = iz as f64 * ct_spacing_xyz[2];

                // Map to MRI physical coords via the rigid transform.
                let p_mri = apply_rigid(r_ct_to_mri, t_ct_to_mri, &[px_ct, py_ct, pz_ct]);

                // Convert to MRI fractional voxel coords (x = col, y = row, z = slice).
                let vx = p_mri[0] / mri_spacing_xyz[0];
                let vy = p_mri[1] / mri_spacing_xyz[1];
                let vz = p_mri[2] / mri_spacing_xyz[2];

                out[iz * ny * nx + iy * nx + ix] =
                    trilinear_sample(mri_data, mri_shape, vx, vy, vz);
            }
        }
    }
    out
}

// ── Section 5: Group 1 — Pure math tests ─────────────────────────────────────

/// # Specification
///
/// The ground-truth rotation must be a proper orthogonal matrix (rotation group
/// SO(3)): `R · R^T = I` and `det(R) = +1`.
///
/// Verification:
/// - Each diagonal element of `R · R^T` must equal `1.0` to within `1e-9`.
/// - Each off-diagonal element of `R · R^T` must equal `0.0` to within `1e-9`.
/// - `det(R)` must equal `+1.0` to within `1e-9`.
///
/// A failure here would indicate a transcription error in `GT_ROT` and would
/// invalidate all downstream tests that rely on the ground-truth transform.
#[test]
fn test_rire_gt_rotation_matrix_is_proper_orthogonal() {
    let rrt = mat3_mul(&GT_ROT, &mat3_transpose(&GT_ROT));

    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let actual = rrt[i * 3 + j];
            assert!(
                (actual - expected).abs() < 1e-9,
                "R·R^T[{},{}] = {:.12e}, expected {:.1} (tol 1e-9)",
                i,
                j,
                actual,
                expected
            );
        }
    }

    let det = mat3_det(&GT_ROT);
    assert!(
        (det - 1.0).abs() < 1e-9,
        "det(GT_ROT) = {:.12e}, expected +1.0 (tol 1e-9)",
        det
    );
}

/// # Specification
///
/// The GT transform must reproduce all 8 RIRE `ct_T1.standard` corners with
/// residual < 0.001 mm (fiducial-based gold standard).
///
/// Each source point `(src_x, src_y, src_z)` transformed by `T(p) = R·p + t`
/// must land within 0.001 mm of the corresponding destination point
/// `(dst_x, dst_y, dst_z)` from `ct_T1.standard`.
///
/// The ground-truth parameters in `GT_ROT` / `GT_TRANS` are documented to
/// reproduce all 8 corners with a maximum residual of 0.000176 mm, well below
/// the RIRE acceptance threshold of 0.01 mm.
#[test]
fn test_rire_gt_eight_corner_verification() {
    for (i, corner) in RIRE_CORNERS.iter().enumerate() {
        let [src_x, src_y, src_z, dst_x, dst_y, dst_z] = *corner;
        let result = apply_rigid(&GT_ROT, &GT_TRANS, &[src_x, src_y, src_z]);

        let dx = result[0] - dst_x;
        let dy = result[1] - dst_y;
        let dz = result[2] - dst_z;
        let residual = (dx * dx + dy * dy + dz * dz).sqrt();

        assert!(
            residual < 0.001,
            "Corner {}: residual = {:.9} mm (>= 0.001 mm)\n\
             src=({:.4}, {:.4}, {:.4})\n\
             got=({:.6}, {:.6}, {:.6})\n\
             exp=({:.4}, {:.4}, {:.4})",
            i + 1,
            residual,
            src_x,
            src_y,
            src_z,
            result[0],
            result[1],
            result[2],
            dst_x,
            dst_y,
            dst_z
        );
    }
}

/// # Specification
///
/// The inverse rigid transform `T^{-1}(T(p)) = p` must hold to within
/// 1e-6 mm at all probe points.
///
/// Validates the `rigid_inverse()` formula:
/// - `R^{-1} = R^T`
/// - `t^{-1} = −R^T · t`
///
/// The 12 non-trivial probe points are distributed throughout the CT physical
/// volume (x, y ∈ [0, 334] mm; z ∈ [0, 112] mm).
///
/// ## Tolerance rationale
///
/// `GT_ROT` is tabulated to 9 decimal places, so `R^T` is not the bit-exact
/// inverse of `R`.  The measured orthogonality residual `‖R·Rᵀ − I‖ ≈ 1e-9`
/// combined with `|p| ≈ 50–334 mm` yields a roundtrip error of order
/// `|p| × 1e-9 ≈ 1e-7 mm`.  The tolerance 1e-6 mm is far below any
/// physically meaningful threshold (sub-nanometer) while comfortably
/// accommodating finite-precision tabulation.
#[test]
fn test_rire_gt_inverse_roundtrip_exact() {
    let probe_points: [[f64; 3]; 12] = [
        [50.0, 50.0, 20.0],
        [100.0, 200.0, 40.0],
        [333.0, 0.0, 0.0],
        [0.0, 333.0, 112.0],
        [166.0, 166.0, 56.0],
        [10.0, 300.0, 80.0],
        [250.0, 100.0, 10.0],
        [330.0, 330.0, 100.0],
        [0.0, 0.0, 112.0],
        [333.9870, 0.0, 0.0],
        [0.0, 333.9870, 0.0],
        [333.9870, 333.9870, 112.0],
    ];

    let (r_inv, t_inv) = rigid_inverse(&GT_ROT, &GT_TRANS);

    for (i, p) in probe_points.iter().enumerate() {
        let p_fwd = apply_rigid(&GT_ROT, &GT_TRANS, p);
        let p_back = apply_rigid(&r_inv, &t_inv, &p_fwd);

        let dx = p_back[0] - p[0];
        let dy = p_back[1] - p[1];
        let dz = p_back[2] - p[2];
        let deviation = (dx * dx + dy * dy + dz * dz).sqrt();

        assert!(
            deviation < 1e-6,
            "Probe {}: T^{{-1}}(T(p)) roundtrip deviation = {:.3e} mm (>= 1e-6)\n\
             p        = ({:.6}, {:.6}, {:.6})\n\
             T(p)     = ({:.6}, {:.6}, {:.6})\n\
             T^{{-1}} = ({:.6}, {:.6}, {:.6})",
            i,
            deviation,
            p[0],
            p[1],
            p[2],
            p_fwd[0],
            p_fwd[1],
            p_fwd[2],
            p_back[0],
            p_back[1],
            p_back[2]
        );
    }
}

/// # Specification
///
/// Any rigid perturbation composed with its inverse must be the identity to
/// machine precision.  This validates the `rigid_inverse()` function for
/// transforms other than the ground truth.
///
/// The perturbation used here:
/// - Rotation: 0.05 rad about the z-axis.
/// - Translation: [3.5, -7.2, 2.1] mm.
///
/// For 10 probe points uniformly distributed in the CT physical volume, the
/// roundtrip `P^{-1}(P(p))` must recover `p` to within 1e-9 mm.
#[test]
fn test_rire_perturbation_and_inverse_math_roundtrip() {
    // 0.05 rad rotation about z-axis (row-major):
    // [ cos(a), -sin(a), 0 ]
    // [ sin(a),  cos(a), 0 ]
    // [   0,       0,    1 ]
    let a: f64 = 0.05;
    let rot_z: [f64; 9] = [a.cos(), -a.sin(), 0.0, a.sin(), a.cos(), 0.0, 0.0, 0.0, 1.0];
    let trans: [f64; 3] = [3.5, -7.2, 2.1];

    let (r_inv, t_inv) = rigid_inverse(&rot_z, &trans);

    // Verify the perturbation rotation is itself proper orthogonal.
    let det = mat3_det(&rot_z);
    assert!(
        (det - 1.0).abs() < 1e-12,
        "Perturbation rotation det = {:.12e}, expected +1.0",
        det
    );

    let probe_points: [[f64; 3]; 10] = [
        [0.0, 0.0, 0.0],
        [100.0, 50.0, 30.0],
        [200.0, 150.0, 60.0],
        [300.0, 250.0, 90.0],
        [50.0, 300.0, 10.0],
        [250.0, 10.0, 100.0],
        [150.0, 150.0, 50.0],
        [333.9870, 333.9870, 112.0],
        [10.0, 10.0, 5.0],
        [320.0, 280.0, 95.0],
    ];

    for (i, p) in probe_points.iter().enumerate() {
        let p_fwd = apply_rigid(&rot_z, &trans, p);
        let p_back = apply_rigid(&r_inv, &t_inv, &p_fwd);

        let dx = p_back[0] - p[0];
        let dy = p_back[1] - p[1];
        let dz = p_back[2] - p[2];
        let deviation = (dx * dx + dy * dy + dz * dz).sqrt();

        assert!(
            deviation < 1e-9,
            "Perturbation probe {}: roundtrip deviation = {:.3e} mm (>= 1e-9)\n\
             p    = ({:.6}, {:.6}, {:.6})\n\
             P(p) = ({:.6}, {:.6}, {:.6})\n\
             back = ({:.6}, {:.6}, {:.6})",
            i,
            deviation,
            p[0],
            p[1],
            p[2],
            p_fwd[0],
            p_fwd[1],
            p_fwd[2],
            p_back[0],
            p_back[1],
            p_back[2]
        );
    }

    // Suppress unused import warning for PI (used in documentation context).
    let _ = PI;
}

// ── Section 6: Group 2 — Image integration tests ─────────────────────────────

/// # Specification
///
/// CT `.mha` must load with correct shape [29, 512, 512] and spacing
/// (0.6536, 0.6536, 4.0 mm).  Intensity range must span at least from air
/// (-1000 HU) to bone cortex (+500 HU).
///
/// Assertions:
/// - `shape == [29, 512, 512]`
/// - `|spacing()[0] - 4.0| < 0.01`      (z / slice spacing)
/// - `|spacing()[1] - 0.653595| < 1e-4` (y / row spacing)
/// - `|spacing()[2] - 0.653595| < 1e-4` (x / col spacing)
/// - `min(data) <= -1000.0`
/// - `max(data) >=  500.0`
///
/// # Reference
/// RIRE training_001_ct.mha — 16-bit signed integer, HU range [-1024, 1969].
#[test]
#[ignore = "requires test_data/registration/rire/"]
fn test_rire_mha_load_ct_metadata() {
    let Some(rire_dir) = find_rire_dir() else {
        eprintln!(
            "RIRE data directory not found; \
             skipping test_rire_mha_load_ct_metadata"
        );
        return;
    };

    let ct_path = rire_dir.join("training_001_ct.mha");
    if !ct_path.exists() {
        eprintln!(
            "CT file {} not found; skipping test_rire_mha_load_ct_metadata",
            ct_path.display()
        );
        return;
    }

    let device = Default::default();
    let image = read_metaimage::<B, _>(&ct_path, &device).expect("CT .mha must load without error");

    // ── Shape [nz, ny, nx] ────────────────────────────────────────────────
    let shape = image.shape();
    assert_eq!(
        shape,
        [29, 512, 512],
        "CT shape must be [29, 512, 512], got {:?}",
        shape
    );

    // ── Spacing: spacing()[0]=z, [1]=y, [2]=x ─────────────────────────────
    let sz = image.spacing()[0]; // z / slice
    let sy = image.spacing()[1]; // y / row
    let sx = image.spacing()[2]; // x / col

    assert!(
        (sz - 4.0).abs() < 0.01,
        "CT z spacing must be ≈ 4.0 mm (±0.01), got {:.6}",
        sz
    );
    assert!(
        (sy - 0.653595).abs() < 1e-4,
        "CT y spacing must be ≈ 0.653595 mm (±1e-4), got {:.6}",
        sy
    );
    assert!(
        (sx - 0.653595).abs() < 1e-4,
        "CT x spacing must be ≈ 0.653595 mm (±1e-4), got {:.6}",
        sx
    );

    // ── Intensity range ───────────────────────────────────────────────────
    let voxels = image.data_vec();
    let vmin = voxels.iter().cloned().fold(f32::INFINITY, f32::min);
    let vmax = voxels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        vmin <= -1000.0,
        "CT min intensity must be <= -1000 HU (air), got {:.2}",
        vmin
    );
    assert!(
        vmax >= 500.0,
        "CT max intensity must be >= +500 HU (bone cortex), got {:.2}",
        vmax
    );

    eprintln!(
        "CT loaded: shape={:?}, spacing=[{:.4},{:.4},{:.4}] mm, range=[{:.1},{:.1}] HU",
        shape, sz, sy, sx, vmin, vmax
    );
}

/// # Specification
///
/// MRI T1 `.mha` must load with correct shape [26, 256, 256] and spacing
/// (1.25, 1.25, 4.0 mm).  Positive-only signal intensities confirm correct
/// unsigned-short decoding.
///
/// Assertions:
/// - `shape == [26, 256, 256]`
/// - `|spacing()[0] - 4.0| < 0.01`  (z / slice spacing)
/// - `|spacing()[1] - 1.25| < 1e-4` (y / row spacing)
/// - `|spacing()[2] - 1.25| < 1e-4` (x / col spacing)
/// - `min(data) >= 0.0`
/// - `max(data) >= 100.0`
///
/// # Reference
/// RIRE training_001_mr_T1.mha — 16-bit signed integer, SI range [2, 1626].
#[test]
#[ignore = "requires test_data/registration/rire/"]
fn test_rire_mha_load_mri_t1_metadata() {
    let Some(rire_dir) = find_rire_dir() else {
        eprintln!(
            "RIRE data directory not found; \
             skipping test_rire_mha_load_mri_t1_metadata"
        );
        return;
    };

    let mri_path = rire_dir.join("training_001_mr_T1.mha");
    if !mri_path.exists() {
        eprintln!(
            "MRI T1 file {} not found; skipping test_rire_mha_load_mri_t1_metadata",
            mri_path.display()
        );
        return;
    }

    let device = Default::default();
    let image =
        read_metaimage::<B, _>(&mri_path, &device).expect("MRI T1 .mha must load without error");

    // ── Shape [nz, ny, nx] ────────────────────────────────────────────────
    let shape = image.shape();
    assert_eq!(
        shape,
        [26, 256, 256],
        "MRI T1 shape must be [26, 256, 256], got {:?}",
        shape
    );

    // ── Spacing: spacing()[0]=z, [1]=y, [2]=x ─────────────────────────────
    let sz = image.spacing()[0]; // z / slice
    let sy = image.spacing()[1]; // y / row
    let sx = image.spacing()[2]; // x / col

    assert!(
        (sz - 4.0).abs() < 0.01,
        "MRI T1 z spacing must be ≈ 4.0 mm (±0.01), got {:.6}",
        sz
    );
    assert!(
        (sy - 1.25).abs() < 1e-4,
        "MRI T1 y spacing must be ≈ 1.25 mm (±1e-4), got {:.6}",
        sy
    );
    assert!(
        (sx - 1.25).abs() < 1e-4,
        "MRI T1 x spacing must be ≈ 1.25 mm (±1e-4), got {:.6}",
        sx
    );

    // ── Intensity range ───────────────────────────────────────────────────
    let voxels = image.data_vec();
    let vmin = voxels.iter().cloned().fold(f32::INFINITY, f32::min);
    let vmax = voxels.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    assert!(
        vmin >= 0.0,
        "MRI T1 min intensity must be >= 0.0 (positive-only signal), got {:.2}",
        vmin
    );
    assert!(
        vmax >= 100.0,
        "MRI T1 max intensity must be >= 100.0 (non-trivial signal), got {:.2}",
        vmax
    );

    eprintln!(
        "MRI T1 loaded: shape={:?}, spacing=[{:.4},{:.4},{:.4}] mm, range=[{:.1},{:.1}]",
        shape, sz, sy, sx, vmin, vmax
    );
}

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

    let ct_data = ct_img.data_vec();
    let mri_data = mri_img.data_vec();

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

    let ct_data = ct_img.data_vec();
    let mri_data = mri_img.data_vec();

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
    // perturbed[iz, iy, ix] = aligned[iz, iy, ix - 5]  if ix >= 5, else 0.0
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
    // recovered[iz, iy, ix] = perturbed[iz, iy, ix + 5]  if ix + 5 < nx, else 0.0
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
         got |Δ| = {:.6}.  This validates T ∘ T^{{-1}} ≈ identity to within \
         boundary effects ({:.2}% padded voxels).",
        ncc_recovered,
        ncc_aligned,
        (ncc_recovered - ncc_aligned).abs(),
        boundary_fraction * 100.0
    );
}
