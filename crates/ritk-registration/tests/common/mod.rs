//! Shared helpers and constants for RIRE registration algorithm integration tests.
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
//! # Dead-code allowance
//!
//! This module is compiled into every integration-test binary that does
//! `mod common;`. Each binary uses a different subset of helpers, so almost
//! every item is "dead" in some binaries. The `#[allow(dead_code)]` is a
//! standard Rust idiom for shared test-utility modules.

#![allow(dead_code)]

use burn_ndarray::NdArray;
use ritk_image::burn::backend::Autodiff;

/// Backend with autodiff — required by `GlobalMiRegistration`.
pub type B = Autodiff<NdArray<f32>>;

// ── Constants ────────────────────────────────────────────────────────────────

/// Ground-truth rotation matrix R (row-major, 3×3) for the CT → MRI T1
/// transform. Derived from the ITK Euler3DTransform parameters in
/// `training_001_ct_to_mr_T1_ground_truth.tfm` using the ITK convention
/// `R = Rz(aZ) · Rx(aX) · Ry(aY)`.
///
/// Row 0: [ 0.997000003, 0.077380155, -0.001818059 ]
/// Row 1: [-0.077397855, 0.996449628, -0.033131713 ]
/// Row 2: [-0.000752132, 0.033173032, 0.999449341 ]
pub const GT_ROT: [f64; 9] = [
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
pub const GT_TRANS: [f64; 3] = [5.03685847, -17.49694636, -27.16499259];

/// RIRE 8-corner fiducial point pairs from `ct_T1.standard`.
///
/// Each row: `[src_x, src_y, src_z, dst_x, dst_y, dst_z]` (mm, RIRE [x,y,z] order).
/// `src` = CT volume corner, `dst` = corresponding MRI T1 physical position.
pub const RIRE_CORNERS: [[f64; 6]; 8] = [
    [0.0000, 0.0000, 0.0000, 5.0369, -17.4970, -27.1650],
    [333.9870, 0.0000, 0.0000, 338.0219, -43.3470, -27.4162],
    [0.0000, 333.9870, 0.0000, 30.8808, 315.3043, -16.0856],
    [333.9870, 333.9870, 0.0000, 363.8658, 289.4544, -16.3368],
    [0.0000, 0.0000, 112.0000, 4.8333, -21.2077, 84.7733],
    [333.9870, 0.0000, 112.0000, 337.8183, -47.0576, 84.5221],
    [0.0000, 333.9870, 112.0000, 30.6772, 311.5937, 95.8527],
    [333.9870, 333.9870, 112.0000, 363.6622, 285.7437, 95.6015],
];

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Search standard locations for the RIRE test data directory.
pub fn find_rire_dir() -> Option<std::path::PathBuf> {
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
pub fn ncc(a: &[f32], b: &[f32]) -> f64 {
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
/// `shape = [nz, ny, nx]`. Returns `(downsampled_data, new_shape)`.
pub fn downsample_stride(data: &[f32], shape: [usize; 3], stride: usize) -> (Vec<f32>, [usize; 3]) {
    let [nz, ny, nx] = shape;
    let new_nz = nz.div_ceil(stride);
    let new_ny = ny.div_ceil(stride);
    let new_nx = nx.div_ceil(stride);
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
pub fn normalize_minmax(data: &[f32]) -> Vec<f32> {
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
pub fn trilinear_sample(vol: &[f32], shape: [usize; 3], px: f64, py: f64, pz: f64) -> f32 {
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
/// * `ct_shape` — `[nz, ny, nx]` of the (possibly downsampled) CT grid.
/// * `ct_spacing` — `[sz, sy, sx]` CT voxel spacing in mm.
/// * `mri_data` — Flat ZYX-ordered MRI intensity data.
/// * `mri_shape` — `[nz, ny, nx]` of the MRI volume.
/// * `mri_spacing`— `[sz, sy, sx]` MRI voxel spacing in mm.
/// * `m` — 4×4 row-major homogeneous transform matrix in RITK `[z,y,x]` space.
pub fn resample_mri_into_ct_ritk(
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
pub fn apply_ritk_m4_to_rire_point(m: &[f64; 16], p_rire: [f64; 3]) -> [f64; 3] {
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
/// `[x, y, z]` convention. Returns `(mean_tre_mm, max_tre_mm)`.
pub fn compute_tre(m: &[f64; 16]) -> (f64, f64) {
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
pub fn identity_m4() -> [f64; 16] {
    [
        1., 0., 0., 0., //
        0., 1., 0., 0., //
        0., 0., 1., 0., //
        0., 0., 0., 1., //
    ]
}

// ── RIRE rigid-transform math helpers ────────────────────────────────────────
// These operate in RIRE [x, y, z] convention (x = col, y = row, z = slice),
// NOT in RITK [z, y, x] convention.

/// Apply a rigid transform `T(p) = R · p + t` to a 3-D point.
///
/// `r` is a row-major 3×3 rotation matrix stored as a flat 9-element array:
/// `r[i*3 + j]` is the element at row `i`, column `j`.
pub fn apply_rigid(r: &[f64; 9], t: &[f64; 3], p: &[f64; 3]) -> [f64; 3] {
    [
        r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
        r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
        r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2],
    ]
}

/// Transpose a row-major 3×3 matrix.
///
/// Output element at `(i, j)` = input element at `(j, i)`.
pub fn mat3_transpose(m: &[f64; 9]) -> [f64; 9] {
    [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]]
}

/// Multiply two row-major 3×3 matrices: returns `a · b`.
pub fn mat3_mul(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
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
pub fn mat3_det(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Compute the inverse of a rigid (rotation + translation) transform.
///
/// For a rigid transform `T(p) = R · p + t`:
/// - `R^{-1} = R^T` (rotation matrices are orthogonal)
/// - `t^{-1} = −R^T · t`
///
/// Returns `(R^T, −R^T · t)`.
pub fn rigid_inverse(r: &[f64; 9], t: &[f64; 3]) -> ([f64; 9], [f64; 3]) {
    let r_inv = mat3_transpose(r);
    // t_inv = −R^T · t (apply R^T to t, then negate)
    let t_inv = [
        -(r_inv[0] * t[0] + r_inv[1] * t[1] + r_inv[2] * t[2]),
        -(r_inv[3] * t[0] + r_inv[4] * t[1] + r_inv[5] * t[2]),
        -(r_inv[6] * t[0] + r_inv[7] * t[1] + r_inv[8] * t[2]),
    ];
    (r_inv, t_inv)
}

/// Resample `mri_data` (a ZYX-ordered flat volume) into the coordinate frame
/// of a CT volume, producing one output sample per CT voxel.
///
/// This operates in RIRE `[x, y, z]` convention (x = col, y = row, z = slice),
/// unlike `resample_mri_into_ct_ritk` which uses RITK `[z, y, x]` order.
///
/// # Arguments
///
/// * `_ct_data` — CT voxel data (not read; present for API symmetry).
/// * `ct_shape` — CT volume shape `[nz, ny, nx]`.
/// * `ct_spacing_xyz` — CT voxel spacing in **(x/col, y/row, z/slice)** order (mm).
/// * `mri_data` — Source MRI voxel data in ZYX order.
/// * `mri_shape` — MRI volume shape `[nz, ny, nx]`.
/// * `mri_spacing_xyz` — MRI voxel spacing in **(x/col, y/row, z/slice)** order (mm).
/// * `r_ct_to_mri` — Rotation matrix mapping CT physical coords → MRI physical coords.
/// * `t_ct_to_mri` — Translation vector (mm) for the same mapping.
///
/// # Algorithm
///
/// For each CT voxel `(iz, iy, ix)`:
/// 1. Compute physical CT coords: `px_ct = ix * ct_spacing_xyz[0]`, etc.
/// 2. Apply the rigid transform: `p_mri = R · p_ct + t`.
/// 3. Convert to MRI fractional voxel coords and trilinearly sample.
#[allow(clippy::too_many_arguments)] // 8-arg test helper; flat signature reads cleanly
pub fn resample_mri_into_ct_space(
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
