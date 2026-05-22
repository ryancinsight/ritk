//! Pure math helpers for the RIRE CT/MRI T1 registration validation example.
//!
//! All functions are self-contained — no external dependencies beyond `std`.

use super::constants::RIRE_CORNERS;

// ── Coordinate / transform helpers ───────────────────────────────────────────

/// Apply a RITK-space 4×4 homogeneous matrix to a RIRE-convention point.
///
/// RIRE uses `[x, y, z]` (col, row, slice) ordering; RITK uses `[z, y, x]`.
/// Permutes in, applies the 4×4 transform, then permutes the result back.
pub fn apply_ritk_m4_to_rire_point(m: &[f64; 16], p_rire: [f64; 3]) -> [f64; 3] {
    // RIRE [x, y, z] → RITK [z, y, x]
    let (pz, py, px) = (p_rire[2], p_rire[1], p_rire[0]);
    let out_z = m[0] * pz + m[1] * py + m[2] * px + m[3];
    let out_y = m[4] * pz + m[5] * py + m[6] * px + m[7];
    let out_x = m[8] * pz + m[9] * py + m[10] * px + m[11];
    // RITK [z, y, x] → RIRE [x, y, z]
    [out_x, out_y, out_z]
}

/// Mean and max TRE (mm) over the 8 RIRE fiducial corner pairs.
///
/// `m` is the 4×4 RITK-space row-major transform matrix.
pub fn compute_tre(m: &[f64; 16]) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut max = 0.0_f64;
    for pair in &RIRE_CORNERS {
        let src = [pair[0], pair[1], pair[2]];
        let gt_dst = [pair[3], pair[4], pair[5]];
        let pred = apply_ritk_m4_to_rire_point(m, src);
        let dx = pred[0] - gt_dst[0];
        let dy = pred[1] - gt_dst[1];
        let dz = pred[2] - gt_dst[2];
        let d = (dx * dx + dy * dy + dz * dz).sqrt();
        sum += d;
        max = max.max(d);
    }
    (sum / RIRE_CORNERS.len() as f64, max)
}

/// Return the 4×4 identity homogeneous matrix (row-major, RITK [z,y,x] space).
pub fn identity_m4() -> [f64; 16] {
    [
        1., 0., 0., 0., //
        0., 1., 0., 0., //
        0., 0., 1., 0., //
        0., 0., 0., 1., //
    ]
}

/// Resample MRI into a CT grid using a RITK 4×4 homogeneous matrix.
///
/// `ct_spacing` and `mri_spacing` are `[sz, sy, sx]` (RITK [z,y,x] order).
/// `m` is the output-to-input mapping (CT world → MRI world) in RITK space.
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
                let pz = iz as f64 * ct_spacing[0];
                let py = iy as f64 * ct_spacing[1];
                let px = ix as f64 * ct_spacing[2];
                let qz = m[0] * pz + m[1] * py + m[2] * px + m[3];
                let qy = m[4] * pz + m[5] * py + m[6] * px + m[7];
                let qx = m[8] * pz + m[9] * py + m[10] * px + m[11];
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

// ── Rigid transform primitives ───────────────────────────────────────────────

/// Apply a rigid transform `T(p) = R · p + t` to a 3-D point.
///
/// `r` is a row-major 3×3 matrix stored as a flat 9-element array where
/// `r[i*3 + j]` is the element at row `i`, column `j`.
pub fn apply_rigid(r: &[f64; 9], t: &[f64; 3], p: &[f64; 3]) -> [f64; 3] {
    [
        r[0] * p[0] + r[1] * p[1] + r[2] * p[2] + t[0],
        r[3] * p[0] + r[4] * p[1] + r[5] * p[2] + t[1],
        r[6] * p[0] + r[7] * p[1] + r[8] * p[2] + t[2],
    ]
}

/// Transpose a row-major 3×3 matrix in place, returning a new array.
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

/// Cofactor-expansion determinant of a row-major 3×3 matrix.
pub fn mat3_det(m: &[f64; 9]) -> f64 {
    m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
}

/// Compute the inverse of a rigid (rotation + translation) transform.
///
/// For `T(p) = R · p + t`:
/// - `R^{-1} = R^T` (rotation matrices are orthogonal)
/// - `t^{-1} = −R^T · t`
///
/// Returns `(R^T, −R^T · t)`.
pub fn rigid_inverse(r: &[f64; 9], t: &[f64; 3]) -> ([f64; 9], [f64; 3]) {
    let r_inv = mat3_transpose(r);
    let t_inv = [
        -(r_inv[0] * t[0] + r_inv[1] * t[1] + r_inv[2] * t[2]),
        -(r_inv[3] * t[0] + r_inv[4] * t[1] + r_inv[5] * t[2]),
        -(r_inv[6] * t[0] + r_inv[7] * t[1] + r_inv[8] * t[2]),
    ];
    (r_inv, t_inv)
}

// ── Statistical / similarity measures ────────────────────────────────────────

/// Pearson normalized cross-correlation of two equal-length `f32` slices.
///
/// Returns `0.0` when either input has standard deviation below `1e-12`
/// (degenerate / constant signal).
pub fn ncc(a: &[f32], b: &[f32]) -> f64 {
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

// ── Volume processing helpers ────────────────────────────────────────────────

/// Downsample a flat ZYX-ordered volume by selecting every `stride`-th voxel
/// along each axis.
///
/// Returns `(downsampled_flat_vec, [new_nz, new_ny, new_nx])`.
pub fn downsample_stride(data: &[f32], shape: [usize; 3], stride: usize) -> (Vec<f32>, [usize; 3]) {
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

/// Minmax-normalize a volume to `[0, 1]`. Avoids division by zero when the
/// volume is constant.
pub fn normalize_minmax(data: &[f32]) -> Vec<f32> {
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let range = (max - min).max(1e-8);
    data.iter().map(|&v| (v - min) / range).collect()
}

// ── Interpolation ────────────────────────────────────────────────────────────

/// Trilinear interpolation of a ZYX-ordered volume at fractional voxel
/// coordinates `(px, py, pz)` in **(x = col, y = row, z = slice)** order.
///
/// Returns `0.0` for samples that fall outside the grid.
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

    (v000 * (1.0 - dx) * (1.0 - dy) * (1.0 - dz)
        + v001 * dx * (1.0 - dy) * (1.0 - dz)
        + v010 * (1.0 - dx) * dy * (1.0 - dz)
        + v011 * dx * dy * (1.0 - dz)
        + v100 * (1.0 - dx) * (1.0 - dy) * dz
        + v101 * dx * (1.0 - dy) * dz
        + v110 * (1.0 - dx) * dy * dz
        + v111 * dx * dy * dz) as f32
}

/// Resample `mri_data` into the coordinate frame of a CT volume using a rigid
/// transform (trilinear interpolation, pure Rust).
///
/// For each CT voxel `(iz, iy, ix)`:
/// 1. Compute physical CT coords in (x=col, y=row, z=slice) order.
/// 2. Map to MRI physical coords via `T(p) = R · p + t`.
/// 3. Convert to MRI fractional voxel coords.
/// 4. Trilinearly sample the MRI volume.
///
/// Returns a flat `Vec<f32>` in ZYX order with `nz * ny * nx` elements.
pub fn resample_mri_into_ct_space(
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
                // Convert to MRI fractional voxel coords.
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

// ── Euler decomposition ──────────────────────────────────────────────────────

/// Decompose a rotation matrix into ITK Euler3D angles (aX, aY, aZ) using the
/// convention R = Rz(aZ) · Rx(aX) · Ry(aY).
///
/// Returns `(ax_rad, ay_rad, az_rad)`.
pub fn euler3d_from_matrix(r: &[f64; 9]) -> (f64, f64, f64) {
    // From the product R = Rz * Rx * Ry the third row gives:
    // R[2,0] = -cos(aX)*sin(aY)
    // R[2,1] = sin(aX)
    // R[2,2] = cos(aX)*cos(aY)
    let ax = r[7].clamp(-1.0, 1.0).asin(); // R[2,1]
    let cos_x = ax.cos();
    let ay = if cos_x.abs() > 1e-9 {
        f64::atan2(-r[6] / cos_x, r[8] / cos_x) // R[2,0], R[2,2]
    } else {
        0.0 // gimbal lock: aY undefined, absorb into aZ
    };
    let az = if cos_x.abs() > 1e-9 {
        f64::atan2(-r[1] / cos_x, r[4] / cos_x) // R[0,1], R[1,1]
    } else {
        f64::atan2(r[3], r[0]) // fallback
    };
    (ax, ay, az)
}
