//! RT Dose Grid slice projection for viewport overlay rendering.
//!
//! Projects an [`RtDoseGrid`] onto a volume MPR slice by mapping each volume
//! voxel to patient space and then into dose grid coordinates via the inverse
//! dose affine transform.
//!
//! # Coordinate systems
//!
//! DICOM ImageOrientationPatient stores two direction cosines:
//! - `F[0..3]` = row direction (direction of increasing column index)
//! - `F[3..6]` = column direction (direction of increasing row index)
//!
//! The dose grid patient-space affine for voxel `(frame, row, col)`:
//! ```text
//! P = origin + col * col_spacing * F[0..3]
//!            + row * row_spacing * F[3..6]
//!            + frame_offset[frame] * normal_dir
//! ```
//! where `normal_dir = F[0..3] Ã— F[3..6]` (unit vector).
//!
//! # Algorithm
//!
//! For each pixel `(row, col)` of the volume slice at `axis/slice_index`:
//! 1. Compute the patient-space position using the volume's `origin`, `direction`,
//!    and `spacing`.
//! 2. Map the patient-space position into dose grid voxel coordinates using the
//!    inverse of the dose affine (nearest-neighbor lookup).
//! 3. If the voxel falls within the dose grid extents, read the dose in Gy.
//! 4. Return `f32::NAN` for voxels outside the dose grid.
//!
//! # Result layout
//!
//! The returned `Vec<f32>` is flat row-major with length `slice_rows * slice_cols`.
//! Values are dose in Gy (`â‰¥ 0.0`) or `f32::NAN` (outside dose extent).

use ritk_io::RtDoseGrid;

/// Project a dose grid onto one MPR slice of the reference volume.
///
/// Returns a row-major flat dose map (`slice_rows Ã— slice_cols`) with values in
/// Gy (`â‰¥ 0.0`) or `f32::NAN` for voxels outside the dose grid spatial extent.
///
/// # Arguments
/// - `rt_dose` â€” the loaded RT Dose grid.
/// - `axis` â€” MPR axis: 0 = axial, 1 = coronal, 2 = sagittal.
/// - `slice_index` â€” slice index along `axis`.
/// - `vol_shape` â€” volume shape `[depth, rows, cols]`.
/// - `vol_origin` â€” volume image origin `[x, y, z]` in mm (LPS patient space).
/// - `vol_direction` â€” volume direction cosine matrix, row-major 3Ã—3.
/// - `vol_spacing` â€” volume voxel spacing `[dz, dy, dx]` in mm.
///
/// Returns `None` when the RT dose grid lacks spatial metadata (origin,
/// orientation, or pixel spacing) required for the affine transform.
pub fn extract_dose_slice_for_volume(
    rt_dose: &RtDoseGrid,
    axis: usize,
    slice_index: usize,
    vol_shape: [usize; 3],
    vol_origin: [f64; 3],
    vol_direction: [f64; 9],
    vol_spacing: [f64; 3],
) -> Option<Vec<f32>> {
    let dose_origin = rt_dose.image_position?;
    let dose_orient = rt_dose.image_orientation?;
    let dose_spacing = rt_dose.pixel_spacing?;

    // Dose grid column direction (DICOM F[0..3]).
    let dc = [dose_orient[0], dose_orient[1], dose_orient[2]];
    // Dose grid row direction (DICOM F[3..6]).
    let dr = [dose_orient[3], dose_orient[4], dose_orient[5]];
    // Dose grid normal = dc Ã— dr (unit vector).
    let dn = cross3(dc, dr);

    // Dose affine columns (physical mm per unit dose-grid step).
    let step_col = [
        dc[0] * dose_spacing[1],
        dc[1] * dose_spacing[1],
        dc[2] * dose_spacing[1],
    ];
    let step_row = [
        dr[0] * dose_spacing[0],
        dr[1] * dose_spacing[0],
        dr[2] * dose_spacing[0],
    ];

    // Slice output dimensions based on axis.
    let [vol_depth, vol_rows, vol_cols] = vol_shape;
    let (slice_rows, slice_cols, fixed_idx) = match axis {
        0 => (
            vol_rows,
            vol_cols,
            slice_index.min(vol_depth.saturating_sub(1)),
        ),
        1 => (
            vol_depth,
            vol_cols,
            slice_index.min(vol_rows.saturating_sub(1)),
        ),
        _ => (
            vol_depth,
            vol_rows,
            slice_index.min(vol_cols.saturating_sub(1)),
        ) };

    let dose_depth = rt_dose.n_frames;
    let dose_rows = rt_dose.rows;
    let dose_cols = rt_dose.cols;

    // Build precomputed frame offset â†’ normal direction mm table.
    // For each frame f, the z-displacement from dose_origin is frame_offsets[f].
    // Dose voxel patient position:
    //   P = dose_origin + col*step_col + row*step_row + frame_offsets[f]*dn
    // Inverse: given patient point P, solve for (frame, row, col).
    //
    // Build 3Ã—3 inverse of [step_col | step_row | dn*1mm] using analytic 3Ã—3 inverse.
    // Columns of A: [step_col, step_row, dn].
    let a = [
        step_col[0],
        step_row[0],
        dn[0],
        step_col[1],
        step_row[1],
        dn[1],
        step_col[2],
        step_row[2],
        dn[2],
    ];
    let inv_a = invert3x3(a)?;

    let mut dose_map = vec![f32::NAN; slice_rows * slice_cols];

    for sr in 0..slice_rows {
        for sc in 0..slice_cols {
            // Compute volume voxel coordinates (vd, vr, vc) for this pixel.
            let (vd, vr, vc): (usize, usize, usize) = match axis {
                0 => (fixed_idx, sr, sc),
                1 => (sr, fixed_idx, sc),
                _ => (sr, sc, fixed_idx) };

            // Volume voxel â†’ patient space.
            // P = vol_origin + vc*vol_direction[0..3]*dx + vr*vol_direction[3..6]*dy + vd*vol_direction[6..9]*dz
            let dx = vol_spacing[2];
            let dy = vol_spacing[1];
            let dz = vol_spacing[0];
            let vd_f = vd as f64;
            let vr_f = vr as f64;
            let vc_f = vc as f64;
            let px = vol_origin[0]
                + vc_f * vol_direction[0] * dx
                + vr_f * vol_direction[3] * dy
                + vd_f * vol_direction[6] * dz;
            let py = vol_origin[1]
                + vc_f * vol_direction[1] * dx
                + vr_f * vol_direction[4] * dy
                + vd_f * vol_direction[7] * dz;
            let pz = vol_origin[2]
                + vc_f * vol_direction[2] * dx
                + vr_f * vol_direction[5] * dy
                + vd_f * vol_direction[8] * dz;

            // Patient space â†’ dose grid coordinates.
            let dp = [
                px - dose_origin[0],
                py - dose_origin[1],
                pz - dose_origin[2],
            ];
            let dose_col_f = inv_a[0] * dp[0] + inv_a[1] * dp[1] + inv_a[2] * dp[2];
            let dose_row_f = inv_a[3] * dp[0] + inv_a[4] * dp[1] + inv_a[5] * dp[2];
            let dose_z_mm = inv_a[6] * dp[0] + inv_a[7] * dp[1] + inv_a[8] * dp[2];

            // Nearest-neighbor in the dose column and row dimensions.
            let dose_col = dose_col_f.round() as isize;
            let dose_row = dose_row_f.round() as isize;
            if dose_col < 0
                || dose_row < 0
                || dose_col >= dose_cols as isize
                || dose_row >= dose_rows as isize
            {
                continue;
            }
            let dose_col = dose_col as usize;
            let dose_row = dose_row as usize;

            // Find closest dose frame by matching frame_offsets[f] â‰ˆ dose_z_mm.
            let closest_frame = rt_dose
                .frame_offsets
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (*a - dose_z_mm)
                        .abs()
                        .partial_cmp(&(*b - dose_z_mm).abs())
                        .unwrap()
                })
                .map(|(i, _)| i)?;

            // Require the nearest frame to be within half the slice spacing.
            let frame_spacing = if rt_dose.frame_offsets.len() >= 2 {
                (rt_dose.frame_offsets[1] - rt_dose.frame_offsets[0]).abs()
            } else {
                1.0
            };
            let nearest_offset = rt_dose.frame_offsets[closest_frame];
            if (nearest_offset - dose_z_mm).abs() > frame_spacing * 0.5 + 1e-3 {
                continue;
            }

            if closest_frame >= dose_depth {
                continue;
            }

            let idx = closest_frame * dose_rows * dose_cols + dose_row * dose_cols + dose_col;
            if idx < rt_dose.dose_gy.len() {
                dose_map[sr * slice_cols + sc] = rt_dose.dose_gy[idx] as f32;
            }
        }
    }

    Some(dose_map)
}

/// Map a dose value (Gy) to an RGBA color using a hot colormap.
///
/// Dose is mapped linearly from `[min_dose, max_dose]` onto the spectrum:
/// - 0%: transparent (alpha=0)
/// - 10â€“30%: blue
/// - 30â€“60%: green
/// - 60â€“85%: yellow
/// - 85â€“100%: red/white
///
/// Returns `[r, g, b, a]` where `a = 0` for zero-dose voxels or NaN.
pub fn dose_to_rgba(dose_gy: f32, min_dose: f32, max_dose: f32, opacity: f32) -> [u8; 4] {
    if dose_gy.is_nan() || dose_gy <= 0.0 || max_dose <= min_dose {
        return [0, 0, 0, 0];
    }
    let t = ((dose_gy - min_dose) / (max_dose - min_dose)).clamp(0.0, 1.0);
    // Thresholds are dose isodose line boundaries (5 segments).
    let (r, g, b) = if t < 0.2 {
        // 0..20%: blue (dark â†’ bright blue)
        let s = t / 0.2;
        (0.0_f32, 0.0, 0.4 + 0.6 * s)
    } else if t < 0.4 {
        // 20..40%: cyan
        let s = (t - 0.2) / 0.2;
        (0.0, s, 1.0)
    } else if t < 0.6 {
        // 40..60%: green
        let s = (t - 0.4) / 0.2;
        (0.0, 1.0, 1.0 - s)
    } else if t < 0.8 {
        // 60..80%: yellow
        let s = (t - 0.6) / 0.2;
        (s, 1.0, 0.0)
    } else {
        // 80..100%: red
        let s = (t - 0.8) / 0.2;
        (1.0, 1.0 - s, 0.0)
    };
    let alpha = (opacity.clamp(0.0, 1.0) * 200.0) as u8; // semi-transparent
    [
        (r * 255.0) as u8,
        (g * 255.0) as u8,
        (b * 255.0) as u8,
        alpha,
    ]
}

// â”€â”€ Internal geometry helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Analytic 3Ã—3 matrix inverse (row-major).
///
/// Returns `None` when the determinant is < 1e-12 (singular matrix).
fn invert3x3(m: [f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6]);
    if det.abs() < 1e-12 {
        return None;
    }
    let inv_det = 1.0 / det;
    Some([
        (m[4] * m[8] - m[5] * m[7]) * inv_det,
        (m[2] * m[7] - m[1] * m[8]) * inv_det,
        (m[1] * m[5] - m[2] * m[4]) * inv_det,
        (m[5] * m[6] - m[3] * m[8]) * inv_det,
        (m[0] * m[8] - m[2] * m[6]) * inv_det,
        (m[2] * m[3] - m[0] * m[5]) * inv_det,
        (m[3] * m[7] - m[4] * m[6]) * inv_det,
        (m[1] * m[6] - m[0] * m[7]) * inv_det,
        (m[0] * m[4] - m[1] * m[3]) * inv_det,
    ])
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod tests;
