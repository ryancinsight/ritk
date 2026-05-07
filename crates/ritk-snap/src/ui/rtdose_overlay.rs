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
//! where `normal_dir = F[0..3] × F[3..6]` (unit vector).
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
//! Values are dose in Gy (`≥ 0.0`) or `f32::NAN` (outside dose extent).

use ritk_io::RtDoseGrid;

/// Project a dose grid onto one MPR slice of the reference volume.
///
/// Returns a row-major flat dose map (`slice_rows × slice_cols`) with values in
/// Gy (`≥ 0.0`) or `f32::NAN` for voxels outside the dose grid spatial extent.
///
/// # Arguments
/// - `rt_dose` — the loaded RT Dose grid.
/// - `axis` — MPR axis: 0 = axial, 1 = coronal, 2 = sagittal.
/// - `slice_index` — slice index along `axis`.
/// - `vol_shape` — volume shape `[depth, rows, cols]`.
/// - `vol_origin` — volume image origin `[x, y, z]` in mm (LPS patient space).
/// - `vol_direction` — volume direction cosine matrix, row-major 3×3.
/// - `vol_spacing` — volume voxel spacing `[dz, dy, dx]` in mm.
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
    // Dose grid normal = dc × dr (unit vector).
    let dn = cross3(dc, dr);

    // Dose affine columns (physical mm per unit dose-grid step).
    let step_col = [dc[0] * dose_spacing[1], dc[1] * dose_spacing[1], dc[2] * dose_spacing[1]];
    let step_row = [dr[0] * dose_spacing[0], dr[1] * dose_spacing[0], dr[2] * dose_spacing[0]];

    // Slice output dimensions based on axis.
    let [vol_depth, vol_rows, vol_cols] = vol_shape;
    let (slice_rows, slice_cols, fixed_idx) = match axis {
        0 => (vol_rows, vol_cols, slice_index.min(vol_depth.saturating_sub(1))),
        1 => (vol_depth, vol_cols, slice_index.min(vol_rows.saturating_sub(1))),
        _ => (vol_depth, vol_rows, slice_index.min(vol_cols.saturating_sub(1))),
    };

    let dose_depth = rt_dose.n_frames;
    let dose_rows = rt_dose.rows;
    let dose_cols = rt_dose.cols;

    // Build precomputed frame offset → normal direction mm table.
    // For each frame f, the z-displacement from dose_origin is frame_offsets[f].
    // Dose voxel patient position:
    //   P = dose_origin + col*step_col + row*step_row + frame_offsets[f]*dn
    // Inverse: given patient point P, solve for (frame, row, col).
    //
    // Build 3×3 inverse of [step_col | step_row | dn*1mm] using analytic 3×3 inverse.
    // Columns of A: [step_col, step_row, dn].
    let a = [
        step_col[0], step_row[0], dn[0],
        step_col[1], step_row[1], dn[1],
        step_col[2], step_row[2], dn[2],
    ];
    let inv_a = invert3x3(a)?;

    let mut dose_map = vec![f32::NAN; slice_rows * slice_cols];

    for sr in 0..slice_rows {
        for sc in 0..slice_cols {
            // Compute volume voxel coordinates (vd, vr, vc) for this pixel.
            let (vd, vr, vc): (usize, usize, usize) = match axis {
                0 => (fixed_idx, sr, sc),
                1 => (sr, fixed_idx, sc),
                _ => (sr, sc, fixed_idx),
            };

            // Volume voxel → patient space.
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

            // Patient space → dose grid coordinates.
            let dp = [px - dose_origin[0], py - dose_origin[1], pz - dose_origin[2]];
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

            // Find closest dose frame by matching frame_offsets[f] ≈ dose_z_mm.
            let closest_frame = rt_dose
                .frame_offsets
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (*a - dose_z_mm).abs().partial_cmp(&(*b - dose_z_mm).abs()).unwrap()
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
/// - 10–30%: blue
/// - 30–60%: green
/// - 60–85%: yellow
/// - 85–100%: red/white
///
/// Returns `[r, g, b, a]` where `a = 0` for zero-dose voxels or NaN.
pub fn dose_to_rgba(dose_gy: f32, min_dose: f32, max_dose: f32, opacity: f32) -> [u8; 4] {
    if dose_gy.is_nan() || dose_gy <= 0.0 || max_dose <= min_dose {
        return [0, 0, 0, 0];
    }
    let t = ((dose_gy - min_dose) / (max_dose - min_dose)).clamp(0.0, 1.0);
    // Thresholds are dose isodose line boundaries (5 segments).
    let (r, g, b) = if t < 0.2 {
        // 0..20%: blue (dark → bright blue)
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

// ── Internal geometry helpers ─────────────────────────────────────────────────

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Analytic 3×3 matrix inverse (row-major).
///
/// Returns `None` when the determinant is < 1e-12 (singular matrix).
fn invert3x3(m: [f64; 9]) -> Option<[f64; 9]> {
    let det = m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to build a minimal RtDoseGrid for testing.
    fn make_dose_grid(
        rows: usize,
        cols: usize,
        n_frames: usize,
        dose_values: Vec<f64>,
        origin: [f64; 3],
        orient: [f64; 6],
        spacing: [f64; 2],
        frame_offsets: Vec<f64>,
    ) -> RtDoseGrid {
        RtDoseGrid {
            rows,
            cols,
            n_frames,
            dose_type: "PHYSICAL".to_owned(),
            dose_summation_type: "PLAN".to_owned(),
            dose_grid_scaling: 1.0,
            frame_offsets,
            dose_gy: dose_values,
            image_position: Some(origin),
            image_orientation: Some(orient),
            pixel_spacing: Some(spacing),
            referenced_rt_plan_sop_instance_uid: None,
        }
    }

    #[test]
    fn invert3x3_identity_returns_identity() {
        // Analytical: inv(I) = I.
        let id = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert3x3(id).expect("identity is invertible");
        for i in 0..9 {
            assert!(
                (inv[i] - id[i]).abs() < 1e-10,
                "inv[{i}] = {} ≠ {}",
                inv[i],
                id[i]
            );
        }
    }

    #[test]
    fn invert3x3_singular_returns_none() {
        // Rows are linearly dependent → det = 0.
        let m = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 1.0, 0.0, 0.0];
        assert!(invert3x3(m).is_none());
    }

    #[test]
    fn invert3x3_round_trip() {
        // A · A⁻¹ = I (Frobenius error < 1e-10).
        let m = [2.0, 1.0, 0.0, 1.5, 3.0, 0.5, 0.0, 0.5, 2.0];
        let inv = invert3x3(m).expect("matrix must be invertible");
        // Multiply m · inv and check for identity.
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0f64;
                for k in 0..3 {
                    sum += m[i * 3 + k] * inv[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-10,
                    "m·inv[{i},{j}] = {sum:.2e} ≠ {expected}"
                );
            }
        }
    }

    #[test]
    fn dose_to_rgba_nan_returns_transparent() {
        let rgba = dose_to_rgba(f32::NAN, 0.0, 60.0, 0.5);
        assert_eq!(rgba[3], 0, "NaN dose should have alpha=0");
    }

    #[test]
    fn dose_to_rgba_zero_dose_returns_transparent() {
        let rgba = dose_to_rgba(0.0, 0.0, 60.0, 0.5);
        assert_eq!(rgba[3], 0, "zero dose should have alpha=0");
    }

    #[test]
    fn dose_to_rgba_max_dose_is_red() {
        // At max dose (t=1.0), the colormap is in the red segment.
        let rgba = dose_to_rgba(60.0, 0.0, 60.0, 1.0);
        assert!(rgba[0] > 200, "red channel should be high at max dose");
        assert!(rgba[3] > 0, "alpha should be non-zero at max dose");
    }

    #[test]
    fn dose_to_rgba_midpoint_is_greenish() {
        // At t=0.5 (midpoint), the colormap yields green.
        let rgba = dose_to_rgba(30.0, 0.0, 60.0, 1.0);
        assert!(rgba[1] > 150, "green channel high at midpoint");
        assert!(rgba[3] > 0, "alpha should be non-zero at midpoint");
    }

    #[test]
    fn cross3_unit_vectors() {
        // Analytical: x × y = z, y × z = x, z × x = y.
        let x = [1.0_f64, 0.0, 0.0];
        let y = [0.0, 1.0, 0.0];
        let z = [0.0, 0.0, 1.0];
        let xy = cross3(x, y);
        assert!((xy[0] - z[0]).abs() < 1e-12 && (xy[1] - z[1]).abs() < 1e-12 && (xy[2] - z[2]).abs() < 1e-12,
            "x×y != z: {:?}", xy);
        let yz = cross3(y, z);
        assert!((yz[0] - x[0]).abs() < 1e-12 && (yz[1] - x[1]).abs() < 1e-12 && (yz[2] - x[2]).abs() < 1e-12,
            "y×z != x: {:?}", yz);
    }

    #[test]
    fn extract_dose_axial_identity_grid() {
        // Volume and dose share the same origin, direction (identity), and spacing (1mm).
        // Dose grid: 4×4×4 axial, all voxels = 2.0 Gy.
        let n = 4;
        let dose = make_dose_grid(
            n, n, n,
            vec![2.0_f64; n * n * n],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], // identity IOP
            [1.0, 1.0],                        // 1mm spacing
            vec![0.0, 1.0, 2.0, 3.0],          // frame offsets
        );

        // Volume: 4×4×4, identity direction, 1mm spacing, same origin.
        let vol_dir = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = extract_dose_slice_for_volume(
            &dose,
            0, // axial
            0, // first slice
            [n, n, n],
            [0.0, 0.0, 0.0],
            vol_dir,
            [1.0, 1.0, 1.0],
        );
        let map = result.expect("should produce a dose map");
        assert_eq!(map.len(), n * n);
        // All voxels should have dose ~2.0 Gy (within the dose grid).
        let valid: Vec<f32> = map.iter().cloned().filter(|v| v.is_finite()).collect();
        assert!(!valid.is_empty(), "expected some finite dose values");
        for &d in &valid {
            assert!(
                (d - 2.0).abs() < 1e-4,
                "expected 2.0 Gy, got {d}"
            );
        }
    }

    #[test]
    fn extract_dose_no_spatial_metadata_returns_none() {
        // RtDoseGrid without spatial metadata → None.
        let dose = RtDoseGrid {
            rows: 4,
            cols: 4,
            n_frames: 1,
            dose_type: "PHYSICAL".into(),
            dose_summation_type: "PLAN".into(),
            dose_grid_scaling: 1.0,
            frame_offsets: vec![0.0],
            dose_gy: vec![1.0; 16],
            image_position: None, // missing
            image_orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            pixel_spacing: Some([1.0, 1.0]),
            referenced_rt_plan_sop_instance_uid: None,
        };
        let result = extract_dose_slice_for_volume(
            &dose,
            0, 0, [4, 4, 4],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        );
        assert!(result.is_none(), "missing origin must return None");
    }
}
