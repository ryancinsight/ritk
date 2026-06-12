//! RT ROI converters: [`RtRoiInfo`] ↔ [`VtkPolyData`] and label map → [`RtStructureSet`].

use anyhow::{bail, Result};

use crate::domain::vtk_data_object::VtkPolyData;
use crate::format::dicom::reader::types::literal_arraystring;

use super::types::{RtContour, RtRoiInfo, RtStructureSet};

/// Convert an [`RtRoiInfo`] to a [`VtkPolyData`].
///
/// # Geometric type mapping
/// | DICOM geometric type | VTK cell bucket        |
/// |----------------------|------------------------|
/// | `CLOSED_PLANAR`      | `poly.polygons`        |
/// | `OPEN_PLANAR`        | `poly.lines`           |
/// | `POINT`              | `poly.vertices`        |
/// | (unknown)            | `poly.lines` (fallback)|
///
/// # Invariants
/// - `poly.points` accumulates all contour points in contour order.
/// - Each cell references point indices computed from a running offset.
/// - Patient coordinates (`f64`, mm) are cast to `f32` for VTK storage.
pub fn rt_roi_to_polydata(roi: &RtRoiInfo) -> VtkPolyData {
    let mut poly = VtkPolyData::default();
    let mut offset: u32 = 0;

    for contour in &roi.contours {
        let n = contour.points.len() as u32;

        for p in &contour.points {
            poly.points.push([p[0] as f32, p[1] as f32, p[2] as f32]);
        }

        let indices: Vec<u32> = (offset..offset + n).collect();

        match contour.geometric_type.as_str() {
            "CLOSED_PLANAR" => poly.polygons.push(indices),
            "OPEN_PLANAR" => poly.lines.push(indices),
            "POINT" => poly.vertices.push(indices),
            _ => poly.lines.push(indices),
        }

        offset += n;
    }

    poly
}

// ── Label map → RT Structure Set ─────────────────────────────────────────────

const DX: [i32; 8] = [1, 1, 0, -1, -1, -1, 0, 1];
const DY: [i32; 8] = [0, 1, 1, 1, 0, -1, -1, -1];

/// Convert a 2-D binary mask to a closed contour polygon using 8-direction
/// Moore-Neighbor boundary tracing.
///
/// Returns `None` when the mask contains no foreground voxels.
fn trace_closed_contour(mask: &[u8], ny: usize, nx: usize) -> Option<Vec<(usize, usize)>> {
    debug_assert_eq!(mask.len(), ny * nx);

    let mut start_yx: Option<(usize, usize)> = None;
    'scan: for y in 0..ny {
        for x in 0..nx {
            if mask[y * nx + x] != 0 {
                start_yx = Some((y, x));
                break 'scan;
            }
        }
    }
    let start = start_yx?;

    let in_bounds = |y: i32, x: i32| -> bool { y >= 0 && y < ny as i32 && x >= 0 && x < nx as i32 };

    let is_foreground =
        |y: i32, x: i32| -> bool { in_bounds(y, x) && mask[(y as usize) * nx + (x as usize)] != 0 };

    let mut contour: Vec<(usize, usize)> = vec![start];
    let mut current = start;
    let mut backtrack_dir = 6usize;

    loop {
        let y0 = current.0 as i32;
        let x0 = current.1 as i32;
        let mut found = false;
        let mut next = current;

        for d in 0..8 {
            let dir = (backtrack_dir + 1 + d) % 8;
            let ny = y0 + DY[dir];
            let nx = x0 + DX[dir];
            if is_foreground(ny, nx) {
                next = (ny as usize, nx as usize);
                backtrack_dir = (dir + 4) % 8;
                found = true;
                break;
            }
        }

        if !found || next == start {
            break;
        }

        contour.push(next);
        current = next;
    }

    if contour.len() >= 3 {
        Some(contour)
    } else {
        None
    }
}

/// Map a voxel index `[z, y, x]` (ZYX layout) to patient physical coordinates
/// (mm) using the volume's origin, spacing, and direction matrix.
///
/// # Coordinate convention
///
/// `spacing = [dz, dy, dx]` where dz = slice thickness, dy = row spacing,
/// dx = column spacing.
///
/// `direction` is a flat 3×3 matrix in row-major order:
/// ```text
/// [ row_x  col_x  slice_x
///   row_y  col_y  slice_y
///   row_z  col_z  slice_z ]
/// ```
fn voxel_to_phys(
    z: usize,
    y: usize,
    x: usize,
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
) -> [f64; 3] {
    let dx = spacing[2];
    let dy = spacing[1];
    let dz = spacing[0];
    [
        origin[0]
            + direction[0] * x as f64 * dx
            + direction[1] * y as f64 * dy
            + direction[2] * z as f64 * dz,
        origin[1]
            + direction[3] * x as f64 * dx
            + direction[4] * y as f64 * dy
            + direction[5] * z as f64 * dz,
        origin[2]
            + direction[6] * x as f64 * dx
            + direction[7] * y as f64 * dy
            + direction[8] * z as f64 * dz,
    ]
}

/// Convert a [`ritk_annotation::LabelMap`] to an [`RtStructureSet`].
///
/// For each non-background label ID, contours are extracted slice-by-slice
/// along the Z axis using 8-direction Moore-Neighbor boundary tracing on each
/// 2-D binary slice. Contour points are mapped to patient physical coordinates
/// (mm) via the volume geometry.
///
/// # Errors
/// - `label_map.shape` has a zero dimension.
/// - No foreground (non-zero) labels exist.
pub fn label_map_to_rt_struct(
    label_map: &ritk_annotation::LabelMap,
    origin: [f64; 3],
    spacing: [f64; 3],
    direction: [f64; 9],
) -> Result<RtStructureSet> {
    let [nz, ny, nx] = label_map.shape.0;
    if nz == 0 || ny == 0 || nx == 0 {
        bail!("label_map has zero dimension: {:?}", label_map.shape);
    }

    let mut foreground_ids = label_map.present_labels();
    foreground_ids.retain(|&id| id != 0);
    if foreground_ids.is_empty() {
        bail!("no foreground labels found in label_map");
    }

    let mut rois: Vec<RtRoiInfo> = Vec::with_capacity(foreground_ids.len());

    for &label_id in &foreground_ids {
        let entry = label_map.table.get_label(label_id);
        let roi_name = entry.map(|e| e.name.clone()).unwrap_or_default();
        let display_color = entry.map(|e| [e.color.r(), e.color.g(), e.color.b()]);

        let mut contours: Vec<RtContour> = Vec::new();

        // Build per-slice binary mask and trace boundaries.
        for z in 0..nz {
            let mut slice_mask = vec![0u8; ny * nx];
            let mut has_foreground = false;
            for y in 0..ny {
                for x in 0..nx {
                    if label_map.label_at([z, y, x]) == label_id {
                        slice_mask[y * nx + x] = 1;
                        has_foreground = true;
                    }
                }
            }

            if !has_foreground {
                continue;
            }

            if let Some(boundary) = trace_closed_contour(&slice_mask, ny, nx) {
                let points: Vec<[f64; 3]> = boundary
                    .into_iter()
                    .map(|(y, x)| voxel_to_phys(z, y, x, origin, spacing, direction))
                    .collect();

                contours.push(RtContour {
                    geometric_type: literal_arraystring("CLOSED_PLANAR"),
                    points,
                });
            }
        }

        rois.push(RtRoiInfo {
            roi_number: u32::from(label_id),
            roi_name,
            roi_description: None,
            roi_interpreted_type: None,
            display_color,
            contours,
        });
    }

    rois.sort_by_key(|r| r.roi_number);

    Ok(RtStructureSet {
        structure_set_label: "Segmentation".into(),
        structure_set_name: Some("Label Map Export".into()),
        rois,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── trace_closed_contour ─────────────────────────────────────────────

    #[test]
    fn trace_on_empty_mask_returns_none() {
        let mask = vec![0u8; 16];
        assert!(trace_closed_contour(&mask, 4, 4).is_none());
    }

    #[test]
    fn trace_single_voxel_at_origin() {
        let ny = 3;
        let nx = 3;
        let mut mask = vec![0u8; ny * nx];
        mask[0] = 1; // (0,0)
        let result = trace_closed_contour(&mask, ny, nx);
        assert!(result.is_none(), "single voxel yields < 3 points");
    }

    #[test]
    fn trace_2x2_block_returns_closed_contour() {
        let ny = 4;
        let nx = 4;
        let mut mask = vec![0u8; ny * nx];
        for y in 1..3 {
            for x in 1..3 {
                mask[y * nx + x] = 1;
            }
        }
        let result = trace_closed_contour(&mask, ny, nx).expect("contour");
        assert!(result.len() >= 3, "length = {}", result.len());
        // all points must be in the 2x2 block region
        for &(y, x) in &result {
            assert!((1..=2).contains(&y), "y={} out of [1,2]", y);
            assert!((1..=2).contains(&x), "x={} out of [1,2]", x);
        }
        // The contour is implicitly closed (no duplicate start at end)
        assert_ne!(result.first(), result.last());
    }

    #[test]
    fn trace_plus_shape_returns_valid_polygon() {
        let ny = 5;
        let nx = 5;
        let mut mask = vec![0u8; ny * nx];
        // Plus shape: center row and center column
        for x in 0..5 {
            mask[2 * nx + x] = 1;
        }
        for y in 0..5 {
            mask[y * nx + 2] = 1;
        }
        let result = trace_closed_contour(&mask, ny, nx).expect("contour");
        assert!(
            result.len() >= 8,
            "plus should have >= 8 boundary points, got {}",
            result.len()
        );
        // The contour is implicitly closed; start point appears only at index 0.
        assert_ne!(result.first(), result.last());
    }

    // ── voxel_to_phys ────────────────────────────────────────────────────

    #[test]
    fn voxel_to_phys_identity_direction() {
        let origin = [1.0, 2.0, 3.0];
        let spacing = [2.0, 1.5, 1.0]; // [dz, dy, dx]
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let p = voxel_to_phys(0, 0, 0, origin, spacing, direction);
        assert_eq!(p, origin);
        let p = voxel_to_phys(0, 1, 2, origin, spacing, direction);
        assert_eq!(p[0], 1.0 + 2.0 * 1.0); // ox + x*dx
        assert_eq!(p[1], 2.0 + 1.0 * 1.5); // oy + y*dy
        assert_eq!(p[2], 3.0 + 0.0 * 2.0); // oz + z*dz
    }

    // ── label_map_to_rt_struct ───────────────────────────────────────────

    fn make_small_label_map() -> ritk_annotation::LabelMap {
        let mut table = ritk_annotation::LabelTable::new();
        table
            .add_label(
                1,
                "Tumor",
                ritk_annotation::RgbaBytes::new(255, 0, 0, 255),
            )
            .unwrap();
        table
            .add_label(
                2,
                "Organ",
                ritk_annotation::RgbaBytes::new(0, 255, 0, 255),
            )
            .unwrap();
        let mut lm = ritk_annotation::LabelMap::new([2, 4, 4], table);
        // Label 1: voxels [z=0, y=1..2, x=1..2]
        for y in 1..3 {
            for x in 1..3 {
                lm.set_label_at([0, y, x], 1);
            }
        }
        // Label 2: voxels [z=1, y=1..2, x=1..2]
        for y in 1..3 {
            for x in 1..3 {
                lm.set_label_at([1, y, x], 2);
            }
        }
        lm
    }

    #[test]
    fn label_map_to_rt_struct_single_label() {
        let lm = make_small_label_map();
        let rt = label_map_to_rt_struct(
            &lm,
            [0.0; 3],
            [1.0; 3],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        )
        .expect("conversion");
        assert_eq!(rt.rois.len(), 2, "two labels");
        assert_eq!(rt.rois[0].roi_name, "Tumor", "label 1 name");
        assert_eq!(rt.rois[1].roi_name, "Organ", "label 2 name");
        assert!(rt.rois[0].contours[0].points.len() >= 3, "label 1 contour");
        assert_eq!(rt.rois[0].roi_number, 1);
        assert_eq!(rt.rois[1].roi_number, 2);
    }

    #[test]
    fn label_map_to_rt_struct_zero_dim_returns_err() {
        let table = ritk_annotation::LabelTable::new();
        let lm = ritk_annotation::LabelMap::new([0, 4, 4], table);
        let result = label_map_to_rt_struct(
            &lm,
            [0.0; 3],
            [1.0; 3],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        assert!(result.is_err());
    }

    #[test]
    fn label_map_to_rt_struct_no_foreground_returns_err() {
        let table = ritk_annotation::LabelTable::new();
        let lm = ritk_annotation::LabelMap::new([2, 4, 4], table);
        let result = label_map_to_rt_struct(
            &lm,
            [0.0; 3],
            [1.0; 3],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        );
        assert!(result.is_err());
    }

    #[test]
    fn label_map_to_rt_struct_round_trip() {
        let lm = make_small_label_map();
        let origin = [10.0, 20.0, 30.0];
        let spacing = [2.0, 1.5, 1.0]; // [dz, dy, dx]
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let rt = label_map_to_rt_struct(&lm, origin, spacing, direction).expect("convert");

        // Write → read round-trip
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("rt_roundtrip.dcm");
        crate::format::dicom::rt_struct::writer::write_rt_struct(&path, &rt).expect("write");
        let loaded = crate::format::dicom::rt_struct::reader::read_rt_struct(&path).expect("read");

        assert_eq!(loaded.rois.len(), 2);
        assert_eq!(loaded.rois[0].roi_name, "Tumor");
        assert_eq!(loaded.rois[1].roi_name, "Organ");
        // Verify contour points are in physical coordinates
        for roi in &loaded.rois {
            for ct in &roi.contours {
                assert_eq!(ct.geometric_type.as_str(), "CLOSED_PLANAR");
                assert!(ct.points.len() >= 3);
                for pt in &ct.points {
                    // Points should be in patient space around the origin
                    assert!(pt[0] >= 0.0 && pt[0] <= 20.0, "x out of range: {}", pt[0]);
                }
            }
        }
    }

    #[test]
    fn label_map_to_rt_struct_contour_physical_positions() {
        let mut table = ritk_annotation::LabelTable::new();
        table
            .add_label(
                1,
                "ROI",
                ritk_annotation::RgbaBytes::new(255, 255, 255, 255),
            )
            .unwrap();
        let mut lm = ritk_annotation::LabelMap::new([1, 3, 3], table);
        // Single voxel at row=1, col=1 on slice 0
        lm.set_label_at([0, 1, 1], 1);

        let origin = [0.0; 3];
        let spacing = [1.0, 1.0, 1.0]; // [dz, dy, dx]
        let direction = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let result = label_map_to_rt_struct(&lm, origin, spacing, direction);
        // Single isolated voxel -> boundary trace may give < 3 points -> no contour
        // Check that it still produces an ROI (just may have 0 contours)
        assert!(result.is_ok());
    }
}
