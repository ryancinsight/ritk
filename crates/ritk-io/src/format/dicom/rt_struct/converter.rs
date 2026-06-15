//! RT ROI converters: [`RtRoiInfo`] ↔ [`VtkPolyData`] and label map → [`RtStructureSet`].

use anyhow::{bail, Result};

use crate::domain::vtk_data_object::VtkPolyData;

use super::types::{ContourGeometricType, RtContour, RtRoiInfo, RtStructureSet};

/// Convert an [`RtRoiInfo`] to a [`VtkPolyData`].
///
/// # Geometric type mapping
/// | DICOM geometric type | VTK cell bucket        |
/// |----------------------|------------------------|
/// | `CLOSED_PLANAR`      | `poly.polygons`        |
/// | `OPEN_PLANAR`        | `poly.lines`           |
/// | `POINT`              | `poly.vertices`        |
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

        match contour.geometric_type {
            ContourGeometricType::ClosedPlanar => poly.polygons.push(indices),
            ContourGeometricType::OpenPlanar => poly.lines.push(indices),
            ContourGeometricType::Point => poly.vertices.push(indices),
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
                    geometric_type: ContourGeometricType::ClosedPlanar,
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
#[path = "tests/converter.rs"]
mod tests;
