//! RT Structure Set contour projection from patient space to viewport image space.
//!
//! This module is the SSOT for converting RT-STRUCT contour points (patient mm)
//! into 2-D row/column coordinates for a selected MPR slice.

use ritk_io::RtStructureSet;

/// One projected RT contour ready for viewport-space raster mapping.
#[derive(Debug, Clone, PartialEq)]
pub struct ProjectedRtContour {
    /// RGB color from RT ROI display color or default fallback.
    pub color: [u8; 3],
    /// Whether the contour is geometrically closed.
    pub closed: bool,
    /// Row/column points in continuous image coordinates.
    pub points_row_col: Vec<[f32; 2]>,
}

/// Project RT-STRUCT contours onto the selected axis/slice.
///
/// Returns only contours whose points lie on the active slice plane within
/// a half-voxel tolerance.
pub fn project_rt_struct_contours_for_slice(
    rt: &RtStructureSet,
    axis: usize,
    slice_index: usize,
    shape: [usize; 3],
    origin: [f64; 3],
    direction: [f64; 9],
    spacing: [f64; 3],
) -> Vec<ProjectedRtContour> {
    let Some(inv_phys_to_voxel) = inverse_phys_to_voxel(direction, spacing) else {
        return Vec::new();
    };

    let (row_dim, col_dim) = match axis {
        0 => (shape[1] as f64, shape[2] as f64),
        1 => (shape[0] as f64, shape[2] as f64),
        _ => (shape[0] as f64, shape[1] as f64),
    };

    let mut out = Vec::new();
    for roi in &rt.rois {
        let color = roi.display_color.unwrap_or([255, 255, 0]);
        for contour in &roi.contours {
            if contour.points.is_empty() {
                continue;
            }

            let voxels: Vec<[f64; 3]> = contour
                .points
                .iter()
                .map(|p| patient_to_voxel(*p, origin, inv_phys_to_voxel))
                .collect();

            let on_slice = voxels
                .iter()
                .all(|v| (axis_coordinate(*v, axis) - slice_index as f64).abs() <= 0.5);
            if !on_slice {
                continue;
            }

            let mut points_row_col: Vec<[f32; 2]> = Vec::new();
            for voxel in voxels {
                let (row, col) = row_col_from_voxel(voxel, axis);
                if row.is_finite()
                    && col.is_finite()
                    && row >= -0.5
                    && col >= -0.5
                    && row <= row_dim - 0.5
                    && col <= col_dim - 0.5
                {
                    points_row_col.push([row as f32, col as f32]);
                }
            }

            if points_row_col.is_empty() {
                continue;
            }

            let closed = contour.geometric_type.trim() == "CLOSED_PLANAR";
            out.push(ProjectedRtContour {
                color,
                closed,
                points_row_col,
            });
        }
    }

    out
}

fn axis_coordinate(voxel: [f64; 3], axis: usize) -> f64 {
    match axis {
        0 => voxel[0],
        1 => voxel[1],
        _ => voxel[2],
    }
}

fn row_col_from_voxel(voxel: [f64; 3], axis: usize) -> (f64, f64) {
    match axis {
        0 => (voxel[1], voxel[2]),
        1 => (voxel[0], voxel[2]),
        _ => (voxel[0], voxel[1]),
    }
}

fn patient_to_voxel(point_mm: [f64; 3], origin: [f64; 3], inv_phys_to_voxel: [f64; 9]) -> [f64; 3] {
    let d = [
        point_mm[0] - origin[0],
        point_mm[1] - origin[1],
        point_mm[2] - origin[2],
    ];
    [
        inv_phys_to_voxel[0] * d[0] + inv_phys_to_voxel[1] * d[1] + inv_phys_to_voxel[2] * d[2],
        inv_phys_to_voxel[3] * d[0] + inv_phys_to_voxel[4] * d[1] + inv_phys_to_voxel[5] * d[2],
        inv_phys_to_voxel[6] * d[0] + inv_phys_to_voxel[7] * d[1] + inv_phys_to_voxel[8] * d[2],
    ]
}

fn inverse_phys_to_voxel(direction: [f64; 9], spacing: [f64; 3]) -> Option<[f64; 9]> {
    let m = [
        direction[0] * spacing[0],
        direction[1] * spacing[1],
        direction[2] * spacing[2],
        direction[3] * spacing[0],
        direction[4] * spacing[1],
        direction[5] * spacing[2],
        direction[6] * spacing[0],
        direction[7] * spacing[1],
        direction[8] * spacing[2],
    ];
    invert_3x3(m)
}

fn invert_3x3(m: [f64; 9]) -> Option<[f64; 9]> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_io::{RtContour, RtRoiInfo, RtStructureSet};

    fn make_rt() -> RtStructureSet {
        RtStructureSet {
            structure_set_label: "SS".to_string(),
            structure_set_name: Some("Name".to_string()),
            rois: vec![RtRoiInfo {
                roi_number: 1,
                roi_name: "PTV".to_string(),
                roi_description: None,
                roi_interpreted_type: Some("PTV".to_string()),
                display_color: Some([255, 0, 0]),
                contours: vec![RtContour {
                    geometric_type: "CLOSED_PLANAR".to_string(),
                    points: vec![
                        [2.0, 3.0, 4.0],
                        [2.0, 5.0, 4.0],
                        [2.0, 5.0, 6.0],
                        [2.0, 3.0, 6.0],
                    ],
                }],
            }],
        }
    }

    #[test]
    fn projects_identity_axial_contour_on_matching_slice() {
        let rt = make_rt();
        let projected = project_rt_struct_contours_for_slice(
            &rt,
            0,
            2,
            [10, 10, 10],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        );
        assert_eq!(projected.len(), 1);
        assert_eq!(projected[0].color, [255, 0, 0]);
        assert!(projected[0].closed);
        assert_eq!(projected[0].points_row_col[0], [3.0, 4.0]);
    }

    #[test]
    fn filters_contour_when_slice_does_not_match() {
        let rt = make_rt();
        let projected = project_rt_struct_contours_for_slice(
            &rt,
            0,
            7,
            [10, 10, 10],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        );
        assert!(projected.is_empty());
    }

    #[test]
    fn uses_default_color_when_roi_display_color_missing() {
        let mut rt = make_rt();
        rt.rois[0].display_color = None;
        let projected = project_rt_struct_contours_for_slice(
            &rt,
            0,
            2,
            [10, 10, 10],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        );
        assert_eq!(projected[0].color, [255, 255, 0]);
    }

    #[test]
    fn returns_empty_when_transform_is_singular() {
        let rt = make_rt();
        let projected = project_rt_struct_contours_for_slice(
            &rt,
            0,
            2,
            [10, 10, 10],
            [0.0, 0.0, 0.0],
            [0.0; 9],
            [1.0, 1.0, 1.0],
        );
        assert!(projected.is_empty());
    }
}
