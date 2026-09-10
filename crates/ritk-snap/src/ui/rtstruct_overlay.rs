//! RT Structure Set contour projection from patient space to viewport image space.
//!
//! This module is the SSOT for converting RT-STRUCT contour points (patient mm)
//! into 2-D row/column coordinates for a selected MPR slice.

use ritk_io::{ContourGeometricType, RtStructureSet};

use crate::geometry::affine::AffineTransform;

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
    let Ok(transform) = AffineTransform::from_parts(origin, direction, spacing) else {
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
                .map(|p| transform.patient_to_voxel(*p))
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

            let closed = contour.geometric_type == ContourGeometricType::ClosedPlanar;
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

#[cfg(test)]
#[path = "tests_rtstruct_overlay.rs"]
mod tests;
