//! RT ROI → VTK PolyData converter.

use crate::domain::vtk_data_object::VtkPolyData;

use super::types::RtRoiInfo;

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
