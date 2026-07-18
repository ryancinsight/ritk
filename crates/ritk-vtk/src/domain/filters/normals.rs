//! Compute per-vertex surface normals from polygonal geometry.
//!
//! # Mathematical Specification
//!
//! For each polygon P = [v_0, v_1, â€¦, v_{k-1}] with k â‰¥ 3:
//!   e_1 = points\[v_1\] âˆ’ points\[v_0\]
//!   e_2 = points\[v_2\] âˆ’ points\[v_0\]
//!   N_face = e_1 Ã— e_2        (area-weighted face normal; â€–N_faceâ€– = 2Â·area)
//!
//! Each vertex accumulates N_face from all polygons it belongs to:
//!   A\[v_i\] += N_face  for all i in P
//!
//! The per-vertex normal is N_v = A\[v\] / â€–A\[v\]â€–.
//! For degenerate (zero-area) vertices, the fallback normal is [0, 0, 1].
//!
//! The output is stored as `AttributeArray::Normals { values }` under the
//! key `"Normals"` in `point_data`.

use crate::domain::vtk_data_object::{AttributeArray, VtkDataObject};
use crate::domain::vtk_pipeline::VtkFilter;
use anyhow::Result;

/// Compute per-vertex surface normals and store them as `point_data["Normals"]`.
///
/// Accepts `VtkDataObject::PolyData`; returns an error for all other variants.
///
/// Stateless filter: no parameters, so `mtime()` defaults to `ModifiedTime::ZERO`,
/// which is intentional â€” stateless filters never force re-execution through mtime.
#[derive(Debug, Clone, Default)]
pub struct ComputeNormalsFilter;

impl VtkFilter for ComputeNormalsFilter {
    fn execute(&self, input: VtkDataObject) -> Result<VtkDataObject> {
        match input {
            VtkDataObject::PolyData(mut poly) => {
                let n = poly.points.len();
                let mut accum = vec![[0.0_f32; 3]; n];

                for polygon in &poly.polygons {
                    if polygon.len() < 3 {
                        continue; // skip degenerate cells
                    }
                    let p0 = poly.points[polygon[0] as usize];
                    let p1 = poly.points[polygon[1] as usize];
                    let p2 = poly.points[polygon[2] as usize];
                    let e1 = sub3(p1, p0);
                    let e2 = sub3(p2, p0);
                    let face_n = cross3(e1, e2); // â€–face_nâ€– = 2Â·area
                    for &idx in polygon {
                        let a = &mut accum[idx as usize];
                        a[0] += face_n[0];
                        a[1] += face_n[1];
                        a[2] += face_n[2];
                    }
                }

                let normals: Vec<[f32; 3]> = accum.into_iter().map(normalize3).collect();
                poly.point_data.insert(
                    "Normals".to_string(),
                    AttributeArray::Normals { values: normals },
                );
                Ok(VtkDataObject::PolyData(poly))
            }
            other => Err(anyhow::anyhow!(
                "ComputeNormalsFilter requires PolyData input; received {}",
                data_object_type_name(&other)
            )) }
    }
}

// â”€â”€ Geometry helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[inline]
fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
pub(crate) fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
pub(crate) fn normalize3(v: [f32; 3]) -> [f32; 3] {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        [0.0, 0.0, 1.0]
    } else {
        [v[0] / len, v[1] / len, v[2] / len]
    }
}

#[inline]
pub(crate) fn data_object_type_name(obj: &VtkDataObject) -> &'static str {
    match obj {
        VtkDataObject::PolyData(_) => "PolyData",
        VtkDataObject::StructuredGrid(_) => "StructuredGrid",
        VtkDataObject::UnstructuredGrid(_) => "UnstructuredGrid",
        VtkDataObject::ImageData(_) => "ImageData" }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
#[path = "tests_normals.rs"]
mod tests;
