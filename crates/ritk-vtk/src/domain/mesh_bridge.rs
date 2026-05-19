//! Bidirectional bridge between [`gaia::IndexedMesh`] and [`VtkPolyData`].
//!
//! # Architecture
//!
//! [`gaia::IndexedMesh<f64>`] is the canonical watertight computation mesh:
//! vertices are spatially welded, every edge is manifold, and the surface is
//! suitable for Boolean CSG, quality analysis, and marching-cubes output.
//!
//! [`VtkPolyData`] is the canonical VTK interchange mesh: it carries arbitrary
//! polygon cells (triangles, quads, polygons), per-point and per-cell
//! attribute arrays, and maps directly to the VTK legacy `DATASET POLYDATA`
//! format consumed by ITK-SNAP, Paraview, and VTK pipelines.
//!
//! # Conversion semantics
//!
//! | Direction | Welding | Precision | Notes |
//! |-----------|---------|-----------|-------|
//! | `IndexedMesh → VtkPolyData` | Preserved | `f64 → f32` | One point per welded vertex; normals in `point_data["Normals"]` |
//! | `VtkPolyData → IndexedMesh` | Applied | `f32 → f64` | Only triangular polygons are converted; non-triangles are skipped |
//!
//! # Invariants
//!
//! **Theorem (round-trip vertex count)**: For any `IndexedMesh M`,
//! `poly_to_indexed_mesh(indexed_mesh_to_poly(M)).vertex_count() ≤ M.vertex_count()`.
//!
//! Equality holds when every pair of vertices in `M` is farther apart than
//! `gaia::Scalar::tolerance()` for `f64` (≈ 1 nm). Positions narrowed to
//! `f32` and back to `f64` may collapse vertices that were distinguishable
//! only in the sub-10 µm range — below the `f32` tolerance threshold.

use gaia::domain::core::index::VertexId;
use gaia::{IndexedMesh, MeshBuilder};
use nalgebra::Point3;

use super::vtk_data_object::{AttributeArray, VtkPolyData};

// ── IndexedMesh → VtkPolyData ─────────────────────────────────────────────────

/// Convert a [`gaia::IndexedMesh<f64>`] to a [`VtkPolyData`].
///
/// Vertices are emitted in [`VertexId`] sequential order (0-based). Per-vertex
/// normals from the `VertexPool` are stored in `point_data["Normals"]`. Each
/// triangle face becomes one polygon entry with three vertex indices.
///
/// Precision: vertex coordinates are narrowed from `f64` to `f32` to match
/// `VtkPolyData::points`' element type. Sub-10 µm differences are lost.
pub fn indexed_mesh_to_poly(mesh: &IndexedMesh) -> VtkPolyData {
    let nv = mesh.vertex_count();
    let nf = mesh.face_count();

    let mut points: Vec<[f32; 3]> = Vec::with_capacity(nv);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(nv);

    for i in 0..nv {
        let vid = VertexId::new(i as u32);
        let p = mesh.vertices.position(vid);
        let n = mesh.vertices.normal(vid);
        points.push([p.x as f32, p.y as f32, p.z as f32]);
        normals.push([n.x as f32, n.y as f32, n.z as f32]);
    }

    let mut polygons: Vec<Vec<u32>> = Vec::with_capacity(nf);
    for (_, face) in mesh.faces.iter_enumerated() {
        let [v0, v1, v2] = face.vertices;
        polygons.push(vec![v0.raw(), v1.raw(), v2.raw()]);
    }

    let mut poly = VtkPolyData {
        points,
        polygons,
        ..Default::default()
    };
    poly.point_data.insert(
        "Normals".to_string(),
        AttributeArray::Normals { values: normals },
    );
    poly
}

// ── VtkPolyData → IndexedMesh ─────────────────────────────────────────────────

/// Convert a [`VtkPolyData`] triangle mesh to a [`gaia::IndexedMesh<f64>`].
///
/// Only triangular polygon cells (exactly 3 vertex indices) are processed.
/// Quad and higher-arity polygon cells are silently skipped. Vertex positions
/// are promoted from `f32` to `f64`. Welding is applied via `MeshBuilder`'s
/// internal `VertexPool` (tolerance ≈ 1 nm), so coincident vertices in the
/// source mesh are automatically deduplicated.
///
/// Out-of-bounds polygon indices (violating `VtkPolyData`'s invariant) are
/// silently skipped rather than panicking.
pub fn poly_to_indexed_mesh(poly: &VtkPolyData) -> IndexedMesh {
    let n_points = poly.points.len();
    let tris: Vec<(Point3<f64>, Point3<f64>, Point3<f64>)> = poly
        .polygons
        .iter()
        .filter(|tri| {
            tri.len() == 3
                && (tri[0] as usize) < n_points
                && (tri[1] as usize) < n_points
                && (tri[2] as usize) < n_points
        })
        .map(|tri| {
            let p = |i: u32| {
                let v = poly.points[i as usize];
                Point3::new(v[0] as f64, v[1] as f64, v[2] as f64)
            };
            (p(tri[0]), p(tri[1]), p(tri[2]))
        })
        .collect();

    let mut builder = MeshBuilder::new();
    builder.add_triangle_soup(&tris);
    builder.build()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytically derived unit tetrahedron (4 faces, 4 unique vertices).
    fn unit_tet_mesh() -> IndexedMesh {
        let mut b = MeshBuilder::new();
        b.add_triangle_soup(&[
            (
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
            ),
            (
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 0.5, 1.0),
            ),
            (
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
                Point3::new(0.5, 0.5, 1.0),
            ),
            (
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
                Point3::new(0.5, 0.5, 1.0),
            ),
        ]);
        b.build()
    }

    #[test]
    fn indexed_to_poly_vertex_count() {
        let mesh = unit_tet_mesh();
        let poly = indexed_mesh_to_poly(&mesh);
        assert_eq!(
            poly.points.len(),
            mesh.vertex_count(),
            "poly point count must equal welded vertex count"
        );
    }

    #[test]
    fn indexed_to_poly_face_count() {
        let mesh = unit_tet_mesh();
        let poly = indexed_mesh_to_poly(&mesh);
        assert_eq!(
            poly.polygons.len(),
            mesh.face_count(),
            "poly polygon count must equal face count"
        );
    }

    #[test]
    fn indexed_to_poly_has_normals() {
        let mesh = unit_tet_mesh();
        let poly = indexed_mesh_to_poly(&mesh);
        let normals = match poly.point_data.get("Normals").expect("Normals must be present") {
            AttributeArray::Normals { values } => values,
            other => panic!("expected Normals, got {other:?}"),
        };
        assert_eq!(
            normals.len(),
            mesh.vertex_count(),
            "normal count must equal vertex count"
        );
    }

    #[test]
    fn indexed_to_poly_coords_preserved() {
        let mesh = unit_tet_mesh();
        let poly = indexed_mesh_to_poly(&mesh);
        let v0 = poly.points[0];
        let eps = 1e-5_f32;
        // At least one vertex must be near the origin (0,0,0).
        let has_origin = poly
            .points
            .iter()
            .any(|p| p[0].abs() < eps && p[1].abs() < eps && p[2].abs() < eps);
        assert!(
            has_origin,
            "origin vertex (0,0,0) must be present in points; first point = {v0:?}"
        );
    }

    #[test]
    fn poly_to_indexed_vertex_welding() {
        // Triangle soup: 4 triangles sharing vertices (a tetrahedron).
        let poly = indexed_mesh_to_poly(&unit_tet_mesh());
        let mesh = poly_to_indexed_mesh(&poly);
        // After round-trip the welded vertex count must not exceed the original.
        assert!(
            mesh.vertex_count() <= poly.points.len(),
            "round-trip must not increase vertex count (got {} > {})",
            mesh.vertex_count(),
            poly.points.len()
        );
        assert_eq!(mesh.face_count(), poly.polygons.len(), "face count invariant");
    }

    #[test]
    fn poly_to_indexed_skips_non_triangles() {
        let poly = VtkPolyData {
            points: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            // One quad (skipped) + one degenerate (skipped).
            polygons: vec![vec![0, 1, 2, 3], vec![0, 1]],
            ..Default::default()
        };
        let mesh = poly_to_indexed_mesh(&poly);
        assert_eq!(mesh.face_count(), 0, "non-triangles must be skipped");
    }

    #[test]
    fn poly_to_indexed_empty_poly_produces_empty_mesh() {
        let poly = VtkPolyData::default();
        let mesh = poly_to_indexed_mesh(&poly);
        assert_eq!(mesh.vertex_count(), 0);
        assert_eq!(mesh.face_count(), 0);
    }
}
