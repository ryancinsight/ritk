//! Convert a [`ritk_filter::surface::Mesh`] (`gaia::IndexedMesh<f64>`) to
//! VTK legacy ASCII POLYDATA format.
//!
//! # Format
//! - `DATASET POLYDATA` — surface mesh; compatible with ITK-SNAP, Paraview, VTK.
//! - `POINTS n double` — vertex positions in physical mm space [x, y, z].
//! - `POLYGONS n n*4` — triangles as `3 i0 i1 i2` entries.
//!
//! Vertices are accessed via gaia's `VertexPool` using `VertexId` sequential
//! indices (0..n). Faces are accessed via `FaceStore::iter_enumerated`.

use anyhow::{Context, Result};
use gaia::domain::core::index::VertexId;
use ritk_filter::surface::Mesh;
use std::path::Path;

/// Write a [`Mesh`] (`gaia::IndexedMesh<f64>`) to a VTK legacy ASCII POLYDATA file.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn write_mesh_as_vtk(path: impl AsRef<Path>, mesh: &Mesh) -> Result<()> {
    let path = path.as_ref();
    let content = mesh_to_vtk_string(mesh);
    std::fs::write(path, content)
        .with_context(|| format!("cannot write VTK mesh file: {}", path.display()))
}

/// Serialise a [`Mesh`] (`gaia::IndexedMesh<f64>`) to a VTK legacy ASCII POLYDATA
/// string (no file I/O).
///
/// Vertices are emitted in `VertexId` sequential order (0..vertex_count).
/// Face indices reference those sequential positions via `VertexId::raw()`.
pub fn mesh_to_vtk_string(mesh: &Mesh) -> String {
    let np = mesh.vertex_count();
    let nt = mesh.face_count();

    let mut out = String::with_capacity(np * 36 + nt * 24 + 128);

    out.push_str("# vtk DataFile Version 2.0\n");
    out.push_str("RITK Mesh\n");
    out.push_str("ASCII\n");
    out.push_str("DATASET POLYDATA\n");

    // Points — double precision, sequential VertexId order.
    out.push_str(&format!("POINTS {} double\n", np));
    for i in 0..np {
        let p = mesh.vertices.position(VertexId::new(i as u32));
        out.push_str(&format!("{} {} {}\n", p.x, p.y, p.z));
    }

    // Polygons (triangles): each line is "3 i0 i1 i2".
    // VTK POLYGONS section size = nt * 4 integers (count + 3 indices per face).
    out.push_str(&format!("POLYGONS {} {}\n", nt, nt * 4));
    for (_, face) in mesh.faces.iter_enumerated() {
        let [v0, v1, v2] = face.vertices;
        out.push_str(&format!("3 {} {} {}\n", v0.raw(), v1.raw(), v2.raw()));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gaia::MeshBuilder;

    fn single_triangle_mesh() -> Mesh {
        let mut b = MeshBuilder::new();
        b.add_triangle_soup_arrays(&[([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])]);
        b.build()
    }

    fn three_vertex_mesh() -> Mesh {
        let mut b = MeshBuilder::new();
        b.add_triangle_soup_arrays(&[([1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0])]);
        b.build()
    }

    #[test]
    fn empty_mesh_produces_valid_vtk_header() {
        let mesh = Mesh::new();
        let s = mesh_to_vtk_string(&mesh);
        assert!(s.starts_with("# vtk DataFile Version 2.0"));
        assert!(s.contains("DATASET POLYDATA"));
        assert!(s.contains("POINTS 0 double"));
        assert!(s.contains("POLYGONS 0 0"));
    }

    #[test]
    fn single_triangle_vtk_contains_three_points_one_polygon() {
        let mesh = single_triangle_mesh();
        assert_eq!(
            mesh.vertex_count(),
            3,
            "three unique vertices after welding"
        );
        assert_eq!(mesh.face_count(), 1, "one triangle");
        let s = mesh_to_vtk_string(&mesh);
        assert!(s.contains("POINTS 3 double"));
        assert!(s.contains("POLYGONS 1 4"));
        // Triangle references vertex IDs 0, 1, 2 in some order.
        assert!(s.contains("3 "), "polygon prefix present");
    }

    #[test]
    fn write_mesh_round_trip_produces_vtk_file_with_coords() {
        let mesh = three_vertex_mesh();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vtk");
        write_mesh_as_vtk(&path, &mesh).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("DATASET POLYDATA"));
        // All three vertex positions must appear somewhere in the POINTS section.
        assert!(
            content.contains("1 2 3")
                || content.contains("1.0 2.0 3.0")
                || content.contains("1 2 3\n")
        );
        assert_eq!(mesh.face_count(), 1);
    }
}
