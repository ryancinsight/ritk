//! Convert a `ritk_core::filter::surface::Mesh` to VTK legacy POLYDATA format.
//!
//! Triangles are emitted as POLYGONS (each polygon has 3 vertices).
//! Vertices are written as POINTS in physical mm space [x, y, z].

use anyhow::{Context, Result};
use ritk_core::filter::surface::Mesh;
use std::path::Path;

/// Write a `Mesh` to a VTK legacy ASCII POLYDATA file.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn write_mesh_as_vtk(path: impl AsRef<Path>, mesh: &Mesh) -> Result<()> {
    let path = path.as_ref();
    let content = mesh_to_vtk_string(mesh);
    std::fs::write(path, content)
        .with_context(|| format!("cannot write VTK mesh file: {}", path.display()))
}

/// Serialise `mesh` to a VTK legacy ASCII POLYDATA string (no file I/O).
pub fn mesh_to_vtk_string(mesh: &Mesh) -> String {
    let np = mesh.vertices.len();
    let nt = mesh.triangles.len();

    let mut out = String::with_capacity(np * 32 + nt * 24 + 128);

    out.push_str("# vtk DataFile Version 2.0\n");
    out.push_str("RITK Mesh\n");
    out.push_str("ASCII\n");
    out.push_str("DATASET POLYDATA\n");

    // Points
    out.push_str(&format!("POINTS {} float\n", np));
    for [x, y, z] in &mesh.vertices {
        out.push_str(&format!("{} {} {}\n", x, y, z));
    }

    // Polygons (triangles): each line is "3 i0 i1 i2"
    // Total size in VTK POLYGONS section = nt * 4 integers.
    out.push_str(&format!("POLYGONS {} {}\n", nt, nt * 4));
    for [i0, i1, i2] in &mesh.triangles {
        out.push_str(&format!("3 {} {} {}\n", i0, i1, i2));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_core::filter::surface::Mesh;

    #[test]
    fn empty_mesh_produces_valid_vtk_header() {
        let mesh = Mesh::new();
        let s = mesh_to_vtk_string(&mesh);
        assert!(s.starts_with("# vtk DataFile Version 2.0"));
        assert!(s.contains("DATASET POLYDATA"));
        assert!(s.contains("POINTS 0 float"));
        assert!(s.contains("POLYGONS 0 0"));
    }

    #[test]
    fn single_triangle_vtk_contains_three_points_one_polygon() {
        let mesh = Mesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        let s = mesh_to_vtk_string(&mesh);
        assert!(s.contains("POINTS 3 float"));
        assert!(s.contains("POLYGONS 1 4"));
        assert!(s.contains("3 0 1 2"));
    }

    #[test]
    fn write_mesh_round_trip_produces_correct_file() {
        let mesh = Mesh {
            vertices: vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            triangles: vec![[0, 1, 2]],
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.vtk");
        write_mesh_as_vtk(&path, &mesh).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("1 2 3"));
        assert!(content.contains("3 0 1 2"));
    }
}
