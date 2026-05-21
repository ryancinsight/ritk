//! Gaia-native mesh I/O: read/write [`gaia::IndexedMesh`] directly.
//!
//! These functions delegate to `gaia::infrastructure::io::{stl,obj,ply,gltf_export}`
//! to produce or consume welded, topologically-consistent [`IndexedMesh<f64>`]
//! values. They complement the [`VtkPolyData`]-based functions in the sibling
//! `stl`, `obj`, `ply`, and `gltf` modules, which are retained for VTK
//! interchange compatibility.
//!
//! # When to use each surface
//!
//! | Surface | Output type | Welding | Use case |
//! |---------|-------------|---------|----------|
//! | `read_stl_mesh` | `VtkPolyData` | None (triangle soup) | VTK pipeline, Paraview |
//! | `read_stl_indexed` | `IndexedMesh` | Yes (VertexPool) | CSG booleans, quality analysis |
//! | `write_stl_binary` | consumes `VtkPolyData` | N/A | VTK round-trip |
//! | `write_indexed_stl_binary` | consumes `IndexedMesh` | N/A | Gaia mesh export |
//!
//! [`VtkPolyData`]: crate::domain::VtkPolyData

use std::path::Path;

use anyhow::{Context, Result};
use gaia::IndexedMesh;

// ── STL ───────────────────────────────────────────────────────────────────────

/// Read an STL file (ASCII or binary) into a welded [`IndexedMesh<f64>`].
///
/// Auto-detects format from the binary-size invariant.  Vertices are welded
/// via gaia's `VertexPool` (1 nm default tolerance).
pub fn read_stl_indexed(path: impl AsRef<Path>) -> Result<IndexedMesh> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening STL file {}", path.display()))?;
    gaia::infrastructure::io::stl::read_stl(file).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Write an [`IndexedMesh`] as binary STL (little-endian, compact).
pub fn write_indexed_stl_binary(path: impl AsRef<Path>, mesh: &IndexedMesh) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("creating STL file {}", path.display()))?;
    gaia::infrastructure::io::stl::write_stl_binary(&mut file, mesh)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

/// Write an [`IndexedMesh`] as ASCII STL.
pub fn write_indexed_stl_ascii(
    path: impl AsRef<Path>,
    name: &str,
    mesh: &IndexedMesh,
) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("creating STL file {}", path.display()))?;
    gaia::infrastructure::io::stl::write_stl_ascii(&mut file, name, mesh)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

// ── OBJ ───────────────────────────────────────────────────────────────────────

/// Read a Wavefront OBJ file into a welded [`IndexedMesh<f64>`].
///
/// Polygon faces with more than 3 vertices are fan-triangulated.
/// Vertices are welded via gaia's `VertexPool`.
pub fn read_obj_indexed(path: impl AsRef<Path>) -> Result<IndexedMesh> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening OBJ file {}", path.display()))?;
    gaia::infrastructure::io::obj::read_obj(file).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Write an [`IndexedMesh`] as Wavefront OBJ.
///
/// Emits `v` (position), `vn` (normal), and `f` (face) records with
/// 1-based indices per the OBJ specification.
pub fn write_indexed_obj(path: impl AsRef<Path>, mesh: &IndexedMesh) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("creating OBJ file {}", path.display()))?;
    gaia::infrastructure::io::obj::write_obj(&mut file, mesh).map_err(|e| anyhow::anyhow!("{}", e))
}

// ── PLY ───────────────────────────────────────────────────────────────────────

/// Read an ASCII PLY file into a welded [`IndexedMesh<f64>`].
pub fn read_ply_indexed(path: impl AsRef<Path>) -> Result<IndexedMesh> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("opening PLY file {}", path.display()))?;
    gaia::infrastructure::io::ply::read_ply(file).map_err(|e| anyhow::anyhow!("{}", e))
}

/// Write an [`IndexedMesh`] as ASCII PLY.
///
/// Emits vertex positions, normals, and triangular face records.
pub fn write_indexed_ply(path: impl AsRef<Path>, mesh: &IndexedMesh) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("creating PLY file {}", path.display()))?;
    gaia::infrastructure::io::ply::write_ply(&mut file, mesh).map_err(|e| anyhow::anyhow!("{}", e))
}

// ── glTF 2.0 ─────────────────────────────────────────────────────────────────

/// Write an [`IndexedMesh`] as a glTF 2.0 Binary (`.glb`) file.
///
/// Produces a self-contained GLB with one mesh primitive containing
/// `POSITION` and `NORMAL` accessors (both `f32`) and a `u32` index buffer.
pub fn write_indexed_glb(path: impl AsRef<Path>, mesh: &IndexedMesh) -> Result<()> {
    let path = path.as_ref();
    let mut file = std::fs::File::create(path)
        .with_context(|| format!("creating GLB file {}", path.display()))?;
    gaia::infrastructure::io::gltf_export::write_glb(&mut file, mesh)
        .map_err(|e| anyhow::anyhow!("{}", e))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use gaia::MeshBuilder;
    use nalgebra::Point3;
    use tempfile::NamedTempFile;

    /// A single equilateral-ish triangle for smoke tests.
    fn single_triangle() -> IndexedMesh {
        let mut b = MeshBuilder::new();
        b.add_triangle_soup(&[(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.5, 1.0, 0.0),
        )]);
        b.build()
    }

    // ── STL round-trip ───────────────────────────────────────────────────────

    #[test]
    fn stl_binary_roundtrip_vertex_and_face_counts() {
        let mesh = single_triangle();
        let f = NamedTempFile::new().unwrap();
        write_indexed_stl_binary(f.path(), &mesh).unwrap();
        let loaded = read_stl_indexed(f.path()).unwrap();
        assert_eq!(loaded.face_count(), 1, "one triangle round-tripped");
        // Vertex count: exactly 3 unique welded vertices.
        assert_eq!(
            loaded.vertex_count(),
            3,
            "three distinct vertices after welding"
        );
    }

    #[test]
    fn stl_ascii_roundtrip_vertex_and_face_counts() {
        let mesh = single_triangle();
        let f = NamedTempFile::new().unwrap();
        write_indexed_stl_ascii(f.path(), "test", &mesh).unwrap();
        let loaded = read_stl_indexed(f.path()).unwrap();
        assert_eq!(loaded.face_count(), 1);
        assert_eq!(loaded.vertex_count(), 3);
    }

    #[test]
    fn stl_binary_roundtrip_coords() {
        let mesh = single_triangle();
        let f = NamedTempFile::new().unwrap();
        write_indexed_stl_binary(f.path(), &mesh).unwrap();
        let loaded = read_stl_indexed(f.path()).unwrap();
        use gaia::domain::core::index::VertexId;
        // Collect all vertex positions.
        let pts: Vec<_> = (0..loaded.vertex_count())
            .map(|i| loaded.vertices.position(VertexId::new(i as u32)))
            .collect();
        let eps = 1e-4_f64; // binary STL stores f32, so tolerance ≥ f32 epsilon
        let has_origin = pts
            .iter()
            .any(|p| p.x.abs() < eps && p.y.abs() < eps && p.z.abs() < eps);
        assert!(
            has_origin,
            "origin vertex must survive round-trip; pts={pts:?}"
        );
    }

    // ── OBJ round-trip ───────────────────────────────────────────────────────

    #[test]
    fn obj_roundtrip_vertex_and_face_counts() {
        let mesh = single_triangle();
        let f = NamedTempFile::new().unwrap();
        write_indexed_obj(f.path(), &mesh).unwrap();
        let loaded = read_obj_indexed(f.path()).unwrap();
        assert_eq!(loaded.face_count(), 1);
        assert_eq!(loaded.vertex_count(), 3);
    }

    // ── PLY round-trip ───────────────────────────────────────────────────────

    #[test]
    fn ply_roundtrip_vertex_and_face_counts() {
        let mesh = single_triangle();
        let f = NamedTempFile::new().unwrap();
        write_indexed_ply(f.path(), &mesh).unwrap();
        let loaded = read_ply_indexed(f.path()).unwrap();
        assert_eq!(loaded.face_count(), 1);
        assert_eq!(loaded.vertex_count(), 3);
    }

    // ── GLB write ────────────────────────────────────────────────────────────

    #[test]
    fn glb_write_produces_valid_glb_header() {
        let mesh = single_triangle();
        let f = NamedTempFile::new().unwrap();
        write_indexed_glb(f.path(), &mesh).unwrap();
        let bytes = std::fs::read(f.path()).unwrap();
        // GLB magic = 0x46546C67 ("glTF" in little-endian).
        assert!(
            bytes.len() >= 4,
            "GLB must be non-empty; got {} bytes",
            bytes.len()
        );
        assert_eq!(&bytes[..4], b"glTF", "GLB magic bytes must be 'glTF'");
    }
}
