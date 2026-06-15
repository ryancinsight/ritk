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
#[path = "tests_mesh_indexed.rs"]
mod tests;
