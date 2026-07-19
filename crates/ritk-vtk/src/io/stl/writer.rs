//! STL writer ← VtkPolyData.
//!
//! Both variants require that every polygon in `polygons` has exactly three
//! vertices.  Quads or higher-order polygons return `Err`.
//!
//! ## Binary STL layout
//! ```text
//! [80 bytes]  ASCII header (filled with "RITK binary STL" + padding spaces)
//! [4  bytes]  n_triangles as u32 LE
//! Per triangle (50 bytes):
//!   [12 bytes] facet normal  (3 × f32 LE)
//!   [12 bytes] vertex 0      (3 × f32 LE)
//!   [12 bytes] vertex 1      (3 × f32 LE)
//!   [12 bytes] vertex 2      (3 × f32 LE)
//!   [ 2 bytes] attribute byte count = 0 (u16 LE)
//! ```
//! If `cell_data["Normals"]` is absent, the facet normal is written as
//! `(0, 0, 0)`.

use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::{bail, Result};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write `poly` to a binary STL file at `path`.
///
/// Returns `Err` if any polygon is not a triangle.
pub fn write_stl_binary(path: impl AsRef<Path>, poly: &VtkPolyData) -> Result<()> {
    validate_triangles(poly)?;
    let file = std::fs::File::create(path.as_ref())?;
    write_stl_binary_to_writer(&mut BufWriter::new(file), poly)
}

/// Write `poly` to an ASCII STL file at `path`.
///
/// Returns `Err` if any polygon is not a triangle.
pub fn write_stl_ascii(path: impl AsRef<Path>, poly: &VtkPolyData) -> Result<()> {
    validate_triangles(poly)?;
    let file = std::fs::File::create(path.as_ref())?;
    write_stl_ascii_to_writer(&mut BufWriter::new(file), poly)
}

// ── In-memory sinks (exposed for testing) ────────────────────────────────────

pub(crate) fn write_stl_binary_to_writer(w: &mut impl Write, poly: &VtkPolyData) -> Result<()> {
    // 80-byte header: "RITK binary STL" padded with spaces.
    let mut header = [b' '; 80];
    let tag = b"RITK binary STL";
    header[..tag.len()].copy_from_slice(tag);
    w.write_all(&header)?;

    let n = poly.polygons.len() as u32;
    w.write_all(&n.to_le_bytes())?;

    let cell_normals = extract_cell_normals(poly);

    for (i, tri) in poly.polygons.iter().enumerate() {
        let [nx, ny, nz] = cell_normals
            .and_then(|ns| ns.get(i))
            .copied()
            .unwrap_or([0.0; 3]);
        write_f32_le(w, nx)?;
        write_f32_le(w, ny)?;
        write_f32_le(w, nz)?;
        for &vi in tri {
            let [x, y, z] = poly.points[vi as usize];
            write_f32_le(w, x)?;
            write_f32_le(w, y)?;
            write_f32_le(w, z)?;
        }
        w.write_all(&[0u8, 0u8])?; // attribute byte count
    }
    w.flush()?;
    Ok(())
}

pub(crate) fn write_stl_ascii_to_writer(w: &mut impl Write, poly: &VtkPolyData) -> Result<()> {
    writeln!(w, "solid ritk")?;

    let cell_normals = extract_cell_normals(poly);

    for (i, tri) in poly.polygons.iter().enumerate() {
        let [nx, ny, nz] = cell_normals
            .and_then(|ns| ns.get(i))
            .copied()
            .unwrap_or([0.0; 3]);
        writeln!(w, "  facet normal {nx} {ny} {nz}")?;
        writeln!(w, "    outer loop")?;
        for &vi in tri {
            let [x, y, z] = poly.points[vi as usize];
            writeln!(w, "      vertex {x} {y} {z}")?;
        }
        writeln!(w, "    endloop")?;
        writeln!(w, "  endfacet")?;
    }

    writeln!(w, "endsolid ritk")?;
    w.flush()?;
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn validate_triangles(poly: &VtkPolyData) -> Result<()> {
    for (i, p) in poly.polygons.iter().enumerate() {
        if p.len() != 3 {
            bail!(
                "polygon {i} has {} vertices; STL requires exactly 3 (triangle)",
                p.len()
            );
        }
    }
    Ok(())
}

fn extract_cell_normals(poly: &VtkPolyData) -> Option<&Vec<[f32; 3]>> {
    poly.cell_data.get("Normals").and_then(|a| match a {
        AttributeArray::Normals { values } => Some(values),
        _ => None,
    })
}

#[inline]
fn write_f32_le(w: &mut impl Write, v: f32) -> Result<()> {
    w.write_all(&v.to_le_bytes())?;
    Ok(())
}
