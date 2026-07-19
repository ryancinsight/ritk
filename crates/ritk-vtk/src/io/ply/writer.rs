//! PLY writer â† VtkPolyData.
//!
//! Writes `format ascii 1.0` and `format binary_little_endian 1.0`.
//!
//! ## Header produced
//! ```text
//! ply
//! format <ascii|binary_little_endian> 1.0
//! element vertex N
//! property float x
//! property float y
//! property float z
//! [property float nx]   â† only when point_data["Normals"] is present
//! [property float ny]
//! [property float nz]
//! element face M
//! property list uchar int vertex_indices
//! end_header
//! ```
//!
//! ## Binary body layout (little-endian)
//! Per vertex:  `float x`, `float y`, `float z` (+ `float nx ny nz` if normals)
//! Per face:    `uchar count`, then `count Ã— int32 LE indices`

use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::Result;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write `poly` as an ASCII PLY 1.0 file to `path`.
pub fn write_ply_ascii(path: impl AsRef<Path>, poly: &VtkPolyData) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    write_ply_ascii_to_writer(&mut BufWriter::new(file), poly)
}

/// Write `poly` as a binary little-endian PLY 1.0 file to `path`.
pub fn write_ply_binary_le(path: impl AsRef<Path>, poly: &VtkPolyData) -> Result<()> {
    let file = std::fs::File::create(path.as_ref())?;
    write_ply_binary_le_to_writer(&mut BufWriter::new(file), poly)
}

// â”€â”€ In-memory sinks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

pub(crate) fn write_ply_ascii_to_writer(w: &mut impl Write, poly: &VtkPolyData) -> Result<()> {
    let normals = extract_normals(poly);
    write_ply_header(
        w,
        "ascii",
        poly.points.len(),
        poly.polygons.len(),
        normals.is_some(),
    )?;

    // Vertex elements
    for (i, [x, y, z]) in poly.points.iter().enumerate() {
        if let Some(ns) = normals {
            let [nx, ny, nz] = ns[i];
            writeln!(w, "{x} {y} {z} {nx} {ny} {nz}")?;
        } else {
            writeln!(w, "{x} {y} {z}")?;
        }
    }

    // Face elements
    for cell in &poly.polygons {
        write!(w, "{}", cell.len())?;
        for &idx in cell {
            write!(w, " {idx}")?;
        }
        writeln!(w)?;
    }

    w.flush()?;
    Ok(())
}

pub(crate) fn write_ply_binary_le_to_writer(w: &mut impl Write, poly: &VtkPolyData) -> Result<()> {
    let normals = extract_normals(poly);
    write_ply_header(
        w,
        "binary_little_endian",
        poly.points.len(),
        poly.polygons.len(),
        normals.is_some(),
    )?;

    // Vertex elements: raw LE f32 per component
    for (i, [x, y, z]) in poly.points.iter().enumerate() {
        w.write_all(&x.to_le_bytes())?;
        w.write_all(&y.to_le_bytes())?;
        w.write_all(&z.to_le_bytes())?;
        if let Some(ns) = normals {
            let [nx, ny, nz] = ns[i];
            w.write_all(&nx.to_le_bytes())?;
            w.write_all(&ny.to_le_bytes())?;
            w.write_all(&nz.to_le_bytes())?;
        }
    }

    // Face elements: uchar count + int32 LE indices
    for cell in &poly.polygons {
        w.write_all(&[cell.len() as u8])?;
        for &idx in cell {
            w.write_all(&(idx as i32).to_le_bytes())?;
        }
    }

    w.flush()?;
    Ok(())
}

// â”€â”€ Helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn write_ply_header(
    w: &mut impl Write,
    format_str: &str,
    n_verts: usize,
    n_faces: usize,
    has_normals: bool,
) -> Result<()> {
    writeln!(w, "ply")?;
    writeln!(w, "format {format_str} 1.0")?;
    writeln!(w, "element vertex {n_verts}")?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    if has_normals {
        writeln!(w, "property float nx")?;
        writeln!(w, "property float ny")?;
        writeln!(w, "property float nz")?;
    }
    writeln!(w, "element face {n_faces}")?;
    writeln!(w, "property list uchar int vertex_indices")?;
    writeln!(w, "end_header")?;
    Ok(())
}

fn extract_normals(poly: &VtkPolyData) -> Option<&Vec<[f32; 3]>> {
    poly.point_data.get("Normals").and_then(|a| match a {
        AttributeArray::Normals { values } => Some(values),
        _ => None,
    })
}
