//! VTK legacy ASCII POLYDATA writer.
//!
//! Writes a `VtkPolyData` to the VTK legacy ASCII format.
//!
//! # Format
//!
//! ```text
//! # vtk DataFile Version 2.0
//! RITK VtkPolyData
//! ASCII
//! DATASET POLYDATA
//! POINTS n_points float
//! x y z
//! ...
//! [VERTICES / LINES / POLYGONS / TRIANGLE_STRIPS]
//! [POINT_DATA / CELL_DATA with attribute arrays]
//! ```

use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::{Context, Result};
use std::io::{BufWriter, Write};
use std::path::Path;

/// Write a `VtkPolyData` to a VTK legacy ASCII file at `path`.
pub fn write_vtk_polydata<P: AsRef<Path>>(path: P, poly: &VtkPolyData) -> Result<()> {
    let path = path.as_ref();
    let file = std::fs::File::create(path)
        .with_context(|| format!("cannot create VTK polydata file: {}", path.display()))?;
    let mut w = BufWriter::new(file);
    write_polydata(&mut w, poly)
}

/// Write a `VtkPolyData` to any `Write` sink (exposed for testing).
pub(crate) fn write_polydata(w: &mut dyn Write, poly: &VtkPolyData) -> Result<()> {
    let np = poly.points.len();

    // Header
    writeln!(w, "# vtk DataFile Version 2.0")?;
    writeln!(w, "RITK VtkPolyData")?;
    writeln!(w, "ASCII")?;
    writeln!(w, "DATASET POLYDATA")?;

    // Points
    writeln!(w, "POINTS {} float", np)?;
    for [x, y, z] in &poly.points {
        writeln!(w, "{} {} {}", x, y, z)?;
    }

    // Cell sections
    write_cell_section(w, "VERTICES", &poly.vertices)?;
    write_cell_section(w, "LINES", &poly.lines)?;
    write_cell_section(w, "POLYGONS", &poly.polygons)?;
    write_cell_section(w, "TRIANGLE_STRIPS", &poly.triangle_strips)?;

    // Point data
    if !poly.point_data.is_empty() {
        writeln!(w, "POINT_DATA {}", np)?;
        for (name, attr) in &poly.point_data {
            write_attribute(w, name, attr)?;
        }
    }

    // Cell data
    let nc = poly.num_cells();
    if !poly.cell_data.is_empty() {
        writeln!(w, "CELL_DATA {}", nc)?;
        for (name, attr) in &poly.cell_data {
            write_attribute(w, name, attr)?;
        }
    }

    Ok(())
}

fn write_cell_section(w: &mut dyn Write, keyword: &str, cells: &[Vec<u32>]) -> Result<()> {
    if cells.is_empty() {
        return Ok(());
    }
    // total_size = sum of (cell.len() + 1) for each cell (the +1 is the leading count integer)
    let total_size: usize = cells.iter().map(|c| c.len() + 1).sum();
    writeln!(w, "{} {} {}", keyword, cells.len(), total_size)?;
    for cell in cells {
        let parts: Vec<String> = std::iter::once(cell.len().to_string())
            .chain(cell.iter().map(|i| i.to_string()))
            .collect();
        writeln!(w, "{}", parts.join(" "))?;
    }
    Ok(())
}

fn write_attribute(w: &mut dyn Write, name: &str, attr: &AttributeArray) -> Result<()> {
    match attr {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
            writeln!(w, "SCALARS {} float {}", name, num_components)?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for v in values {
                writeln!(w, "{}", v)?;
            }
        }
        AttributeArray::Vectors { values } => {
            writeln!(w, "VECTORS {} float", name)?;
            for [x, y, z] in values {
                writeln!(w, "{} {} {}", x, y, z)?;
            }
        }
        AttributeArray::Normals { values } => {
            writeln!(w, "NORMALS {} float", name)?;
            for [x, y, z] in values {
                writeln!(w, "{} {} {}", x, y, z)?;
            }
        }
        AttributeArray::TextureCoords { values, dim } => {
            writeln!(w, "TEXTURE_COORDINATES {} float {}", name, dim)?;
            for chunk in values.chunks(*dim) {
                let parts: Vec<String> = chunk.iter().map(|v| v.to_string()).collect();
                writeln!(w, "{}", parts.join(" "))?;
            }
        }
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tests_writer.rs"]
mod tests;
