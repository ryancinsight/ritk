//! VTK XML UnstructuredGrid (.vtu) writer (ASCII inline format).
//!
//! # Format Reference
//! VTK XML Format Specification v0.1, Kitware Inc.
//!
//! # Invariants
//! - `write_vtu_unstructured_grid` validates via `grid.validate()` before writing.
//! - `write_vtu_str` produces a well-formed ASCII-inline VTU document.
//! - Offsets satisfy: `offsets[i] = sum(cell_len[0..=i])` (1-based cumulative).
//! - Types are emitted as the canonical `u8::from(cell_type)` integer code.

use crate::domain::vtk_data_object::VtkUnstructuredGrid;
use crate::io::xml_write_attr::write_attr_xml;
use anyhow::{anyhow, Context, Result};
use std::fmt::Write;
use std::path::Path;

/// Write a [`VtkUnstructuredGrid`] to an ASCII-inline VTU XML file.
///
/// Validates the grid before writing; returns `Err` on any invariant violation
/// or I/O failure.
pub fn write_vtu_unstructured_grid<P: AsRef<Path>>(
    path: P,
    grid: &VtkUnstructuredGrid,
) -> Result<()> {
    grid.validate().map_err(|e| anyhow!("{}", e))?;
    std::fs::write(path.as_ref(), write_vtu_str(grid).as_bytes())
        .with_context(|| format!("cannot write VTU: {}", path.as_ref().display()))
}

/// Serialize a [`VtkUnstructuredGrid`] to an ASCII-inline VTU XML string.
///
/// Does not validate the grid; use `write_vtu_unstructured_grid` for validated
/// file output.
pub fn write_vtu_str(grid: &VtkUnstructuredGrid) -> String {
    let dq = '"'; // ASCII double-quote
    let eq = '='; // ASCII =
    let gt = '>'; // ASCII >
    let np = grid.points.len();
    let nc = grid.cells.len();

    let mut s = String::new();
    writeln!(s, "<?xml version=\"1.0\"?>").unwrap();
    writeln!(
        s,
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )
    .unwrap();
    writeln!(s, "  <UnstructuredGrid>").unwrap();

    // ── <Piece> ──────────────────────────────────────────────────────────────
    {
        let mut piece = String::from("    <Piece");
        piece.push_str(" NumberOfPoints");
        piece.push(eq);
        piece.push(dq);
        piece.push_str(&np.to_string());
        piece.push(dq);
        piece.push_str(" NumberOfCells");
        piece.push(eq);
        piece.push(dq);
        piece.push_str(&nc.to_string());
        piece.push(dq);
        piece.push(gt);
        writeln!(s, "{}", piece).unwrap();
    }

    // ── <Points> ─────────────────────────────────────────────────────────────
    writeln!(s, "      <Points>").unwrap();
    writeln!(
        s,
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">"
    )
    .unwrap();
    write!(s, "       ").unwrap();
    for [x, y, z] in &grid.points {
        write!(s, " {:.6} {:.6} {:.6}", x, y, z).unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "        </DataArray>").unwrap();
    writeln!(s, "      </Points>").unwrap();

    // ── <Cells> ──────────────────────────────────────────────────────────────
    // connectivity: flat list of all point indices across all cells.
    // offsets[i]  : cumulative sum of cell sizes through cell i (1-based).
    // types       : u8::from(cell_type) for each cell.
    let mut conn: Vec<u32> = Vec::new();
    let mut offs: Vec<u32> = Vec::new();
    let mut cum: u32 = 0;
    for cell in &grid.cells {
        conn.extend_from_slice(cell);
        cum += cell.len() as u32;
        offs.push(cum);
    }

    writeln!(s, "      <Cells>").unwrap();

    // connectivity
    writeln!(
        s,
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">"
    )
    .unwrap();
    write!(s, "       ").unwrap();
    for v in &conn {
        write!(s, " {}", v).unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "        </DataArray>").unwrap();

    // offsets
    writeln!(
        s,
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"
    )
    .unwrap();
    write!(s, "       ").unwrap();
    for v in &offs {
        write!(s, " {}", v).unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "        </DataArray>").unwrap();

    // types
    writeln!(
        s,
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )
    .unwrap();
    write!(s, "       ").unwrap();
    for t in &grid.cell_types {
        write!(s, " {}", u8::from(*t)).unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "        </DataArray>").unwrap();

    writeln!(s, "      </Cells>").unwrap();

    // ── <PointData> ──────────────────────────────────────────────────────────
    if !grid.point_data.is_empty() {
        writeln!(s, "      <PointData>").unwrap();
        for (name, attr) in &grid.point_data {
            write_attr_xml(&mut s, name, attr);
        }
        writeln!(s, " </PointData>").unwrap();
    }

    // ── <CellData> ───────────────────────────────────────────────────────────
    if !grid.cell_data.is_empty() {
        writeln!(s, "      <CellData>").unwrap();
        for (name, attr) in &grid.cell_data {
            write_attr_xml(&mut s, name, attr);
        }
        writeln!(s, " </CellData>").unwrap();
    }

    writeln!(s, "    </Piece>").unwrap();
    writeln!(s, "  </UnstructuredGrid>").unwrap();
    writeln!(s, "</VTKFile>").unwrap();
    s
}

#[cfg(test)]
#[path = "tests_writer.rs"]
mod tests;
