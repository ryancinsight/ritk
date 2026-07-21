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
    writeln!(s, "<?xml version=\"1.0\"?>").expect("infallible write");
    writeln!(
        s,
        "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )
    .expect("infallible: validated precondition");
    writeln!(s, "  <UnstructuredGrid>").expect("infallible write");

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
        writeln!(s, "{}", piece).expect("infallible write");
    }

    // ── <Points> ─────────────────────────────────────────────────────────────
    writeln!(s, "      <Points>").expect("infallible write");
    writeln!(
        s,
        "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">"
    )
    .expect("infallible: validated precondition");
    write!(s, "       ").expect("infallible write");
    for [x, y, z] in &grid.points {
        write!(s, " {:.6} {:.6} {:.6}", x, y, z).expect("infallible write");
    }
    writeln!(s).expect("infallible write");
    writeln!(s, "        </DataArray>").expect("infallible write");
    writeln!(s, "      </Points>").expect("infallible write");

    writeln!(s, "      <Cells>").expect("infallible write");

    // connectivity
    writeln!(
        s,
        "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">"
    )
    .expect("infallible: validated precondition");
    write!(s, "       ").expect("infallible write");
    for cell in &grid.cells {
        for index in cell {
            write!(s, " {index}").expect("infallible write");
        }
    }
    writeln!(s).expect("infallible write");
    writeln!(s, "        </DataArray>").expect("infallible write");

    // offsets
    writeln!(
        s,
        "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">"
    )
    .expect("infallible: validated precondition");
    write!(s, "       ").expect("infallible write");
    let mut offset = 0usize;
    for cell in &grid.cells {
        offset += cell.len();
        write!(s, " {offset}").expect("infallible write");
    }
    writeln!(s).expect("infallible write");
    writeln!(s, "        </DataArray>").expect("infallible write");

    // types
    writeln!(
        s,
        "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">"
    )
    .expect("infallible: validated precondition");
    write!(s, "       ").expect("infallible write");
    for t in &grid.cell_types {
        write!(s, " {}", u8::from(*t)).expect("infallible write");
    }
    writeln!(s).expect("infallible write");
    writeln!(s, "        </DataArray>").expect("infallible write");

    writeln!(s, "      </Cells>").expect("infallible write");

    // ── <PointData> ──────────────────────────────────────────────────────────
    if !grid.point_data.is_empty() {
        writeln!(s, "      <PointData>").expect("infallible write");
        for (name, attr) in &grid.point_data {
            write_attr_xml(&mut s, name, attr);
        }
        writeln!(s, " </PointData>").expect("infallible write");
    }

    // ── <CellData> ───────────────────────────────────────────────────────────
    if !grid.cell_data.is_empty() {
        writeln!(s, "      <CellData>").expect("infallible write");
        for (name, attr) in &grid.cell_data {
            write_attr_xml(&mut s, name, attr);
        }
        writeln!(s, " </CellData>").expect("infallible write");
    }

    writeln!(s, "    </Piece>").expect("infallible write");
    writeln!(s, "  </UnstructuredGrid>").expect("infallible write");
    writeln!(s, "</VTKFile>").expect("infallible write");
    s
}

#[cfg(test)]
#[path = "tests_writer.rs"]
mod tests;
