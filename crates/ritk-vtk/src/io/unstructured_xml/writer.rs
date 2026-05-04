//! VTK XML UnstructuredGrid (.vtu) writer (ASCII inline format).
//!
//! # Format Reference
//! VTK XML Format Specification v0.1, Kitware Inc.
//!
//! # Invariants
//! - `write_vtu_unstructured_grid` validates via `grid.validate()` before writing.
//! - `write_vtu_str` produces a well-formed ASCII-inline VTU document.
//! - Offsets satisfy: `offsets[i] = sum(cell_len[0..=i])` (1-based cumulative).
//! - Types are emitted as the canonical `VtkCellType::to_u8()` integer code.

use crate::domain::vtk_data_object::{AttributeArray, VtkUnstructuredGrid};
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
    let dq = char::from(34u8); // ASCII double-quote
    let eq = char::from(61u8); // ASCII =
    let gt = char::from(62u8); // ASCII >
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
    // types       : VtkCellType::to_u8() for each cell.
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
        write!(s, " {}", t.to_u8()).unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "        </DataArray>").unwrap();

    writeln!(s, "      </Cells>").unwrap();

    // ── <PointData> ──────────────────────────────────────────────────────────
    if !grid.point_data.is_empty() {
        writeln!(s, "      <PointData>").unwrap();
        for (name, attr) in &grid.point_data {
            write_attr(&mut s, name, attr);
        }
        writeln!(s, "      </PointData>").unwrap();
    }

    // ── <CellData> ───────────────────────────────────────────────────────────
    if !grid.cell_data.is_empty() {
        writeln!(s, "      <CellData>").unwrap();
        for (name, attr) in &grid.cell_data {
            write_attr(&mut s, name, attr);
        }
        writeln!(s, "      </CellData>").unwrap();
    }

    writeln!(s, "    </Piece>").unwrap();
    writeln!(s, "  </UnstructuredGrid>").unwrap();
    writeln!(s, "</VTKFile>").unwrap();
    s
}

/// Emit a single attribute `<DataArray>` element into `s`.
fn write_attr(s: &mut String, name: &str, attr: &AttributeArray) {
    let dq = char::from(34u8);
    let gt = char::from(62u8);

    let hdr = |ncomp: usize| -> String {
        let mut h = String::from("        <DataArray type=");
        h.push(dq);
        h.push_str("Float32");
        h.push(dq);
        h.push_str(" Name=");
        h.push(dq);
        h.push_str(name);
        h.push(dq);
        h.push_str(" NumberOfComponents=");
        h.push(dq);
        h.push_str(&ncomp.to_string());
        h.push(dq);
        h.push_str(" format=");
        h.push(dq);
        h.push_str("ascii");
        h.push(dq);
        h.push(gt);
        h
    };

    match attr {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
            writeln!(s, "{}", hdr(*num_components)).unwrap();
            write!(s, "       ").unwrap();
            for x in values {
                write!(s, " {:.6}", x).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, "        </DataArray>").unwrap();
        }
        AttributeArray::Vectors { values } => {
            writeln!(s, "{}", hdr(3)).unwrap();
            write!(s, "       ").unwrap();
            for [x, y, z] in values {
                write!(s, " {:.6} {:.6} {:.6}", x, y, z).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, "        </DataArray>").unwrap();
        }
        AttributeArray::Normals { values } => {
            writeln!(s, "{}", hdr(3)).unwrap();
            write!(s, "       ").unwrap();
            for [x, y, z] in values {
                write!(s, " {:.6} {:.6} {:.6}", x, y, z).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, "        </DataArray>").unwrap();
        }
        AttributeArray::TextureCoords { values, dim } => {
            writeln!(s, "{}", hdr(*dim)).unwrap();
            write!(s, "       ").unwrap();
            for x in values {
                write!(s, " {:.6}", x).unwrap();
            }
            writeln!(s).unwrap();
            writeln!(s, "        </DataArray>").unwrap();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{AttributeArray, VtkCellType, VtkUnstructuredGrid};
    use tempfile::NamedTempFile;

    fn tetra() -> VtkUnstructuredGrid {
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        g.cells = vec![vec![0u32, 1, 2, 3]];
        g.cell_types = vec![VtkCellType::Tetra];
        g
    }

    #[test]
    fn test_write_vtu_str_contains_vtk_header() {
        let s = write_vtu_str(&tetra());
        assert!(s.contains("VTKFile"), "output must contain VTKFile element");
        assert!(
            s.contains("UnstructuredGrid"),
            "output must contain UnstructuredGrid"
        );
        assert!(s.contains("Piece"), "output must contain Piece");
    }

    #[test]
    fn test_write_vtu_str_number_of_points_and_cells() {
        let s = write_vtu_str(&tetra());
        // NumberOfPoints="4"
        let dq = char::from(34u8);
        let np_attr: String = ["NumberOfPoints=", &dq.to_string(), "4", &dq.to_string()].concat();
        assert!(s.contains(&np_attr), "must contain NumberOfPoints=4");
        let nc_attr: String = ["NumberOfCells=", &dq.to_string(), "1", &dq.to_string()].concat();
        assert!(s.contains(&nc_attr), "must contain NumberOfCells=1");
    }

    #[test]
    fn test_write_vtu_str_offsets_cumulative() {
        // Two triangles: sizes [3, 3] → offsets [3, 6].
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [1.5, 1.0, 0.0],
        ];
        g.cells = vec![vec![0u32, 1, 2], vec![1u32, 2, 3]];
        g.cell_types = vec![VtkCellType::Triangle, VtkCellType::Triangle];
        let s = write_vtu_str(&g);
        // Offsets section must contain "3" and "6" (cumulative counts).
        assert!(
            s.contains("Name=\"offsets\""),
            "must have offsets DataArray"
        );
        // Verify content includes both offset values.
        assert!(s.contains(" 3"), "offsets must include 3");
        assert!(s.contains(" 6"), "offsets must include 6");
    }

    #[test]
    fn test_write_vtu_str_cell_type_code() {
        // Tetra type code = 10; Triangle = 5.
        let s = write_vtu_str(&tetra());
        assert!(s.contains("Name=\"types\""), "must have types DataArray");
        assert!(s.contains(" 10"), "types must include code 10 for Tetra");
    }

    #[test]
    fn test_write_vtu_str_connectivity_order() {
        let s = write_vtu_str(&tetra());
        assert!(
            s.contains("Name=\"connectivity\""),
            "must have connectivity DataArray"
        );
        // Points 0,1,2,3 must appear in connectivity.
        assert!(
            s.contains(" 0 ") || s.contains(" 0\n"),
            "connectivity must contain 0"
        );
        assert!(s.contains(" 3"), "connectivity must contain 3");
    }

    #[test]
    fn test_write_vtu_str_empty_grid() {
        let g = VtkUnstructuredGrid::default();
        let s = write_vtu_str(&g);
        let dq = char::from(34u8);
        let np_attr: String = ["NumberOfPoints=", &dq.to_string(), "0", &dq.to_string()].concat();
        assert!(s.contains(&np_attr), "must contain NumberOfPoints=0");
        let nc_attr: String = ["NumberOfCells=", &dq.to_string(), "0", &dq.to_string()].concat();
        assert!(s.contains(&nc_attr), "must contain NumberOfCells=0");
        // No PointData or CellData sections emitted for empty grid.
        assert!(
            !s.contains("<PointData>"),
            "empty grid must not emit PointData"
        );
        assert!(
            !s.contains("<CellData>"),
            "empty grid must not emit CellData"
        );
    }

    #[test]
    fn test_write_vtu_str_point_data_emitted() {
        let mut g = tetra();
        g.point_data.insert(
            "pressure".to_string(),
            AttributeArray::Scalars {
                values: vec![1.0, 2.0, 3.0, 4.0],
                num_components: 1,
            },
        );
        let s = write_vtu_str(&g);
        assert!(s.contains("<PointData>"), "must emit PointData section");
        assert!(
            s.contains("Name=\"pressure\""),
            "must emit pressure DataArray"
        );
        // Values 1.0 through 4.0 must appear.
        assert!(s.contains("1.000000"), "must contain value 1.000000");
        assert!(s.contains("4.000000"), "must contain value 4.000000");
    }

    #[test]
    fn test_write_vtu_str_cell_data_emitted() {
        let mut g = tetra();
        g.cell_data.insert(
            "stress".to_string(),
            AttributeArray::Scalars {
                values: vec![42.0],
                num_components: 1,
            },
        );
        let s = write_vtu_str(&g);
        assert!(s.contains("<CellData>"), "must emit CellData section");
        assert!(s.contains("Name=\"stress\""), "must emit stress DataArray");
        assert!(s.contains("42.000000"), "must contain value 42.000000");
    }

    #[test]
    fn test_write_vtu_unstructured_grid_creates_file() {
        let tmp = NamedTempFile::new().expect("temp file");
        write_vtu_unstructured_grid(tmp.path(), &tetra()).expect("write must succeed");
        let content = std::fs::read_to_string(tmp.path()).expect("read back");
        assert!(
            content.contains("VTKFile"),
            "written file must contain VTKFile"
        );
        assert!(
            content.contains("NumberOfPoints"),
            "written file must contain NumberOfPoints"
        );
    }

    #[test]
    fn test_write_vtu_unstructured_grid_invalid_grid_returns_err() {
        let mut g = VtkUnstructuredGrid::new();
        g.cells = vec![vec![0u32, 1, 2]];
        g.cell_types = vec![]; // cell_types.len() != cells.len()
        let tmp = NamedTempFile::new().expect("temp file");
        let result = write_vtu_unstructured_grid(tmp.path(), &g);
        assert!(
            result.is_err(),
            "invalid grid (type count mismatch) must return Err"
        );
    }

    #[test]
    fn test_write_vtu_str_vectors_in_point_data() {
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[0.0, 0.0, 0.0]];
        g.cells = vec![vec![0u32]];
        g.cell_types = vec![VtkCellType::Vertex];
        g.point_data.insert(
            "vel".to_string(),
            AttributeArray::Vectors {
                values: vec![[1.0, 2.0, 3.0]],
            },
        );
        let s = write_vtu_str(&g);
        assert!(s.contains("Name=\"vel\""), "must emit vel DataArray");
        assert!(
            s.contains("NumberOfComponents=\"3\""),
            "vectors must have 3 components"
        );
        assert!(s.contains("1.000000"), "must contain x=1.0");
        assert!(s.contains("3.000000"), "must contain z=3.0");
    }
}
