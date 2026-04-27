//! VTK XML UnstructuredGrid (.vtu) reader (ASCII inline format).
//!
//! # Format Reference
//! VTK XML Format Specification v0.1, Kitware Inc.
//!
//! # Parsing Contract
//! - Finds the first `<Piece>` tag and reads `NumberOfPoints` / `NumberOfCells`.
//! - `<Points>` section: single DataArray of `n_points * 3` f32 coordinates.
//! - `<Cells>` section contains three named DataArrays:
//!   - `"connectivity"` : flat point-index list (length = sum of all cell sizes).
//!   - `"offsets"`      : cumulative cell-size sums; `offsets[i] = Σ size[0..=i]`.
//!   - `"types"`        : per-cell VTK integer type codes.
//! - Cell `i` spans `connectivity[offsets[i-1]..offsets[i]]` (`offsets[-1] = 0`).
//! - Unknown type codes are mapped to `VtkCellType::Vertex` with a `tracing::warn`.
//! - `<PointData>` and `<CellData>` are optional; absent sections yield empty maps.

use crate::domain::vtk_data_object::{AttributeArray, VtkCellType, VtkUnstructuredGrid};
use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

/// Read a VTU XML (ASCII inline) file from disk into a [`VtkUnstructuredGrid`].
pub fn read_vtu_unstructured_grid<P: AsRef<Path>>(path: P) -> Result<VtkUnstructuredGrid> {
    let s = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("cannot open VTU: {}", path.as_ref().display()))?;
    parse_vtu(&s)
}

/// Parse an ASCII-inline VTU XML string into a [`VtkUnstructuredGrid`].
pub(crate) fn parse_vtu(input: &str) -> Result<VtkUnstructuredGrid> {
    // ── Piece header ─────────────────────────────────────────────────────────
    let piece = find_tag(input, "Piece")
        .ok_or_else(|| anyhow::anyhow!("missing <Piece> tag in VTU document"))?;
    let n_points: usize = attr_usize(&piece, "NumberOfPoints")?;
    let n_cells: usize = attr_usize(&piece, "NumberOfCells")?;

    // ── Points ────────────────────────────────────────────────────────────────
    let points_sec =
        find_section(input, "Points").ok_or_else(|| anyhow::anyhow!("missing <Points> section"))?;
    let coords = parse_floats(&extract_da_content(&points_sec));
    if coords.len() != n_points * 3 {
        bail!(
            "expected {} coord values for {} points, got {}",
            n_points * 3,
            n_points,
            coords.len()
        );
    }
    let points: Vec<[f32; 3]> = coords.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    // ── Cells ─────────────────────────────────────────────────────────────────
    let cells_sec =
        find_section(input, "Cells").ok_or_else(|| anyhow::anyhow!("missing <Cells> section"))?;

    let conn_da = named_da(&cells_sec, "connectivity")
        .ok_or_else(|| anyhow::anyhow!("missing connectivity DataArray in <Cells>"))?;
    let offs_da = named_da(&cells_sec, "offsets")
        .ok_or_else(|| anyhow::anyhow!("missing offsets DataArray in <Cells>"))?;
    let types_da = named_da(&cells_sec, "types")
        .ok_or_else(|| anyhow::anyhow!("missing types DataArray in <Cells>"))?;

    let connectivity: Vec<u32> = parse_ints(&extract_da_content(&conn_da))
        .into_iter()
        .map(|v| v as u32)
        .collect();
    let offsets: Vec<usize> = parse_ints(&extract_da_content(&offs_da))
        .into_iter()
        .map(|v| v as usize)
        .collect();
    let type_codes: Vec<i32> = parse_ints(&extract_da_content(&types_da));

    if offsets.len() != n_cells {
        bail!(
            "offsets count {} != NumberOfCells {}",
            offsets.len(),
            n_cells
        );
    }
    if type_codes.len() != n_cells {
        bail!(
            "types count {} != NumberOfCells {}",
            type_codes.len(),
            n_cells
        );
    }

    // Reconstruct cells: cell i = connectivity[offsets[i-1]..offsets[i]].
    let mut cells: Vec<Vec<u32>> = Vec::with_capacity(n_cells);
    let mut prev: usize = 0;
    for &off in &offsets {
        if off > connectivity.len() {
            bail!(
                "offset {} exceeds connectivity length {}",
                off,
                connectivity.len()
            );
        }
        cells.push(connectivity[prev..off].to_vec());
        prev = off;
    }

    let cell_types: Vec<VtkCellType> = type_codes
        .iter()
        .map(|&v| {
            VtkCellType::from_u8(v as u8).unwrap_or_else(|| {
                tracing::warn!(
                    code = v,
                    "unknown VTK cell type code in VTU; mapped to Vertex"
                );
                VtkCellType::Vertex
            })
        })
        .collect();

    // ── Attribute sections ───────────────────────────────────────────────────
    let point_data = find_section(input, "PointData")
        .map(|sec| parse_attrs(&sec))
        .unwrap_or_default();
    let cell_data = find_section(input, "CellData")
        .map(|sec| parse_attrs(&sec))
        .unwrap_or_default();

    let mut grid = VtkUnstructuredGrid::default();
    grid.points = points;
    grid.cells = cells;
    grid.cell_types = cell_types;
    grid.point_data = point_data;
    grid.cell_data = cell_data;

    grid.validate().map_err(|e| anyhow::anyhow!("{}", e))?;
    Ok(grid)
}

// ── XML helpers ───────────────────────────────────────────────────────────────

/// Return the opening tag string for the first occurrence of `<tag ...>` or
/// `<tag>` in `s`, including the closing `>`.
fn find_tag(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let start = s.find(&open)?;
    let end = s[start..].find('>')? + 1;
    Some(s[start..start + end].to_string())
}

/// Return the substring from the first `<tag` to the matching `</tag>` (inclusive).
fn find_section(s: &str, tag: &str) -> Option<String> {
    let open = format!("<{}", tag);
    let close = format!("</{}>", tag);
    let start = s.find(&open)?;
    let end_offset = s[start..].find(&close)? + close.len();
    Some(s[start..start + end_offset].to_string())
}

/// Parse the `name="value"` attribute from an XML tag string.
fn attr_val(tag: &str, name: &str) -> Option<String> {
    let dq = char::from(34u8); // "
    let mut pat = name.to_string();
    pat.push(char::from(61u8)); // =
    pat.push(dq);
    let start = tag.find(&pat)? + pat.len();
    let rest = &tag[start..];
    let end = rest.find(dq)?;
    Some(rest[..end].to_string())
}

/// Parse a `usize` attribute from an XML tag string.
fn attr_usize(tag: &str, name: &str) -> Result<usize> {
    let v = attr_val(tag, name)
        .ok_or_else(|| anyhow::anyhow!("attribute '{}' not found in tag: {}", name, tag))?;
    v.parse()
        .with_context(|| format!("cannot parse attribute '{}' as usize: {}", name, v))
}

/// Extract the text content of the first `<DataArray ...>...</DataArray>` in `section`.
fn extract_da_content(section: &str) -> String {
    let da_start = match section.find("<DataArray") {
        Some(p) => p,
        None => return String::new(),
    };
    let rest = &section[da_start..];
    let gt = match rest.find('>') {
        Some(p) => p + 1,
        None => return String::new(),
    };
    let lt = rest[gt..].find("</").map(|p| gt + p).unwrap_or(rest.len());
    rest[gt..lt].trim().to_string()
}

/// Find a named `<DataArray Name="name" ...>...</DataArray>` within `section`.
fn named_da(section: &str, name: &str) -> Option<String> {
    let dq = char::from(34u8);
    let mut np = String::from("Name=");
    np.push(dq);
    np.push_str(name);
    np.push(dq);
    let attr_pos = section.find(&np)?;
    let da_start = section[..attr_pos].rfind("<DataArray")?;
    let rest = &section[da_start..];
    let close = "</DataArray>";
    let end = rest.find(close)? + close.len();
    Some(rest[..end].to_string())
}

/// Parse space/newline-delimited f32 values from a string slice.
fn parse_floats(s: &str) -> Vec<f32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse space/newline-delimited i32 values from a string slice.
fn parse_ints(s: &str) -> Vec<i32> {
    s.split_whitespace()
        .filter_map(|t| t.parse().ok())
        .collect()
}

/// Parse all `<DataArray>` elements in a PointData/CellData section into an
/// attribute map.  Arrays with `NumberOfComponents="3"` are decoded as Vectors
/// (or Normals if the name contains "normal"); others are decoded as Scalars.
fn parse_attrs(section: &str) -> HashMap<String, AttributeArray> {
    let mut map = HashMap::new();
    let mut rest = section;
    let close = "</DataArray>";
    loop {
        let start = match rest.find("<DataArray") {
            Some(s) => s,
            None => break,
        };
        rest = &rest[start..];
        let te = match rest.find('>') {
            Some(e) => e + 1,
            None => break,
        };
        let tag = &rest[..te];
        let name = attr_val(tag, "Name").unwrap_or_default();
        let ncomp: usize = attr_val(tag, "NumberOfComponents")
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let de = match rest.find(close) {
            Some(e) => e,
            None => break,
        };
        let data = rest[te..de].trim().to_string();
        let floats = parse_floats(&data);
        if !name.is_empty() {
            let attr = match ncomp {
                3 => {
                    let v3: Vec<[f32; 3]> =
                        floats.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
                    if name.to_lowercase().contains("normal") {
                        AttributeArray::Normals { values: v3 }
                    } else {
                        AttributeArray::Vectors { values: v3 }
                    }
                }
                2 => AttributeArray::TextureCoords {
                    values: floats,
                    dim: 2,
                },
                n => AttributeArray::Scalars {
                    values: floats,
                    num_components: n,
                },
            };
            map.insert(name, attr);
        }
        rest = &rest[de + close.len()..];
    }
    map
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{AttributeArray, VtkCellType, VtkUnstructuredGrid};
    use crate::format::vtk::unstructured_xml::writer::write_vtu_str;

    // ── Fixtures ──────────────────────────────────────────────────────────────

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

    /// Build a minimal VTU XML string from raw section content for error injection.
    fn minimal_vtu(
        np: usize,
        nc: usize,
        points: &str,
        conn: &str,
        offs: &str,
        types: &str,
    ) -> String {
        format!(
            "<?xml version=\"1.0\"?>\n\
             <VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n\
               <UnstructuredGrid>\n\
                 <Piece NumberOfPoints=\"{}\" NumberOfCells=\"{}\">\n\
                   <Points>\n\
                     <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n\
                       {}\n\
                     </DataArray>\n\
                   </Points>\n\
                   <Cells>\n\
                     <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n\
                       {}\n\
                     </DataArray>\n\
                     <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n\
                       {}\n\
                     </DataArray>\n\
                     <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">\n\
                       {}\n\
                     </DataArray>\n\
                   </Cells>\n\
                 </Piece>\n\
               </UnstructuredGrid>\n\
             </VTKFile>",
            np, nc, points, conn, offs, types
        )
    }

    // ── Round-trip tests ──────────────────────────────────────────────────────

    #[test]
    fn test_tetra_roundtrip() {
        let g = tetra();
        let s = write_vtu_str(&g);
        let r = parse_vtu(&s).expect("parse must succeed");
        assert_eq!(r.n_points(), 4);
        assert_eq!(r.n_cells(), 1);
        assert_eq!(r.cells[0], vec![0u32, 1, 2, 3]);
        assert_eq!(r.cell_types[0], VtkCellType::Tetra);
        assert_eq!(r.cell_types[0].to_u8(), 10, "Tetra VTK code must be 10");
        assert!(
            (r.points[0][0] - 0.0).abs() < 1e-5,
            "p[0].x = {}",
            r.points[0][0]
        );
        assert!(
            (r.points[1][0] - 1.0).abs() < 1e-5,
            "p[1].x = {}",
            r.points[1][0]
        );
        assert!(
            (r.points[3][2] - 1.0).abs() < 1e-5,
            "p[3].z = {}",
            r.points[3][2]
        );
    }

    #[test]
    fn test_empty_grid_roundtrip() {
        let g = VtkUnstructuredGrid::default();
        let s = write_vtu_str(&g);
        let r = parse_vtu(&s).expect("empty grid must parse");
        assert_eq!(r.n_points(), 0);
        assert_eq!(r.n_cells(), 0);
        assert!(r.cells.is_empty());
        assert!(r.cell_types.is_empty());
        assert!(r.point_data.is_empty());
        assert!(r.cell_data.is_empty());
    }

    #[test]
    fn test_two_triangles_roundtrip() {
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
        let r = parse_vtu(&s).expect("two triangles must parse");
        assert_eq!(r.n_cells(), 2);
        assert_eq!(r.cells[0], vec![0u32, 1, 2], "cell 0 connectivity");
        assert_eq!(r.cells[1], vec![1u32, 2, 3], "cell 1 connectivity");
        assert_eq!(r.cell_types[0], VtkCellType::Triangle);
        assert_eq!(r.cell_types[1], VtkCellType::Triangle);
        assert_eq!(r.cell_types[0].to_u8(), 5, "Triangle VTK code must be 5");
    }

    #[test]
    fn test_point_data_scalars_roundtrip() {
        let mut g = tetra();
        g.point_data.insert(
            "pressure".to_string(),
            AttributeArray::Scalars {
                values: vec![1.0, 2.0, 3.0, 4.0],
                num_components: 1,
            },
        );
        let r = parse_vtu(&write_vtu_str(&g)).expect("scalars roundtrip");
        match r.point_data.get("pressure").expect("pressure attr") {
            AttributeArray::Scalars {
                values,
                num_components,
            } => {
                assert_eq!(*num_components, 1);
                assert_eq!(values.len(), 4);
                assert!((values[0] - 1.0).abs() < 1e-5, "values[0] = {}", values[0]);
                assert!((values[3] - 4.0).abs() < 1e-5, "values[3] = {}", values[3]);
            }
            other => panic!("expected Scalars, got {:?}", other),
        }
    }

    #[test]
    fn test_cell_data_scalars_roundtrip() {
        let mut g = tetra();
        g.cell_data.insert(
            "stress".to_string(),
            AttributeArray::Scalars {
                values: vec![42.0],
                num_components: 1,
            },
        );
        let r = parse_vtu(&write_vtu_str(&g)).expect("cell scalars roundtrip");
        match r.cell_data.get("stress").expect("stress attr") {
            AttributeArray::Scalars { values, .. } => {
                assert_eq!(values.len(), 1);
                assert!((values[0] - 42.0).abs() < 1e-5, "values[0] = {}", values[0]);
            }
            other => panic!("expected Scalars, got {:?}", other),
        }
    }

    #[test]
    fn test_vectors_roundtrip() {
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
        let r = parse_vtu(&write_vtu_str(&g)).expect("vectors roundtrip");
        match r.point_data.get("vel").expect("vel attr") {
            AttributeArray::Vectors { values } => {
                assert_eq!(values.len(), 1);
                assert!((values[0][0] - 1.0).abs() < 1e-5, "v.x = {}", values[0][0]);
                assert!((values[0][1] - 2.0).abs() < 1e-5, "v.y = {}", values[0][1]);
                assert!((values[0][2] - 3.0).abs() < 1e-5, "v.z = {}", values[0][2]);
            }
            other => panic!("expected Vectors, got {:?}", other),
        }
    }

    #[test]
    fn test_normals_roundtrip() {
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[0.0, 0.0, 0.0]];
        g.cells = vec![vec![0u32]];
        g.cell_types = vec![VtkCellType::Vertex];
        g.point_data.insert(
            "normals".to_string(),
            AttributeArray::Normals {
                values: vec![[0.0, 0.0, 1.0]],
            },
        );
        let r = parse_vtu(&write_vtu_str(&g)).expect("normals roundtrip");
        match r.point_data.get("normals").expect("normals attr") {
            AttributeArray::Normals { values } => {
                assert_eq!(values.len(), 1);
                assert!((values[0][2] - 1.0).abs() < 1e-5, "n.z = {}", values[0][2]);
            }
            other => panic!("expected Normals, got {:?}", other),
        }
    }

    #[test]
    fn test_cell_types_variety_roundtrip() {
        // VTK codes: Vertex=1, Triangle=5, Tetra=10, Hexahedron=12.
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[0.0; 3]; 8];
        g.cells = vec![
            vec![0u32],
            vec![0u32, 1, 2],
            vec![0u32, 1, 2, 3],
            vec![0u32, 1, 2, 3, 4, 5, 6, 7],
        ];
        g.cell_types = vec![
            VtkCellType::Vertex,
            VtkCellType::Triangle,
            VtkCellType::Tetra,
            VtkCellType::Hexahedron,
        ];
        let r = parse_vtu(&write_vtu_str(&g)).expect("variety roundtrip");
        assert_eq!(r.cell_types[0], VtkCellType::Vertex);
        assert_eq!(r.cell_types[1], VtkCellType::Triangle);
        assert_eq!(r.cell_types[2], VtkCellType::Tetra);
        assert_eq!(r.cell_types[3], VtkCellType::Hexahedron);
        // Canonical code verification.
        assert_eq!(r.cell_types[0].to_u8(), 1);
        assert_eq!(r.cell_types[1].to_u8(), 5);
        assert_eq!(r.cell_types[2].to_u8(), 10);
        assert_eq!(r.cell_types[3].to_u8(), 12);
    }

    #[test]
    fn test_point_data_and_cell_data_both_present_roundtrip() {
        let mut g = tetra();
        g.point_data.insert(
            "temperature".to_string(),
            AttributeArray::Scalars {
                values: vec![100.0, 200.0, 300.0, 400.0],
                num_components: 1,
            },
        );
        g.cell_data.insert(
            "volume".to_string(),
            AttributeArray::Scalars {
                values: vec![0.1667],
                num_components: 1,
            },
        );
        let r = parse_vtu(&write_vtu_str(&g)).expect("dual attr roundtrip");
        match r.point_data.get("temperature").expect("temperature") {
            AttributeArray::Scalars { values, .. } => {
                assert_eq!(values.len(), 4);
                assert!((values[0] - 100.0).abs() < 1e-3);
                assert!((values[3] - 400.0).abs() < 1e-3);
            }
            other => panic!("expected Scalars, got {:?}", other),
        }
        match r.cell_data.get("volume").expect("volume") {
            AttributeArray::Scalars { values, .. } => {
                assert_eq!(values.len(), 1);
                assert!((values[0] - 0.1667).abs() < 1e-4, "volume = {}", values[0]);
            }
            other => panic!("expected Scalars, got {:?}", other),
        }
    }

    // ── Boundary tests ────────────────────────────────────────────────────────

    #[test]
    fn test_single_vertex_cell_roundtrip() {
        // Single-point "vertex" cell: connectivity=[0], offsets=[1], types=[1].
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[7.0, 8.0, 9.0]];
        g.cells = vec![vec![0u32]];
        g.cell_types = vec![VtkCellType::Vertex];
        let r = parse_vtu(&write_vtu_str(&g)).expect("single vertex roundtrip");
        assert_eq!(r.n_points(), 1);
        assert_eq!(r.n_cells(), 1);
        assert_eq!(r.cells[0], vec![0u32]);
        assert_eq!(r.cell_types[0], VtkCellType::Vertex);
        assert!((r.points[0][0] - 7.0).abs() < 1e-5);
        assert!((r.points[0][1] - 8.0).abs() < 1e-5);
        assert!((r.points[0][2] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_mixed_cell_sizes_connectivity_reconstruction() {
        // Mixing a triangle (3 pts) and a tetra (4 pts) — offsets [3, 7].
        let mut g = VtkUnstructuredGrid::new();
        g.points = vec![[0.0; 3]; 5];
        g.cells = vec![vec![0u32, 1, 2], vec![0u32, 1, 2, 3]];
        g.cell_types = vec![VtkCellType::Triangle, VtkCellType::Tetra];
        let r = parse_vtu(&write_vtu_str(&g)).expect("mixed cells roundtrip");
        assert_eq!(r.n_cells(), 2);
        assert_eq!(r.cells[0].len(), 3, "triangle has 3 indices");
        assert_eq!(r.cells[1].len(), 4, "tetra has 4 indices");
        assert_eq!(r.cells[0], vec![0u32, 1, 2]);
        assert_eq!(r.cells[1], vec![0u32, 1, 2, 3]);
    }

    // ── Error tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_missing_piece_tag_error() {
        let input = "<?xml version=\"1.0\"?>\
                     <VTKFile><UnstructuredGrid></UnstructuredGrid></VTKFile>";
        let r = parse_vtu(input);
        assert!(r.is_err(), "missing <Piece> must return Err");
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("Piece"), "error must mention 'Piece': {}", msg);
    }

    #[test]
    fn test_missing_cells_section_error() {
        // Valid points but no <Cells> section.
        let s = "<?xml version=\"1.0\"?>\n\
                 <VTKFile type=\"UnstructuredGrid\">\n\
                   <UnstructuredGrid>\n\
                     <Piece NumberOfPoints=\"0\" NumberOfCells=\"0\">\n\
                       <Points>\n\
                         <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n\
                         </DataArray>\n\
                       </Points>\n\
                     </Piece>\n\
                   </UnstructuredGrid>\n\
                 </VTKFile>";
        let r = parse_vtu(s);
        assert!(r.is_err(), "missing <Cells> must return Err");
        let msg = r.unwrap_err().to_string();
        assert!(
            msg.to_lowercase().contains("cell"),
            "error must mention 'Cells': {}",
            msg
        );
    }

    #[test]
    fn test_wrong_coord_count_error() {
        // NumberOfPoints=4 but only 3 floats (need 12) provided.
        let s = minimal_vtu(4, 0, "0.0 0.0 0.0", "", "", "");
        let r = parse_vtu(&s);
        assert!(r.is_err(), "coord count mismatch must return Err");
        let msg = r.unwrap_err().to_string();
        // Error contains the expected count (12) or actual count (3) or "coord".
        assert!(
            msg.contains("12") || msg.contains("coord") || msg.contains("3"),
            "error must mention coord counts: {}",
            msg
        );
    }

    #[test]
    fn test_offsets_count_mismatch_error() {
        // NumberOfCells=1 but two offsets provided.
        let s = minimal_vtu(
            4,
            1,
            "0 0 0  1 0 0  0 1 0  0 0 1",
            "0 1 2 3",
            "4 8", // two offsets for one cell
            "10",
        );
        let r = parse_vtu(&s);
        assert!(r.is_err(), "offsets count mismatch must return Err");
        let msg = r.unwrap_err().to_string();
        assert!(
            msg.contains("offsets"),
            "error must mention 'offsets': {}",
            msg
        );
    }

    #[test]
    fn test_offset_exceeds_connectivity_error() {
        // Offset 999 exceeds connectivity length 4.
        let s = minimal_vtu(4, 1, "0 0 0  1 0 0  0 1 0  0 0 1", "0 1 2 3", "999", "10");
        let r = parse_vtu(&s);
        assert!(r.is_err(), "offset > connectivity length must return Err");
        let msg = r.unwrap_err().to_string();
        assert!(
            msg.contains("999") || msg.contains("offset") || msg.contains("connectivity"),
            "error must mention the out-of-bounds offset: {}",
            msg
        );
    }

    #[test]
    fn test_types_count_mismatch_error() {
        // NumberOfCells=2 but only one type provided.
        let s = minimal_vtu(
            4,
            2,
            "0 0 0  1 0 0  0 1 0  0 0 1",
            "0 1 2  0 1 2 3",
            "3 7",
            "5", // one type for two cells
        );
        let r = parse_vtu(&s);
        assert!(r.is_err(), "types count mismatch must return Err");
        let msg = r.unwrap_err().to_string();
        assert!(
            msg.contains("types") || msg.contains("type"),
            "error must mention 'types': {}",
            msg
        );
    }

    #[test]
    fn test_from_file_roundtrip() {
        use crate::format::vtk::unstructured_xml::writer::write_vtu_unstructured_grid;
        use tempfile::NamedTempFile;
        let g = tetra();
        let tmp = NamedTempFile::new().expect("temp file");
        write_vtu_unstructured_grid(tmp.path(), &g).expect("write");
        let r = read_vtu_unstructured_grid(tmp.path()).expect("read");
        assert_eq!(r.n_points(), 4);
        assert_eq!(r.n_cells(), 1);
        assert_eq!(r.cells[0], vec![0u32, 1, 2, 3]);
        assert_eq!(r.cell_types[0], VtkCellType::Tetra);
    }

    #[test]
    fn test_nonexistent_file_error() {
        let r = read_vtu_unstructured_grid("/nonexistent_dir_xyz/file.vtu");
        assert!(r.is_err(), "nonexistent path must return Err");
    }
}
