//! VTK legacy POLYDATA reader.
//!
//! Parses the VTK legacy ASCII and BINARY POLYDATA dataset format.
//!
//! # Supported sections
//! - POINTS (float, double)
//! - VERTICES, LINES, POLYGONS, TRIANGLE_STRIPS
//! - POINT_DATA with SCALARS (+ LOOKUP_TABLE), VECTORS, NORMALS
//! - CELL_DATA with SCALARS, VECTORS, NORMALS
//!
//! # Encoding
//! ASCII: whitespace-separated tokens.
//! BINARY: big-endian encoding -- f32/f64 for coordinates, i32 for cell indices.

use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use anyhow::{bail, Context, Result};
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Read a VTK legacy POLYDATA file from disk.
pub fn read_vtk_polydata<P: AsRef<Path>>(path: P) -> Result<VtkPolyData> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open VTK polydata file: {}", path.display()))?;
    let mut reader = BufReader::new(file);
    parse_polydata(&mut reader)
}

/// Parse a VTK legacy POLYDATA dataset from a buffered reader.
///
/// Exposed for in-memory testing without file I/O.
pub(crate) fn parse_polydata(reader: &mut dyn BufRead) -> Result<VtkPolyData> {
    // Line 1: version
    let line1 = read_line(reader)?.with_context(|| "EOF before version line")?;
    if !line1.starts_with("# vtk DataFile Version") {
        bail!("not a VTK legacy file (got: '{}')", line1);
    }
    // Line 2: description (ignored)
    let _desc = read_line(reader)?.with_context(|| "EOF before description line")?;
    // Line 3: encoding
    let enc_line = read_line(reader)?.with_context(|| "EOF before encoding line")?;
    let binary = match enc_line.to_ascii_uppercase().trim() {
        "ASCII" => false,
        "BINARY" => true,
        other => bail!("unsupported VTK encoding '{}'", other),
    };
    // Line 4: DATASET
    let ds_line = read_line(reader)?.with_context(|| "EOF before DATASET line")?;
    if !ds_line.to_ascii_uppercase().starts_with("DATASET POLYDATA") {
        bail!("expected DATASET POLYDATA, got '{}'", ds_line);
    }

    let mut poly = VtkPolyData::default();
    let mut in_point_data = false;
    let mut in_cell_data = false;
    // Track what is expected next after a SCALARS header
    let mut pending_scalars: Option<(String, usize, bool)> = None; // (name, ncomp, is_point_data)
    let mut pending_vectors: Option<(String, bool)> = None; // (name, is_point_data)
    let mut pending_normals: Option<(String, bool)> = None;

    loop {
        if let Some((name, ncomp, is_pd)) = pending_scalars.take() {
            // Expect LOOKUP_TABLE line next.
            let lt = read_line(reader)?.with_context(|| "EOF waiting for LOOKUP_TABLE")?;
            if !lt.to_ascii_uppercase().starts_with("LOOKUP_TABLE") {
                bail!("expected LOOKUP_TABLE after SCALARS, got '{}'", lt);
            }
            let n = if is_pd { poly.points.len() } else { poly.num_cells() };
            let total = n * ncomp;
            let values = if binary {
                read_binary_f32(reader, total)?
            } else {
                read_ascii_f32(reader, total)?
            };
            let arr = AttributeArray::Scalars { values, num_components: ncomp };
            if is_pd {
                poly.point_data.insert(name, arr);
            } else {
                poly.cell_data.insert(name, arr);
            }
            continue;
        }
        if let Some((name, is_pd)) = pending_vectors.take() {
            let n = if is_pd { poly.points.len() } else { poly.num_cells() };
            let flat = if binary {
                read_binary_f32(reader, n * 3)?
            } else {
                read_ascii_f32(reader, n * 3)?
            };
            let values: Vec<[f32; 3]> = flat
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            let arr = AttributeArray::Vectors { values };
            if is_pd { poly.point_data.insert(name, arr); } else { poly.cell_data.insert(name, arr); }
            continue;
        }
        if let Some((name, is_pd)) = pending_normals.take() {
            let n = if is_pd { poly.points.len() } else { poly.num_cells() };
            let flat = if binary {
                read_binary_f32(reader, n * 3)?
            } else {
                read_ascii_f32(reader, n * 3)?
            };
            let values: Vec<[f32; 3]> = flat
                .chunks_exact(3)
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            let arr = AttributeArray::Normals { values };
            if is_pd { poly.point_data.insert(name, arr); } else { poly.cell_data.insert(name, arr); }
            continue;
        }

        let line = match read_line(reader)? {
            Some(l) => l,
            None => break,
        };
        let upper = line.to_ascii_uppercase();
        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() { continue; }

        if upper.starts_with("POINTS") {
            if tokens.len() < 2 { bail!("POINTS line malformed: '{}'", line); }
            let n: usize = tokens[1].parse().with_context(|| "bad POINTS count")?;
            let is_double = tokens
                .get(2)
                .map(|s| s.to_ascii_lowercase() == "double")
                .unwrap_or(false);
            poly.points = if binary {
                if is_double {
                    let raw = read_binary_f64(reader, n * 3)?;
                    raw.chunks_exact(3)
                        .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32])
                        .collect()
                } else {
                    let raw = read_binary_f32(reader, n * 3)?;
                    raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
                }
            } else {
                let raw = read_ascii_f32(reader, n * 3)?;
                raw.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect()
            };
        } else if upper.starts_with("POLYGONS") {
            poly.polygons = read_cell_section(reader, &tokens, binary)?;
        } else if upper.starts_with("LINES") {
            poly.lines = read_cell_section(reader, &tokens, binary)?;
        } else if upper.starts_with("VERTICES") {
            poly.vertices = read_cell_section(reader, &tokens, binary)?;
        } else if upper.starts_with("TRIANGLE_STRIPS") {
            poly.triangle_strips = read_cell_section(reader, &tokens, binary)?;
        } else if upper.starts_with("POINT_DATA") {
            in_point_data = true;
            in_cell_data = false;
        } else if upper.starts_with("CELL_DATA") {
            in_cell_data = true;
            in_point_data = false;
        } else if upper.starts_with("SCALARS") {
            if tokens.len() < 3 { bail!("SCALARS line malformed: '{}'", line); }
            let name = tokens[1].to_string();
            let ncomp: usize = tokens.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
            pending_scalars = Some((name, ncomp, in_point_data));
        } else if upper.starts_with("VECTORS") {
            if tokens.len() < 2 { bail!("VECTORS line malformed: '{}'", line); }
            pending_vectors = Some((tokens[1].to_string(), in_point_data));
        } else if upper.starts_with("NORMALS") {
            if tokens.len() < 2 { bail!("NORMALS line malformed: '{}'", line); }
            pending_normals = Some((tokens[1].to_string(), in_point_data));
        } else if upper.starts_with("LOOKUP_TABLE") {
            // Standalone LOOKUP_TABLE outside SCALARS context: skip.
        }
        // Unknown lines are skipped for forward compatibility.
        let _ = (in_cell_data, in_point_data);
    }

    Ok(poly)
}

// ── Internal I/O helpers ──────────────────────────────────────────────────────

fn read_line(reader: &mut dyn BufRead) -> Result<Option<String>> {
    let mut buf = String::new();
    loop {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 { return Ok(None); }
        let trimmed = buf.trim();
        if !trimmed.is_empty() { return Ok(Some(trimmed.to_owned())); }
    }
}

fn read_ascii_f32(reader: &mut dyn BufRead, count: usize) -> Result<Vec<f32>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = String::new();
    while out.len() < count {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 { break; }
        for tok in buf.split_whitespace() {
            if out.len() >= count { break; }
            let v: f32 = tok.parse().with_context(|| format!("bad f32 token '{}'", tok))?;
            out.push(v);
        }
    }
    if out.len() != count {
        bail!("expected {} f32 values, got {}", count, out.len());
    }
    Ok(out)
}

fn read_ascii_i32(reader: &mut dyn BufRead, count: usize) -> Result<Vec<i32>> {
    let mut out = Vec::with_capacity(count);
    let mut buf = String::new();
    while out.len() < count {
        buf.clear();
        let n = reader.read_line(&mut buf)?;
        if n == 0 { break; }
        for tok in buf.split_whitespace() {
            if out.len() >= count { break; }
            let v: i32 = tok.parse().with_context(|| format!("bad i32 token '{}'", tok))?;
            out.push(v);
        }
    }
    if out.len() != count {
        bail!("expected {} i32 values, got {}", count, out.len());
    }
    Ok(out)
}

fn read_binary_f32(reader: &mut dyn Read, count: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; count * 4];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("truncated binary f32 (need {} values)", count))?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| f32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn read_binary_f64(reader: &mut dyn Read, count: usize) -> Result<Vec<f64>> {
    let mut buf = vec![0u8; count * 8];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("truncated binary f64 (need {} values)", count))?;
    Ok(buf
        .chunks_exact(8)
        .map(|c| f64::from_be_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect())
}

fn read_binary_i32(reader: &mut dyn Read, count: usize) -> Result<Vec<i32>> {
    let mut buf = vec![0u8; count * 4];
    reader
        .read_exact(&mut buf)
        .with_context(|| format!("truncated binary i32 (need {} values)", count))?;
    Ok(buf
        .chunks_exact(4)
        .map(|c| i32::from_be_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Parse a cell section (POLYGONS, LINES, VERTICES, or TRIANGLE_STRIPS).
///
/// `tokens` is the already-parsed keyword line: [KEYWORD, n_cells, total_size].
fn read_cell_section(
    reader: &mut dyn BufRead,
    tokens: &[&str],
    binary: bool,
) -> Result<Vec<Vec<u32>>> {
    if tokens.len() < 3 { bail!("cell section header malformed: {:?}", tokens); }
    let n_cells: usize = tokens[1].parse().with_context(|| "bad cell section count")?;
    let total_size: usize = tokens[2].parse().with_context(|| "bad cell section total_size")?;

    if binary {
        let ints = read_binary_i32(reader, total_size)?;
        parse_cells_from_ints(&ints, n_cells)
    } else {
        let ints = read_ascii_i32(reader, total_size)?;
        parse_cells_from_ints(&ints, n_cells)
    }
}

fn parse_cells_from_ints(ints: &[i32], n_cells: usize) -> Result<Vec<Vec<u32>>> {
    let mut cells = Vec::with_capacity(n_cells);
    let mut pos = 0;
    for _ in 0..n_cells {
        if pos >= ints.len() { bail!("truncated cell data"); }
        let count = ints[pos] as usize;
        pos += 1;
        if pos + count > ints.len() { bail!("cell overruns data buffer"); }
        let cell: Vec<u32> = ints[pos..pos + count].iter().map(|&i| i as u32).collect();
        cells.push(cell);
        pos += count;
    }
    Ok(cells)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn parse_str(s: &str) -> Result<VtkPolyData> {
        let mut cursor = Cursor::new(s.as_bytes());
        parse_polydata(&mut cursor)
    }

    const TRIANGLE_ASCII: &str = "\
# vtk DataFile Version 2.0\n\
triangle test\n\
ASCII\n\
DATASET POLYDATA\n\
POINTS 3 float\n\
0.0 0.0 0.0\n\
1.0 0.0 0.0\n\
0.5 1.0 0.0\n\
POLYGONS 1 4\n\
3 0 1 2\n";

    #[test]
    fn test_read_ascii_triangle() {
        let poly = parse_str(TRIANGLE_ASCII).unwrap();
        assert_eq!(poly.points.len(), 3);
        assert!((poly.points[0][0]).abs() < 1e-6);
        assert!((poly.points[1][0] - 1.0).abs() < 1e-6);
        assert!((poly.points[2][1] - 1.0).abs() < 1e-6);
        assert_eq!(poly.polygons, vec![vec![0u32, 1, 2]]);
        assert!(poly.lines.is_empty());
    }

    #[test]
    fn test_read_ascii_polydata_with_lines() {
        let s = "# vtk DataFile Version 2.0\ntest\nASCII\nDATASET POLYDATA\n\
POINTS 3 float\n0.0 0.0 0.0\n1.0 0.0 0.0\n2.0 0.0 0.0\n\
LINES 1 3\n2 0 1\n";
        let poly = parse_str(s).unwrap();
        assert_eq!(poly.lines.len(), 1);
        assert_eq!(poly.lines[0], vec![0u32, 1]);
    }

    #[test]
    fn test_read_ascii_point_data_scalars() {
        let s = "# vtk DataFile Version 2.0\ntest\nASCII\nDATASET POLYDATA\n\
POINTS 3 float\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.5 1.0 0.0\n\
POLYGONS 1 4\n3 0 1 2\n\
POINT_DATA 3\n\
SCALARS intensity float 1\n\
LOOKUP_TABLE default\n\
10.0\n20.0\n30.0\n";
        let poly = parse_str(s).unwrap();
        let attr = poly.point_data.get("intensity").unwrap();
        match attr {
            AttributeArray::Scalars { values, num_components } => {
                assert_eq!(*num_components, 1);
                assert!((values[0] - 10.0).abs() < 1e-6);
                assert!((values[1] - 20.0).abs() < 1e-6);
                assert!((values[2] - 30.0).abs() < 1e-6);
            }
            _ => panic!("expected Scalars"),
        }
    }

    #[test]
    fn test_read_error_wrong_dataset() {
        let s = "# vtk DataFile Version 2.0\ntest\nASCII\nDATASET STRUCTURED_POINTS\n";
        assert!(parse_str(s).is_err());
    }

    #[test]
    fn test_read_error_empty_file() {
        assert!(parse_str("").is_err());
    }

    #[test]
    fn test_read_ascii_with_cell_data() {
        let s = "# vtk DataFile Version 2.0\ntest\nASCII\nDATASET POLYDATA\n\
POINTS 3 float\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.5 1.0 0.0\n\
POLYGONS 1 4\n3 0 1 2\n\
CELL_DATA 1\n\
SCALARS pressure float 1\n\
LOOKUP_TABLE default\n\
42.5\n";
        let poly = parse_str(s).unwrap();
        let attr = poly.cell_data.get("pressure").unwrap();
        match attr {
            AttributeArray::Scalars { values, .. } => {
                assert!((values[0] - 42.5).abs() < 1e-5, "expected 42.5, got {}", values[0]);
            }
            _ => panic!("expected Scalars"),
        }
    }

    #[test]
    fn test_read_ascii_multiple_cell_types() {
        let s = "# vtk DataFile Version 2.0\ntest\nASCII\nDATASET POLYDATA\n\
POINTS 4 float\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n\
VERTICES 2 4\n1 0\n1 1\n\
POLYGONS 1 5\n4 0 1 2 3\n";
        let poly = parse_str(s).unwrap();
        assert_eq!(poly.vertices.len(), 2);
        assert_eq!(poly.polygons.len(), 1);
        assert_eq!(poly.num_cells(), 3);
    }

    #[test]
    fn test_read_ascii_vectors_normals() {
        let s = "# vtk DataFile Version 2.0\ntest\nASCII\nDATASET POLYDATA\n\
POINTS 2 float\n0.0 0.0 0.0\n1.0 0.0 0.0\n\
LINES 1 3\n2 0 1\n\
POINT_DATA 2\n\
VECTORS velocity float\n\
1.0 0.0 0.0\n0.0 1.0 0.0\n\
NORMALS norm float\n\
0.0 0.0 1.0\n0.0 0.0 1.0\n";
        let poly = parse_str(s).unwrap();
        match poly.point_data.get("velocity").unwrap() {
            AttributeArray::Vectors { values } => {
                assert_eq!(values.len(), 2);
                assert!((values[0][0] - 1.0).abs() < 1e-6);
                assert!((values[1][1] - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected Vectors"),
        }
        match poly.point_data.get("norm").unwrap() {
            AttributeArray::Normals { values } => {
                assert_eq!(values.len(), 2);
                assert!((values[0][2] - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected Normals"),
        }
    }
}
