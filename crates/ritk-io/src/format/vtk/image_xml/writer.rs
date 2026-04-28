//! VTK XML ImageData (.vti) writer (ASCII inline format).
//!
//! # Format Reference
//! VTK XML Format Specification v0.1, Kitware Inc.
//!
//! # Invariants
//! - `write_vti_image_data` validates via `grid.validate()` before writing.
//! - `write_vti_str` produces a well-formed ASCII-inline VTI document.
//! - WholeExtent and Piece Extent are always equal (single-piece writer).
//! - Origin and Spacing are formatted with 6 decimal places.
//! - DataArray values are formatted with 6 decimal places.

use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use anyhow::{anyhow, Context, Result};
use std::fmt::Write;
use std::path::Path;

/// Write a [`VtkImageData`] to an ASCII-inline VTI XML file.
///
/// Validates the grid before writing; returns `Err` on any invariant violation
/// or I/O failure.
pub fn write_vti_image_data<P: AsRef<Path>>(path: P, grid: &VtkImageData) -> Result<()> {
    grid.validate().map_err(|e| anyhow!("{}", e))?;
    std::fs::write(path.as_ref(), write_vti_str(grid).as_bytes())
        .with_context(|| format!("cannot write VTI: {}", path.as_ref().display()))
}

/// Serialize a [`VtkImageData`] to an ASCII-inline VTI XML string.
///
/// Does not validate the grid; use `write_vti_image_data` for validated
/// file output.
pub fn write_vti_str(grid: &VtkImageData) -> String {
    let e = &grid.whole_extent;
    let extent_str = format!("{} {} {} {} {} {}", e[0], e[1], e[2], e[3], e[4], e[5]);
    let origin_str = format!(
        "{:.6} {:.6} {:.6}",
        grid.origin[0], grid.origin[1], grid.origin[2]
    );
    let spacing_str = format!(
        "{:.6} {:.6} {:.6}",
        grid.spacing[0], grid.spacing[1], grid.spacing[2]
    );

    let mut s = String::new();
    writeln!(s, "<?xml version=\"1.0\"?>").unwrap();
    writeln!(
        s,
        "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )
    .unwrap();
    writeln!(
        s,
        "  <ImageData WholeExtent=\"{}\" Origin=\"{}\" Spacing=\"{}\">",
        extent_str, origin_str, spacing_str
    )
    .unwrap();
    writeln!(s, "    <Piece Extent=\"{}\">", extent_str).unwrap();

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
    writeln!(s, "  </ImageData>").unwrap();
    writeln!(s, "</VTKFile>").unwrap();
    s
}

/// Emit a single attribute `<DataArray>` element into `s`.
fn write_attr(s: &mut String, name: &str, attr: &AttributeArray) {
    let dq = char::from(34u8); // "
    let gt = char::from(62u8); // >

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

// ── Binary-appended writer ────────────────────────────────────────────────────

/// Number of components for a [`AttributeArray`] (used to emit `NumberOfComponents`).
fn attr_ncomp(attr: &AttributeArray) -> usize {
    match attr {
        AttributeArray::Scalars { num_components, .. } => *num_components,
        AttributeArray::Vectors { .. } | AttributeArray::Normals { .. } => 3,
        AttributeArray::TextureCoords { dim, .. } => *dim,
    }
}

/// Flatten a [`AttributeArray`] to a contiguous `Vec<f32>` for binary serialization.
///
/// - `Scalars`: values as-is.
/// - `Vectors`, `Normals`: interleaved `[x, y, z, x, y, z, ...]`.
/// - `TextureCoords`: values as-is.
fn flatten_attr(attr: &AttributeArray) -> Vec<f32> {
    match attr {
        AttributeArray::Scalars { values, .. } => values.clone(),
        AttributeArray::Vectors { values } => {
            values.iter().flat_map(|v| v.iter().copied()).collect()
        }
        AttributeArray::Normals { values } => {
            values.iter().flat_map(|v| v.iter().copied()).collect()
        }
        AttributeArray::TextureCoords { values, .. } => values.clone(),
    }
}

/// Emit a self-closing `<DataArray ... format="appended" offset="N"/>` line into `s`.
fn write_da_appended_tag(s: &mut String, name: &str, ncomp: usize, offset: usize) {
    let dq = char::from(34u8); // "
    s.push_str("        <DataArray type=");
    s.push(dq);
    s.push_str("Float32");
    s.push(dq);
    s.push_str(" Name=");
    s.push(dq);
    s.push_str(name);
    s.push(dq);
    s.push_str(" NumberOfComponents=");
    s.push(dq);
    write!(s, "{}", ncomp).unwrap();
    s.push(dq);
    s.push_str(" format=");
    s.push(dq);
    s.push_str("appended");
    s.push(dq);
    s.push_str(" offset=");
    s.push(dq);
    write!(s, "{}", offset).unwrap();
    s.push(dq);
    s.push_str("/>\n");
}

/// Serialize a [`VtkImageData`] to a binary-appended VTI byte buffer.
///
/// # Format
/// Produces a VTK XML ImageData document with `encoding="raw"` AppendedData.
/// Each DataArray block in the binary region is: `uint32 LE` byte-count header
/// followed by that many bytes of `float32 LE` values.
///
/// # Invariants
/// - Validates the grid before serialization; returns `Err` on any violation.
/// - DataArrays within each section are sorted by name (lexicographic) for
///   deterministic offset computation and reproducible output.
/// - Offsets satisfy: `offset[0] = 0`,
///   `offset[i+1] = offset[i] + 4 + flat_values[i].len() * 4`.
pub fn write_vti_binary_appended_bytes(grid: &VtkImageData) -> Result<Vec<u8>> {
    grid.validate().map_err(|e| anyhow!("{}", e))?;

    let e = &grid.whole_extent;
    let extent_str = format!("{} {} {} {} {} {}", e[0], e[1], e[2], e[3], e[4], e[5]);
    let origin_str = format!(
        "{:.6} {:.6} {:.6}",
        grid.origin[0], grid.origin[1], grid.origin[2]
    );
    let spacing_str = format!(
        "{:.6} {:.6} {:.6}",
        grid.spacing[0], grid.spacing[1], grid.spacing[2]
    );

    // Collect arrays in deterministic (lexicographic) order within each section
    // so offset computation is reproducible regardless of HashMap iteration order.
    let mut pd: Vec<(&str, &AttributeArray)> = grid
        .point_data
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    pd.sort_unstable_by(|a, b| a.0.cmp(b.0));

    let mut cd: Vec<(&str, &AttributeArray)> = grid
        .cell_data
        .iter()
        .map(|(k, v)| (k.as_str(), v))
        .collect();
    cd.sort_unstable_by(|a, b| a.0.cmp(b.0));

    // Flatten all arrays (point first, then cell) and compute byte offsets.
    // offset[0] = 0; offset[i+1] = offset[i] + 4 + flat[i].len() * 4
    let all: Vec<(&str, &AttributeArray)> = pd.iter().chain(cd.iter()).copied().collect();
    let flat: Vec<Vec<f32>> = all.iter().map(|(_, a)| flatten_attr(a)).collect();
    let mut offsets: Vec<usize> = Vec::with_capacity(all.len() + 1);
    offsets.push(0);
    for fv in &flat {
        let prev = *offsets.last().unwrap();
        offsets.push(prev + 4 + fv.len() * 4);
    }

    // ── Build XML header ─────────────────────────────────────────────────────
    let mut xml = String::new();
    writeln!(xml, "<?xml version=\"1.0\"?>").unwrap();
    writeln!(
        xml,
        "<VTKFile type=\"ImageData\" version=\"0.1\" byte_order=\"LittleEndian\" header_type=\"UInt32\">"
    )
    .unwrap();
    writeln!(
        xml,
        "  <ImageData WholeExtent=\"{}\" Origin=\"{}\" Spacing=\"{}\">",
        extent_str, origin_str, spacing_str
    )
    .unwrap();
    writeln!(xml, "    <Piece Extent=\"{}\">", extent_str).unwrap();

    let pd_len = pd.len();
    if !pd.is_empty() {
        writeln!(xml, "      <PointData>").unwrap();
        for (i, (name, attr)) in pd.iter().enumerate() {
            write_da_appended_tag(&mut xml, name, attr_ncomp(attr), offsets[i]);
        }
        writeln!(xml, "      </PointData>").unwrap();
    }

    if !cd.is_empty() {
        writeln!(xml, "      <CellData>").unwrap();
        for (i, (name, attr)) in cd.iter().enumerate() {
            write_da_appended_tag(&mut xml, name, attr_ncomp(attr), offsets[pd_len + i]);
        }
        writeln!(xml, "      </CellData>").unwrap();
    }

    writeln!(xml, "    </Piece>").unwrap();
    writeln!(xml, "  </ImageData>").unwrap();
    writeln!(xml, "  <AppendedData encoding=\"raw\">").unwrap();

    // ── Assemble final buffer: XML header + '_' + binary blobs + XML footer ──
    let mut result: Vec<u8> = xml.into_bytes();
    result.push(b'_');
    for fv in &flat {
        let n_bytes = (fv.len() * 4) as u32;
        result.extend_from_slice(&n_bytes.to_le_bytes());
        for &v in fv {
            result.extend_from_slice(&v.to_le_bytes());
        }
    }
    result.extend_from_slice(b"\n  </AppendedData>\n</VTKFile>\n");

    Ok(result)
}

/// Write a [`VtkImageData`] to a binary-appended VTI XML file.
///
/// Validates the grid before writing. Uses `encoding="raw"` AppendedData format.
/// Returns `Err` on validation failure or I/O error.
pub fn write_vti_binary_appended_to_file<P: AsRef<Path>>(
    path: P,
    grid: &VtkImageData,
) -> Result<()> {
    let bytes = write_vti_binary_appended_bytes(grid)?;
    std::fs::write(path.as_ref(), &bytes).with_context(|| {
        format!(
            "cannot write binary-appended VTI: {}",
            path.as_ref().display()
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
    use tempfile::NamedTempFile;

    /// Build a 2×2×2-point grid (extent [0,1,0,1,0,1], 8 points, 1 cell)
    /// with a scalar point-data field named "density".
    fn grid_2x2x2() -> VtkImageData {
        VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            origin: [1.0, 2.0, 3.0],
            spacing: [0.5, 0.5, 0.5],
            point_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "density".to_string(),
                    AttributeArray::Scalars {
                        values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        num_components: 1,
                    },
                );
                m
            },
            cell_data: std::collections::HashMap::new(),
        }
    }

    // ── 1 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_produces_vtkfile_header() {
        let s = write_vti_str(&VtkImageData::default());
        assert!(
            s.contains("<?xml version=\"1.0\"?>"),
            "must contain XML declaration"
        );
        assert!(s.contains("<VTKFile"), "must contain VTKFile element");
        assert!(
            s.contains("type=\"ImageData\""),
            "VTKFile type must be ImageData"
        );
        assert!(s.contains("</VTKFile>"), "must contain closing VTKFile tag");
    }

    // ── 2 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_extent_in_output() {
        let grid = VtkImageData {
            whole_extent: [0, 3, 0, 4, 0, 5],
            ..Default::default()
        };
        let s = write_vti_str(&grid);
        // WholeExtent attribute must contain the exact formatted extent string.
        assert!(
            s.contains("WholeExtent=\"0 3 0 4 0 5\""),
            "WholeExtent attribute must be '0 3 0 4 0 5'; got:\n{}",
            s
        );
        // Piece Extent must equal WholeExtent.
        assert!(
            s.contains("Piece Extent=\"0 3 0 4 0 5\""),
            "Piece Extent must match WholeExtent"
        );
    }

    // ── 3 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_origin_and_spacing() {
        let grid = VtkImageData {
            whole_extent: [0, 0, 0, 0, 0, 0],
            origin: [1.5, 2.25, 3.125],
            spacing: [0.1, 0.2, 0.4],
            ..Default::default()
        };
        let s = write_vti_str(&grid);
        assert!(
            s.contains("Origin=\"1.500000 2.250000 3.125000\""),
            "Origin must be formatted with 6 d.p.; got:\n{}",
            s
        );
        assert!(
            s.contains("Spacing=\"0.100000 0.200000 0.400000\""),
            "Spacing must be formatted with 6 d.p.; got:\n{}",
            s
        );
    }

    // ── 4 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_scalar_point_data() {
        let s = write_vti_str(&grid_2x2x2());
        assert!(s.contains("<PointData>"), "must emit PointData section");
        assert!(
            s.contains("Name=\"density\""),
            "must emit density DataArray"
        );
        // All 8 scalar values must appear formatted with 6 d.p.
        for expected in &[
            "1.000000", "2.000000", "3.000000", "4.000000", "5.000000", "6.000000", "7.000000",
            "8.000000",
        ] {
            assert!(
                s.contains(expected),
                "DataArray must contain value {}; got:\n{}",
                expected,
                s
            );
        }
    }

    // ── 5 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_multicomponent() {
        // Single-point grid; 3-component velocity vector.
        let mut grid = VtkImageData {
            whole_extent: [0, 0, 0, 0, 0, 0],
            ..Default::default()
        };
        grid.point_data.insert(
            "velocity".to_string(),
            AttributeArray::Vectors {
                values: vec![[1.0f32, 2.0, 3.0]],
            },
        );
        let s = write_vti_str(&grid);
        assert!(
            s.contains("NumberOfComponents=\"3\""),
            "vectors must emit NumberOfComponents=3; got:\n{}",
            s
        );
        assert!(
            s.contains("Name=\"velocity\""),
            "must emit velocity DataArray"
        );
        assert!(s.contains("1.000000"), "must contain x component");
        assert!(s.contains("3.000000"), "must contain z component");
    }

    // ── 6 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_cell_data() {
        // extent [0,1,0,1,0,1] → n_cells = 1×1×1 = 1
        let mut grid = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            ..Default::default()
        };
        grid.cell_data.insert(
            "pressure".to_string(),
            AttributeArray::Scalars {
                values: vec![99.0f32],
                num_components: 1,
            },
        );
        let s = write_vti_str(&grid);
        assert!(s.contains("<CellData>"), "must emit CellData section");
        assert!(
            s.contains("Name=\"pressure\""),
            "must emit pressure DataArray"
        );
        assert!(s.contains("99.000000"), "must contain cell value 99.0");
    }

    // ── 7 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_empty_grid() {
        // Default VtkImageData: extent [0,0,0,0,0,0], no data arrays.
        let s = write_vti_str(&VtkImageData::default());
        assert!(
            s.contains("WholeExtent=\"0 0 0 0 0 0\""),
            "empty grid must have zero extent; got:\n{}",
            s
        );
        assert!(
            !s.contains("<PointData>"),
            "empty grid must not emit PointData"
        );
        assert!(
            !s.contains("<CellData>"),
            "empty grid must not emit CellData"
        );
        assert!(s.contains("<VTKFile"), "must still emit VTKFile");
        assert!(s.contains("<Piece"), "must still emit Piece");
    }

    // ── 8 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_file_roundtrip_via_string() {
        use crate::format::vtk::image_xml::reader::parse_vti;

        let grid = grid_2x2x2();
        let xml = write_vti_str(&grid);
        let parsed = parse_vti(&xml).expect("parse_vti must succeed on writer output");

        assert_eq!(
            parsed.whole_extent, grid.whole_extent,
            "extent must round-trip exactly"
        );
        for i in 0..3 {
            assert!(
                (parsed.origin[i] - grid.origin[i]).abs() < 1e-5,
                "origin[{}] mismatch: {} vs {}",
                i,
                parsed.origin[i],
                grid.origin[i]
            );
            assert!(
                (parsed.spacing[i] - grid.spacing[i]).abs() < 1e-5,
                "spacing[{}] mismatch: {} vs {}",
                i,
                parsed.spacing[i],
                grid.spacing[i]
            );
        }
        // Verify scalar values in point_data.
        let orig_vals = match grid.point_data.get("density").unwrap() {
            AttributeArray::Scalars { values, .. } => values.clone(),
            _ => panic!("expected Scalars"),
        };
        let parsed_vals = match parsed.point_data.get("density").unwrap() {
            AttributeArray::Scalars { values, .. } => values.clone(),
            _ => panic!("expected Scalars after parse"),
        };
        assert_eq!(orig_vals.len(), parsed_vals.len(), "value count must match");
        for (i, (o, p)) in orig_vals.iter().zip(parsed_vals.iter()).enumerate() {
            assert!(
                (o - p).abs() < 1e-5,
                "density[{}]: wrote {:.6}, parsed {:.6}",
                i,
                o,
                p
            );
        }
    }

    // ── 9 ─────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_rejects_invalid_grid() {
        // extent [0,1,0,1,0,1] → n_points = 8; supply 5 scalars → validation Err.
        let mut grid = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            ..Default::default()
        };
        grid.point_data.insert(
            "bad".to_string(),
            AttributeArray::Scalars {
                values: vec![0.0f32; 5],
                num_components: 1,
            },
        );
        let tmp = NamedTempFile::new().expect("temp file");
        let result = write_vti_image_data(tmp.path(), &grid);
        assert!(
            result.is_err(),
            "invalid grid (scalar length mismatch) must return Err"
        );
    }

    // ── 10 ────────────────────────────────────────────────────────────────────
    #[test]
    fn test_write_vti_to_file_succeeds() {
        let tmp = NamedTempFile::new().expect("temp file");
        write_vti_image_data(tmp.path(), &grid_2x2x2()).expect("write must succeed");
        let bytes = std::fs::read(tmp.path()).expect("must read back file");
        assert!(!bytes.is_empty(), "written file must be non-empty");
        let content = String::from_utf8(bytes).expect("must be valid UTF-8");
        assert!(
            content.contains("<VTKFile"),
            "file content must contain VTKFile"
        );
        assert!(
            content.contains("ImageData"),
            "file content must reference ImageData"
        );
    }

    // ── 11 ────────────────────────────────────────────────────────────────────
    /// Invariant: binary-appended output contains `format="appended"` in the
    /// XML header and `AppendedData encoding="raw"` in the header, and the
    /// total byte count exceeds the XML-only length (binary data is present).
    #[test]
    fn test_write_vti_binary_appended_header_contains_appended_format() {
        // 2×2×1 grid → n_points = 4; one scalar PointData array with 4 values.
        let grid = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 0],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            point_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "values".to_string(),
                    AttributeArray::Scalars {
                        values: vec![1.0f32, 2.0, 3.0, 4.0],
                        num_components: 1,
                    },
                );
                m
            },
            cell_data: std::collections::HashMap::new(),
        };

        let bytes = write_vti_binary_appended_bytes(&grid)
            .expect("write_vti_binary_appended_bytes must succeed on valid grid");

        // Locate the `_` marker: find <AppendedData, then its closing `>`,
        // then the first `_` byte after that.
        let ad_start = bytes
            .windows(b"<AppendedData".len())
            .position(|w| w == b"<AppendedData")
            .expect("<AppendedData tag must be present");
        let gt_rel = bytes[ad_start..]
            .iter()
            .position(|&b| b == b'>')
            .expect("<AppendedData tag must have closing >");
        let us_rel = bytes[ad_start + gt_rel + 1..]
            .iter()
            .position(|&b| b == b'_')
            .expect("'_' marker must be present after AppendedData tag");
        let underscore_abs = ad_start + gt_rel + 1 + us_rel;

        let header =
            std::str::from_utf8(&bytes[..underscore_abs]).expect("header must be valid UTF-8");

        assert!(
            header.contains("format=\"appended\""),
            "header must contain format=\"appended\"; header:\n{}",
            header
        );
        assert!(
            header.contains("AppendedData encoding=\"raw\""),
            "header must contain AppendedData encoding=\"raw\"; header:\n{}",
            header
        );
        // Binary block: 4 bytes (uint32 count) + 4 * 4 = 16 bytes (floats) = 20 bytes minimum.
        // Total = header_len + 1 ('_') + 20 (binary) + footer_len > header_len + 20.
        assert!(
            bytes.len() > underscore_abs + 20,
            "total bytes must exceed header_end+20 (binary data must be present); \
             total={}, header_end={}",
            bytes.len(),
            underscore_abs
        );
    }

    // ── 12 ────────────────────────────────────────────────────────────────────
    /// Invariant: binary-appended round-trip preserves whole_extent, origin,
    /// spacing, and all scalar values within f32 representation error (< 1e-6).
    #[test]
    fn test_write_vti_binary_appended_roundtrip() {
        use crate::format::vtk::image_xml::reader::read_vti_binary_appended_bytes;

        let expected_values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let grid = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            origin: [1.0, 2.0, 3.0],
            spacing: [0.5, 0.5, 0.5],
            point_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "scalars".to_string(),
                    AttributeArray::Scalars {
                        values: expected_values.clone(),
                        num_components: 1,
                    },
                );
                m
            },
            cell_data: std::collections::HashMap::new(),
        };

        let bytes = write_vti_binary_appended_bytes(&grid)
            .expect("write_vti_binary_appended_bytes must succeed");
        let parsed = read_vti_binary_appended_bytes(&bytes)
            .expect("read_vti_binary_appended_bytes must parse writer output");

        assert_eq!(
            parsed.whole_extent, grid.whole_extent,
            "whole_extent must round-trip exactly"
        );
        for i in 0..3 {
            assert!(
                (parsed.origin[i] - grid.origin[i]).abs() < 1e-5,
                "origin[{}]: expected {}, got {}",
                i,
                grid.origin[i],
                parsed.origin[i]
            );
            assert!(
                (parsed.spacing[i] - grid.spacing[i]).abs() < 1e-5,
                "spacing[{}]: expected {}, got {}",
                i,
                grid.spacing[i],
                parsed.spacing[i]
            );
        }
        let parsed_vals = match parsed.point_data.get("scalars") {
            Some(AttributeArray::Scalars { values, .. }) => values.clone(),
            other => panic!(
                "expected Scalars variant for 'scalars' key, got {:?}",
                other
            ),
        };
        assert_eq!(
            parsed_vals.len(),
            expected_values.len(),
            "scalar value count must match after round-trip"
        );
        for (i, (e, g)) in expected_values.iter().zip(parsed_vals.iter()).enumerate() {
            assert!(
                (e - g).abs() < 1e-6,
                "scalars[{}]: expected {}, got {} (diff {})",
                i,
                e,
                g,
                (e - g).abs()
            );
        }
    }

    // ── 13 ────────────────────────────────────────────────────────────────────
    /// Invariant: offset[0] = 0; offset[1] = 4 + n_values[0] * 4.
    ///
    /// Proof: offset[i+1] = offset[i] + sizeof(uint32) + n_bytes[i]
    ///   where n_bytes[i] = n_values[i] * sizeof(float32).
    /// For array "A" (2 float32s): n_bytes[0] = 2 * 4 = 8.
    /// offset[1] = 0 + 4 + 8 = 12.
    #[test]
    fn test_write_vti_binary_appended_offset_correctness() {
        // 2×1×1 grid → n_points = 2.
        // "A": Scalars, num_components=1, values=[1.0, 2.0]     → 2 values (2×1=2 ✓).
        // "B": Scalars, num_components=2, values=[3.0,4.0,5.0,6.0] → 4 values (2×2=4 ✓).
        // Lexicographic sort: "A" first (offset=0), "B" second (offset=12).
        let mut grid = VtkImageData {
            whole_extent: [0, 1, 0, 0, 0, 0],
            ..Default::default()
        };
        grid.point_data.insert(
            "A".to_string(),
            AttributeArray::Scalars {
                values: vec![1.0f32, 2.0],
                num_components: 1,
            },
        );
        grid.point_data.insert(
            "B".to_string(),
            AttributeArray::Scalars {
                values: vec![3.0f32, 4.0, 5.0, 6.0],
                num_components: 2,
            },
        );

        let bytes =
            write_vti_binary_appended_bytes(&grid).expect("write must succeed on valid grid");

        // Extract the XML header (bytes strictly before the '_' marker).
        let ad_start = bytes
            .windows(b"<AppendedData".len())
            .position(|w| w == b"<AppendedData")
            .expect("<AppendedData tag must be present");
        let gt_rel = bytes[ad_start..]
            .iter()
            .position(|&b| b == b'>')
            .expect("<AppendedData tag must close");
        let us_rel = bytes[ad_start + gt_rel + 1..]
            .iter()
            .position(|&b| b == b'_')
            .expect("'_' marker must be present");
        let underscore_abs = ad_start + gt_rel + 1 + us_rel;

        let header =
            std::str::from_utf8(&bytes[..underscore_abs]).expect("header must be valid UTF-8");

        // "A" (first alphabetically) must carry offset=0.
        assert!(
            header.contains("offset=\"0\""),
            "array 'A' must have offset=0; header:\n{}",
            header
        );
        // "B" (second alphabetically) must carry offset=12 = 4 (uint32 count) + 2*4 (two f32s).
        assert!(
            header.contains("offset=\"12\""),
            "array 'B' must have offset=12 (=4+2*4); header:\n{}",
            header
        );
    }

    // ── 14 ────────────────────────────────────────────────────────────────────
    /// Invariant: a CellData-only binary-appended grid round-trips to an
    /// identical grid with no PointData and exactly the original CellData values.
    #[test]
    fn test_write_vti_binary_appended_cell_data_only_roundtrip() {
        use crate::format::vtk::image_xml::reader::read_vti_binary_appended_bytes;

        // extent [0,1,0,1,0,1] → n_cells = 1×1×1 = 1; n_points = 2×2×2 = 8
        let grid = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            point_data: std::collections::HashMap::new(),
            cell_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "pressure".to_string(),
                    AttributeArray::Scalars {
                        values: vec![42.0f32],
                        num_components: 1,
                    },
                );
                m
            },
        };

        let bytes = write_vti_binary_appended_bytes(&grid)
            .expect("write_vti_binary_appended_bytes must succeed on valid cell-data-only grid");
        let parsed = read_vti_binary_appended_bytes(&bytes)
            .expect("read_vti_binary_appended_bytes must succeed on cell-data-only bytes");

        assert_eq!(
            parsed.whole_extent, grid.whole_extent,
            "whole_extent must match exactly"
        );
        assert!(
            parsed.point_data.is_empty(),
            "cell-only grid must have no PointData after parse"
        );
        assert!(
            parsed.cell_data.contains_key("pressure"),
            "must have 'pressure' CellData key"
        );
        let values = match parsed.cell_data.get("pressure").unwrap() {
            AttributeArray::Scalars { values, .. } => values.clone(),
            other => panic!("expected Scalars variant for 'pressure', got {:?}", other),
        };
        assert_eq!(values.len(), 1, "pressure CellData must have 1 value");
        assert!(
            (values[0] - 42.0f32).abs() < 1e-6,
            "pressure[0]: expected 42.0, got {} (diff {})",
            values[0],
            (values[0] - 42.0f32).abs()
        );
    }

    // ── 15 ────────────────────────────────────────────────────────────────────
    /// Invariant: both PointData and CellData survive binary-appended round-trip
    /// simultaneously with all values preserved within f32 representation error.
    #[test]
    fn test_write_vti_binary_appended_mixed_point_and_cell_data() {
        use crate::format::vtk::image_xml::reader::read_vti_binary_appended_bytes;

        // extent [0,1,0,1,0,1] → n_points = 8, n_cells = 1
        let grid = VtkImageData {
            whole_extent: [0, 1, 0, 1, 0, 1],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            point_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "density".to_string(),
                    AttributeArray::Scalars {
                        values: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        num_components: 1,
                    },
                );
                m
            },
            cell_data: {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    "material".to_string(),
                    AttributeArray::Scalars {
                        values: vec![7.0f32],
                        num_components: 1,
                    },
                );
                m
            },
        };

        let bytes = write_vti_binary_appended_bytes(&grid)
            .expect("write_vti_binary_appended_bytes must succeed on mixed grid");
        let parsed = read_vti_binary_appended_bytes(&bytes)
            .expect("read_vti_binary_appended_bytes must succeed on mixed bytes");

        assert_eq!(
            parsed.whole_extent, grid.whole_extent,
            "whole_extent must match exactly"
        );
        assert!(
            parsed.point_data.contains_key("density"),
            "parsed must contain 'density' PointData key"
        );
        let pd_vals = match parsed.point_data.get("density").unwrap() {
            AttributeArray::Scalars { values, .. } => values.clone(),
            other => panic!("expected Scalars for 'density', got {:?}", other),
        };
        assert_eq!(pd_vals.len(), 8, "density must have 8 values");
        let expected_density = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        for (i, (&got, &exp)) in pd_vals.iter().zip(expected_density.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "density[{}]: expected {}, got {} (diff {})",
                i,
                exp,
                got,
                (got - exp).abs()
            );
        }
        assert!(
            parsed.cell_data.contains_key("material"),
            "parsed must contain 'material' CellData key"
        );
        let cd_vals = match parsed.cell_data.get("material").unwrap() {
            AttributeArray::Scalars { values, .. } => values.clone(),
            other => panic!("expected Scalars for 'material', got {:?}", other),
        };
        assert_eq!(cd_vals.len(), 1, "material must have 1 value");
        assert!(
            (cd_vals[0] - 7.0f32).abs() < 1e-6,
            "material[0]: expected 7.0, got {} (diff {})",
            cd_vals[0],
            (cd_vals[0] - 7.0f32).abs()
        );
    }

    // ── 16 ────────────────────────────────────────────────────────────────────
    /// Invariant: when both PointData and CellData are present, the CellData
    /// array's offset equals the total byte size of all PointData blocks.
    ///
    /// Proof: offset[cd_0] = Σ(4 + n_values[pd_i] * 4) for all pd_i.
    /// For "pd" (2 float32s): block = 4 + 2*4 = 12 bytes → CellData offset = 12.
    #[test]
    fn test_write_vti_binary_appended_cell_data_offset_after_point_data() {
        // extent [0,1,0,0,0,0] → n_points = 2×1×1 = 2; n_cells = 1×1×1 = 1
        // PointData block: 4 (uint32 count) + 2*4 (two f32s) = 12 bytes → CellData offset = 12
        let mut grid = VtkImageData {
            whole_extent: [0, 1, 0, 0, 0, 0],
            origin: [0.0, 0.0, 0.0],
            spacing: [1.0, 1.0, 1.0],
            ..Default::default()
        };
        grid.point_data.insert(
            "pd".to_string(),
            AttributeArray::Scalars {
                values: vec![1.0f32, 2.0],
                num_components: 1,
            },
        );
        grid.cell_data.insert(
            "cd".to_string(),
            AttributeArray::Scalars {
                values: vec![9.0f32],
                num_components: 1,
            },
        );

        let bytes =
            write_vti_binary_appended_bytes(&grid).expect("write must succeed on valid grid");

        let ad_start = bytes
            .windows(b"<AppendedData".len())
            .position(|w| w == b"<AppendedData")
            .expect("<AppendedData tag must be present");
        let gt_rel = bytes[ad_start..]
            .iter()
            .position(|&b| b == b'>')
            .expect("<AppendedData tag must close");
        let us_rel = bytes[ad_start + gt_rel + 1..]
            .iter()
            .position(|&b| b == b'_')
            .expect("'_' marker must be present");
        let underscore_abs = ad_start + gt_rel + 1 + us_rel;

        let header =
            std::str::from_utf8(&bytes[..underscore_abs]).expect("header must be valid UTF-8");

        assert!(
            header.contains("Name=\"pd\""),
            "header must contain Name=\"pd\"; header:\n{}",
            header
        );
        assert!(
            header.contains("Name=\"cd\""),
            "header must contain Name=\"cd\"; header:\n{}",
            header
        );
        // "pd" is the only PointData array → its offset must be 0.
        assert!(
            header.contains("offset=\"0\""),
            "PointData array 'pd' must have offset=0; header:\n{}",
            header
        );
        // "cd" starts after the single PointData block: 4 (uint32) + 2*4 = 12 bytes.
        assert!(
            header.contains("offset=\"12\""),
            "CellData array 'cd' must have offset=12 (=4+2*4); header:\n{}",
            header
        );
    }
}
