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
}
