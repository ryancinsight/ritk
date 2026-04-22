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
    if cells.is_empty() { return Ok(()); }
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
        AttributeArray::Scalars { values, num_components } => {
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
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
    use crate::format::vtk::polydata::reader::parse_polydata;
    use std::io::Cursor;
    use tempfile::NamedTempFile;

    fn round_trip(poly: &VtkPolyData) -> VtkPolyData {
        let mut buf = Vec::new();
        write_polydata(&mut buf, poly).unwrap();
        let mut cursor = Cursor::new(buf);
        parse_polydata(&mut cursor).unwrap()
    }

    #[test]
    fn test_write_ascii_triangle() {
        let poly = VtkPolyData {
            points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            polygons: vec![vec![0, 1, 2]],
            ..Default::default()
        };
        let result = round_trip(&poly);
        assert_eq!(result.points.len(), 3);
        assert!((result.points[1][0] - 1.0).abs() < 1e-5);
        assert!((result.points[2][1] - 1.0).abs() < 1e-5);
        assert_eq!(result.polygons, vec![vec![0u32, 1, 2]]);
    }

    #[test]
    fn test_write_empty_polydata() {
        let poly = VtkPolyData::default();
        let result = round_trip(&poly);
        assert_eq!(result.points.len(), 0);
        assert_eq!(result.num_cells(), 0);
    }

    #[test]
    fn test_write_with_point_data_scalars() {
        let mut poly = VtkPolyData {
            points: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            polygons: vec![vec![0, 1, 2]],
            ..Default::default()
        };
        poly.point_data.insert(
            "temperature".to_string(),
            AttributeArray::Scalars { values: vec![36.0, 37.0, 38.0], num_components: 1 },
        );
        let result = round_trip(&poly);
        match result.point_data.get("temperature").unwrap() {
            AttributeArray::Scalars { values, .. } => {
                assert!((values[0] - 36.0).abs() < 1e-5);
                assert!((values[1] - 37.0).abs() < 1e-5);
                assert!((values[2] - 38.0).abs() < 1e-5);
            }
            _ => panic!("expected Scalars"),
        }
    }

    #[test]
    fn test_write_preserves_all_cell_types() {
        let poly = VtkPolyData {
            points: vec![[0.0; 3], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            vertices: vec![vec![0]],
            lines: vec![vec![0, 1]],
            polygons: vec![vec![0, 1, 2]],
            triangle_strips: vec![vec![0, 1, 2, 3]],
            ..Default::default()
        };
        let result = round_trip(&poly);
        assert_eq!(result.vertices.len(), 1);
        assert_eq!(result.lines.len(), 1);
        assert_eq!(result.polygons.len(), 1);
        assert_eq!(result.triangle_strips.len(), 1);
        assert_eq!(result.vertices[0], vec![0u32]);
        assert_eq!(result.lines[0], vec![0u32, 1]);
        assert_eq!(result.polygons[0], vec![0u32, 1, 2]);
        assert_eq!(result.triangle_strips[0], vec![0u32, 1, 2, 3]);
    }

    #[test]
    fn test_write_error_bad_path() {
        let poly = VtkPolyData::default();
        let result = write_vtk_polydata("/nonexistent_dir/output.vtk", &poly);
        assert!(result.is_err(), "write to nonexistent path must fail");
    }

    #[test]
    fn test_roundtrip_validate() {
        let poly = VtkPolyData {
            points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
            polygons: vec![vec![0, 1, 2]],
            ..Default::default()
        };
        let tmp = NamedTempFile::new().unwrap();
        write_vtk_polydata(tmp.path(), &poly).unwrap();
        let result =
            crate::format::vtk::polydata::reader::read_vtk_polydata(tmp.path()).unwrap();
        assert!(
            result.validate().is_ok(),
            "round-trip result must satisfy VtkPolyData::validate()"
        );
    }

    #[test]
    fn test_write_vectors_round_trip() {
        let mut poly = VtkPolyData {
            points: vec![[0.0; 3], [1.0, 0.0, 0.0]],
            lines: vec![vec![0, 1]],
            ..Default::default()
        };
        poly.point_data.insert(
            "velocity".to_string(),
            AttributeArray::Vectors { values: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] },
        );
        let result = round_trip(&poly);
        match result.point_data.get("velocity").unwrap() {
            AttributeArray::Vectors { values } => {
                assert_eq!(values.len(), 2);
                assert!((values[0][0] - 1.0).abs() < 1e-5);
                assert!((values[1][1] - 1.0).abs() < 1e-5);
            }
            _ => panic!("expected Vectors"),
        }
    }
}
