//! VTK XML PolyData (.vtp) writer (ASCII inline format).

use crate::domain::vtk_data_object::VtkPolyData;
use crate::io::xml_write_attr::write_attr_xml;
use anyhow::{Context, Result};
use std::fmt::Write;
use std::path::Path;

pub fn write_vtp_polydata<P: AsRef<Path>>(path: P, poly: &VtkPolyData) -> Result<()> {
    std::fs::write(path.as_ref(), write_vtp_str(poly).as_bytes())
        .with_context(|| format!("cannot write VTP: {}", path.as_ref().display()))
}

pub(crate) fn write_vtp_str(poly: &VtkPolyData) -> String {
    let mut s = String::new();
    let np = poly.points.len();
    let nv = poly.vertices.len();
    let nl = poly.lines.len();
    let np2 = poly.polygons.len();
    let ns = poly.triangle_strips.len();
    writeln!(s, r#"<?xml version="1.0"?>"#).unwrap();
    writeln!(
        s,
        r#"<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">"#
    )
    .unwrap();
    writeln!(s, "  <PolyData>").unwrap();
    {
        let mut piece = String::from("    <Piece");
        for (k, v) in &[
            ("NumberOfPoints", np),
            ("NumberOfVerts", nv),
            ("NumberOfLines", nl),
            ("NumberOfPolys", np2),
            ("NumberOfStrips", ns),
        ] {
            piece.push(' ');
            piece.push_str(k);
            piece.push('=');
            piece.push('"');
            piece.push_str(&v.to_string());
            piece.push('"');
        }
        piece.push('>');
        writeln!(s, "{}", piece).unwrap();
    }
    writeln!(s, "      <Points>").unwrap();
    writeln!(
        s,
        r#"        <DataArray type="Float32" NumberOfComponents="3" format="ascii">"#
    )
    .unwrap();
    write!(s, "       ").unwrap();
    for [x, y, z] in &poly.points {
        write!(s, " {:.6} {:.6} {:.6}", x, y, z).unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "        </DataArray>").unwrap();
    writeln!(s, "      </Points>").unwrap();
    write_cells(&mut s, "Verts", &poly.vertices);
    write_cells(&mut s, "Lines", &poly.lines);
    write_cells(&mut s, "Polys", &poly.polygons);
    write_cells(&mut s, "Strips", &poly.triangle_strips);
    if !poly.point_data.is_empty() {
        writeln!(s, "      <PointData>").unwrap();
        for (name, attr) in &poly.point_data {
            write_attr_xml(&mut s, name, attr);
        }
        writeln!(s, "      </PointData>").unwrap();
    }
    if !poly.cell_data.is_empty() {
        writeln!(s, "      <CellData>").unwrap();
        for (name, attr) in &poly.cell_data {
            write_attr_xml(&mut s, name, attr);
        }
        writeln!(s, "      </CellData>").unwrap();
    }
    writeln!(s, "    </Piece>").unwrap();
    writeln!(s, "  </PolyData>").unwrap();
    writeln!(s, "</VTKFile>").unwrap();
    s
}

fn write_cells(s: &mut String, tag: &str, cells: &[Vec<u32>]) {
    let mut conn: Vec<u32> = Vec::new();
    let mut offs: Vec<u32> = Vec::new();
    let mut cum = 0u32;
    for cell in cells {
        conn.extend_from_slice(cell);
        cum += cell.len() as u32;
        offs.push(cum);
    }
    writeln!(s, "      <{}>", tag).unwrap();
    writeln!(
        s,
        r#"        <DataArray type="Int32" Name="connectivity" format="ascii">"#
    )
    .unwrap();
    write!(s, "       ").unwrap();
    for v in &conn {
        write!(s, " {}", v).unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "        </DataArray>").unwrap();
    writeln!(
        s,
        r#"        <DataArray type="Int32" Name="offsets" format="ascii">"#
    )
    .unwrap();
    write!(s, "       ").unwrap();
    for v in &offs {
        write!(s, " {}", v).unwrap();
    }
    writeln!(s).unwrap();
    writeln!(s, "        </DataArray>").unwrap();
    writeln!(s, "      </{}>", tag).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
    use crate::io::polydata_xml::reader::parse_vtp;

    fn triangle() -> VtkPolyData {
        let mut p = VtkPolyData::default();
        p.points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        p.polygons = vec![vec![0, 1, 2]];
        p
    }

    #[test]
    fn test_triangle_counts() {
        let s = write_vtp_str(&triangle());
        assert!(
            s.contains("NumberOfPoints=\"3\""),
            "missing NumberOfPoints=3 in output"
        );
        assert!(
            s.contains("NumberOfPolys=\"1\""),
            "missing NumberOfPolys=1 in output"
        );
    }
    #[test]
    fn test_roundtrip() {
        let orig = triangle();
        let s = write_vtp_str(&orig);
        let parsed = parse_vtp(&s).unwrap();
        assert_eq!(parsed.points.len(), 3);
        assert_eq!(parsed.polygons.len(), 1);
        assert_eq!(parsed.polygons[0], vec![0u32, 1, 2]);
        assert!((parsed.points[1][0] - 1.0).abs() < 1e-5);
    }
    #[test]
    fn test_empty_polydata() {
        let s = write_vtp_str(&VtkPolyData::default());
        let parsed = parse_vtp(&s).unwrap();
        assert_eq!(parsed.points.len(), 0);
        assert_eq!(parsed.polygons.len(), 0);
    }
    #[test]
    fn test_scalars_roundtrip() {
        let mut p = triangle();
        p.point_data.insert(
            "pres".to_string(),
            AttributeArray::Scalars {
                values: vec![1.0, 2.0, 3.0],
                num_components: 1 },
        );
        let parsed = parse_vtp(&write_vtp_str(&p)).unwrap();
        assert_eq!(parsed.point_data.len(), 1);
        if let Some(AttributeArray::Scalars { values, .. }) = parsed.point_data.get("pres") {
            assert!((values[0] - 1.0).abs() < 1e-4);
        } else {
            panic!("not Scalars");
        }
    }
    #[test]
    fn test_vectors_roundtrip() {
        let mut p = VtkPolyData::default();
        p.points = vec![[0.0, 0.0, 0.0]];
        p.point_data.insert(
            "vel".to_string(),
            AttributeArray::Vectors {
                values: vec![[1.0, 0.0, 0.0]] },
        );
        let parsed = parse_vtp(&write_vtp_str(&p)).unwrap();
        if let Some(AttributeArray::Vectors { values }) = parsed.point_data.get("vel") {
            assert!((values[0][0] - 1.0).abs() < 1e-4);
        } else {
            panic!("not Vectors");
        }
    }
    #[test]
    fn test_bad_path_error() {
        let result = write_vtp_polydata("/nonexistent_xyz/f.vtp", &VtkPolyData::default());
        assert!(result.is_err());
    }
}
