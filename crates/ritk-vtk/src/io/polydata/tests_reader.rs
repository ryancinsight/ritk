//! Tests for polydata reader.

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
    let poly = parse_str(TRIANGLE_ASCII).expect("infallible: validated precondition");
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
    let poly = parse_str(s).expect("infallible: validated precondition");
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
    let poly = parse_str(s).expect("infallible: validated precondition");
    let attr = poly.point_data.get("intensity").expect("valid index");
    match attr {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
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
    let poly = parse_str(s).expect("infallible: validated precondition");
    let attr = poly.cell_data.get("pressure").expect("valid index");
    match attr {
        AttributeArray::Scalars { values, .. } => {
            assert!(
                (values[0] - 42.5).abs() < 1e-5,
                "expected 42.5, got {}",
                values[0]
            );
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
    let poly = parse_str(s).expect("infallible: validated precondition");
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
    let poly = parse_str(s).expect("infallible: validated precondition");
    match poly.point_data.get("velocity").expect("valid index") {
        AttributeArray::Vectors { values } => {
            assert_eq!(values.len(), 2);
            assert!((values[0][0] - 1.0).abs() < 1e-6);
            assert!((values[1][1] - 1.0).abs() < 1e-6);
        }
        _ => panic!("expected Vectors"),
    }
    match poly.point_data.get("norm").expect("valid index") {
        AttributeArray::Normals { values } => {
            assert_eq!(values.len(), 2);
            assert!((values[0][2] - 1.0).abs() < 1e-6);
        }
        _ => panic!("expected Normals"),
    }
}
