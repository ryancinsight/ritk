use super::*;
use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use crate::io::polydata::reader::parse_polydata;
use std::io::Cursor;
use tempfile::NamedTempFile;

fn round_trip(poly: &VtkPolyData) -> VtkPolyData {
    let mut buf = Vec::new();
    write_polydata(&mut buf, poly).expect("infallible: validated precondition");
    let mut cursor = Cursor::new(buf);
    parse_polydata(&mut cursor).expect("infallible: validated precondition")
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
        AttributeArray::Scalars {
            values: vec![36.0, 37.0, 38.0],
            num_components: 1,
        },
    );
    let result = round_trip(&poly);
    match result.point_data.get("temperature").expect("valid index") {
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
    let tmp = NamedTempFile::new().expect("infallible: validated precondition");
    write_vtk_polydata(tmp.path(), &poly).expect("infallible: validated precondition");
    let result = crate::io::polydata::reader::read_vtk_polydata(tmp.path()).expect("infallible: validated precondition");
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
        AttributeArray::Vectors {
            values: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        },
    );
    let result = round_trip(&poly);
    match result.point_data.get("velocity").expect("valid index") {
        AttributeArray::Vectors { values } => {
            assert_eq!(values.len(), 2);
            assert!((values[0][0] - 1.0).abs() < 1e-5);
            assert!((values[1][1] - 1.0).abs() < 1e-5);
        }
        _ => panic!("expected Vectors"),
    }
}
