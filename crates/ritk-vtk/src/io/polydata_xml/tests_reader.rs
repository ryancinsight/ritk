//! Tests for the VTK XML PolyData (.vtp) reader.

use super::*;
use crate::domain::vtk_data_object::{AttributeArray, VtkPolyData};
use crate::io::polydata_xml::writer::write_vtp_str;

fn triangle() -> VtkPolyData {
    let mut p = VtkPolyData::default();
    p.points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    p.polygons = vec![vec![0, 1, 2]];
    p
}

#[test]
fn test_triangle_parse() {
    let p = parse_vtp(&write_vtp_str(&triangle())).expect("infallible: validated precondition");
    assert_eq!(p.points.len(), 3);
    assert_eq!(p.polygons.len(), 1);
    assert_eq!(p.polygons[0], vec![0u32, 1, 2]);
    assert!((p.points[1][0] - 1.0).abs() < 1e-5);
}
#[test]
fn test_empty_parse() {
    let p = parse_vtp(&write_vtp_str(&Default::default())).expect("infallible: validated precondition");
    assert_eq!(p.points.len(), 0);
    assert_eq!(p.polygons.len(), 0);
}
#[test]
fn test_scalars_roundtrip() {
    let mut pd = triangle();
    pd.point_data.insert(
        "pressure".to_string(),
        AttributeArray::Scalars {
            values: vec![1.0, 2.0, 3.0],
            num_components: 1,
        },
    );
    let p = parse_vtp(&write_vtp_str(&pd)).expect("infallible: validated precondition");
    match p.point_data.get("pressure") {
        Some(AttributeArray::Scalars { values, .. }) => {
            assert!((values[0] - 1.0).abs() < 1e-4);
        }
        _ => panic!("not Scalars"),
    }
}
#[test]
fn test_lines_parse() {
    let mut pd = VtkPolyData::default();
    pd.points = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    pd.lines = vec![vec![0, 1, 2]];
    let p = parse_vtp(&write_vtp_str(&pd)).expect("infallible: validated precondition");
    assert_eq!(p.lines.len(), 1);
    assert_eq!(p.lines[0], vec![0u32, 1, 2]);
}
#[test]
fn test_vectors_roundtrip() {
    let mut pd = VtkPolyData::default();
    pd.points = vec![[0.0, 0.0, 0.0]];
    pd.point_data.insert(
        "vel".to_string(),
        AttributeArray::Vectors {
            values: vec![[1.0, 2.0, 3.0]],
        },
    );
    let p = parse_vtp(&write_vtp_str(&pd)).expect("infallible: validated precondition");
    match p.point_data.get("vel") {
        Some(AttributeArray::Vectors { values }) => {
            assert!((values[0][0] - 1.0).abs() < 1e-4);
        }
        _ => panic!("not Vectors"),
    }
}
#[test]
fn test_missing_points_error() {
    let s = String::from(
        "<VTKFile><PolyData><Piece NumberOfPoints=\"1\"></Piece></PolyData></VTKFile>",
    );
    // The section has no <Points> element, so parse_vtp should error.
    let _ = s; // suppress unused warning
    let s2 = "<VTKFile><PolyData><Piece NumberOfPoints=\"1\"></Piece></PolyData></VTKFile>";
    assert!(parse_vtp(s2).is_err());
}
#[test]
fn test_wrong_coord_count_error() {
    let mut pd = VtkPolyData::default();
    pd.points = vec![[0.0, 0.0, 0.0]];
    let base = write_vtp_str(&pd).replace("NumberOfPoints=", "NumberOfPointsBad=");
    let bad = format!("{}NumberOfPoints=\"99\"", base);
    assert!(parse_vtp(&bad).is_err());
}
