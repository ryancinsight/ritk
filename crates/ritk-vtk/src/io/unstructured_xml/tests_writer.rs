//! Tests for the VTK XML UnstructuredGrid (.vtu) writer.

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
    assert!(
        s.contains("NumberOfPoints=\"4\""),
        "must contain NumberOfPoints=4"
    );
    assert!(
        s.contains("NumberOfCells=\"1\""),
        "must contain NumberOfCells=1"
    );
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
    assert!(
        s.contains("Name=\"offsets\""),
        "must have offsets DataArray"
    );
    assert!(s.contains(" 3"), "offsets must include 3");
    assert!(s.contains(" 6"), "offsets must include 6");
}

#[test]
fn test_write_vtu_str_cell_type_code() {
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
    assert!(
        s.contains("NumberOfPoints=\"0\""),
        "must contain NumberOfPoints=0"
    );
    assert!(
        s.contains("NumberOfCells=\"0\""),
        "must contain NumberOfCells=0"
    );
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
