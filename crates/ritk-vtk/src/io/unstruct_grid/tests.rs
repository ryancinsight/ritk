#![allow(clippy::needless_range_loop)]

use super::*;

use crate::domain::vtk_data_object::{AttributeArray, VtkCellType, VtkUnstructuredGrid};
use tempfile::NamedTempFile;

#[test]
fn test_unstructured_grid_roundtrip_tetrahedra() {
    let mut grid = VtkUnstructuredGrid::new();
    grid.points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    grid.cells = vec![vec![0u32, 1, 2, 3]];
    grid.cell_types = vec![VtkCellType::Tetra];
    grid.point_data.insert(
        "pressure".to_string(),
        AttributeArray::Scalars {
            values: vec![0.0, 1.0, 2.0, 3.0],
            num_components: 1,
        },
    );
    grid.cell_data.insert(
        "stress".to_string(),
        AttributeArray::Scalars {
            values: vec![42.0],
            num_components: 1,
        },
    );

    let tmp = NamedTempFile::new().expect("temp");
    write_vtk_unstructured_grid(tmp.path(), &grid).expect("write");
    let r = read_vtk_unstructured_grid(tmp.path()).expect("read");

    assert_eq!(r.n_points(), 4);
    assert_eq!(r.n_cells(), 1);
    assert_eq!(r.cells[0], vec![0u32, 1, 2, 3]);
    assert_eq!(r.cell_types[0], VtkCellType::Tetra);
    assert_eq!(u8::from(r.cell_types[0]), 10);

    match r.point_data.get("pressure").expect("pressure") {
        AttributeArray::Scalars { values, .. } => {
            assert_eq!(values.len(), 4);
            for i in 0..4 {
                assert!(
                    (values[i] - i as f32).abs() < 1e-5,
                    "pressure[{}]: exp {} got {}",
                    i,
                    i as f32,
                    values[i]
                );
            }
        }
        other => panic!("expected Scalars: {:?}", other),
    }

    match r.cell_data.get("stress").expect("stress") {
        AttributeArray::Scalars { values, .. } => {
            assert_eq!(values.len(), 1);
            assert!(
                (values[0] - 42.0).abs() < 1e-5,
                "stress[0]: exp 42.0 got {}",
                values[0]
            );
        }
        other => panic!("expected Scalars: {:?}", other),
    }
}

#[test]
fn test_unstructured_grid_roundtrip_multiple_cells() {
    let mut grid = VtkUnstructuredGrid::new();
    grid.points = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [1.5, 1.0, 0.0],
    ];
    grid.cells = vec![vec![0u32, 1, 2], vec![1u32, 2, 3]];
    grid.cell_types = vec![VtkCellType::Triangle, VtkCellType::Triangle];

    let tmp = NamedTempFile::new().expect("temp");
    write_vtk_unstructured_grid(tmp.path(), &grid).expect("write");
    let r = read_vtk_unstructured_grid(tmp.path()).expect("read");

    assert_eq!(r.n_cells(), 2);
    assert_eq!(r.cells[0], vec![0u32, 1, 2]);
    assert_eq!(r.cells[1], vec![1u32, 2, 3]);
    assert_eq!(r.cell_types[0], VtkCellType::Triangle);
    assert_eq!(r.cell_types[1], VtkCellType::Triangle);
    assert_eq!(u8::from(r.cell_types[0]), 5);
}

#[test]
fn test_unstructured_grid_validate_rejects_wrong_types_count() {
    let mut grid = VtkUnstructuredGrid::new();
    grid.points = vec![[0.0f32; 3]; 4];
    grid.cells = vec![vec![0u32, 1, 2], vec![1u32, 2, 3]];
    grid.cell_types = vec![VtkCellType::Triangle]; // 2 cells, 1 type

    let tmp = NamedTempFile::new().expect("temp");
    assert!(write_vtk_unstructured_grid(tmp.path(), &grid).is_err());
}

#[test]
fn test_unstructured_grid_validate_rejects_out_of_range_index() {
    let mut grid = VtkUnstructuredGrid::new();
    grid.points = vec![[0.0f32; 3]; 3];
    grid.cells = vec![vec![0u32, 1, 99]]; // index 99 >= n_points=3
    grid.cell_types = vec![VtkCellType::Triangle];

    let tmp = NamedTempFile::new().expect("temp");
    assert!(write_vtk_unstructured_grid(tmp.path(), &grid).is_err());
}
