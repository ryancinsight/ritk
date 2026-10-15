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

#[test]
fn test_unstructured_grid_reader_rejects_bad_cell_index() {
    let input = b"# vtk DataFile Version 3.0
bad cell index
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 3 float
0 0 0 1 0 0 0 1 0
CELLS 1 4
3 0 not_an_index 2
CELL_TYPES 1
5
";
    let mut reader = std::io::Cursor::new(input.as_slice());
    let err = parse_unstructured_grid(&mut reader)
        .expect_err("malformed ASCII cell point index must return Err");

    let message = err.to_string();
    assert!(
        message.contains("bad CELLS point index"),
        "error must name malformed cell index: {message}"
    );
}

/// A CELLS header count larger than the file must not drive allocation.
///
/// `CELLS <n_cells> <size>` is an ASCII header line, so `n_cells` is fully
/// caller-controlled. Pre-reserving it let a ~200-byte file demand petabytes
/// before a single cell line was read — the length-field-driven allocation
/// class. The count is a claim, so the read must be bounded by the lines that
/// actually arrive.
///
/// The count is chosen so the pre-reservation cannot be satisfied even by an
/// overcommitting allocator: a smaller value merely succeeds and leaves the
/// defect invisible, since the loop then ends at the real end of the input for
/// the same reason an honest read would.
#[test]
fn hostile_cells_count_does_not_drive_allocation() {
    let input = "\
# vtk DataFile Version 3.0
hostile cells
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 3 float
0 0 0
1 0 0
0 1 0
CELLS 1000000000000000 4
3 0 1 2
";
    let mut reader = std::io::Cursor::new(input.as_bytes());
    let err = parse_unstructured_grid(&mut reader)
        .expect_err("a cell count the file cannot supply must error");

    let message = err.to_string();
    assert!(
        message.contains("EOF CELLS"),
        "the read must end at the real end of the input, got: {message}"
    );
}

/// A CELL_TYPES header count larger than the file must not drive allocation.
///
/// Mirrors the `CELLS` case for the `CELL_TYPES <n>` header line.
#[test]
fn hostile_cell_types_count_does_not_drive_allocation() {
    let input = "\
# vtk DataFile Version 3.0
hostile cell types
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 3 float
0 0 0
1 0 0
0 1 0
CELLS 1 4
3 0 1 2
CELL_TYPES 1000000000000
5
";
    let mut reader = std::io::Cursor::new(input.as_bytes());
    let err = parse_unstructured_grid(&mut reader)
        .expect_err("a cell-type count the file cannot supply must error");

    let message = err.to_string();
    assert!(
        message.contains("EOF CELL_TYPES"),
        "the read must end at the real end of the input, got: {message}"
    );
}

/// A POINTS count that overflows `usize` when tripled is rejected, not wrapped.
///
/// The coordinate count is `n_points * 3`; a header value near `usize::MAX`
/// would otherwise overflow before the bounded reader ever saw it.
#[test]
fn hostile_points_count_is_rejected_before_overflow() {
    let input = "\
# vtk DataFile Version 3.0
hostile points
ASCII
DATASET UNSTRUCTURED_GRID
POINTS 18446744073709551615 float
";
    let mut reader = std::io::Cursor::new(input.as_bytes());
    let err =
        parse_unstructured_grid(&mut reader).expect_err("an overflowing point count must error");

    let message = err.to_string();
    assert!(
        message.contains("overflows usize"),
        "the overflow must be reported, not wrapped, got: {message}"
    );
}
