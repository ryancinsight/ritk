use crate::domain::vtk_data_object::{AttributeArray, VtkCellType, VtkUnstructuredGrid};
use crate::io::unstructured_xml::reader::parse_vtu;
use crate::io::unstructured_xml::writer::write_vtu_str;

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
fn test_tetra_roundtrip() {
    let g = tetra();
    let s = write_vtu_str(&g);
    let r = parse_vtu(&s).expect("parse must succeed");
    assert_eq!(r.n_points(), 4);
    assert_eq!(r.n_cells(), 1);
    assert_eq!(r.cells[0], vec![0u32, 1, 2, 3]);
    assert_eq!(r.cell_types[0], VtkCellType::Tetra);
    assert_eq!(r.cell_types[0].to_u8(), 10, "Tetra VTK code must be 10");
    assert!(
        (r.points[0][0] - 0.0).abs() < 1e-5,
        "p[0].x = {}",
        r.points[0][0]
    );
    assert!(
        (r.points[1][0] - 1.0).abs() < 1e-5,
        "p[1].x = {}",
        r.points[1][0]
    );
    assert!(
        (r.points[3][2] - 1.0).abs() < 1e-5,
        "p[3].z = {}",
        r.points[3][2]
    );
}

#[test]
fn test_empty_grid_roundtrip() {
    let g = VtkUnstructuredGrid::default();
    let s = write_vtu_str(&g);
    let r = parse_vtu(&s).expect("empty grid must parse");
    assert_eq!(r.n_points(), 0);
    assert_eq!(r.n_cells(), 0);
    assert!(r.cells.is_empty());
    assert!(r.cell_types.is_empty());
    assert!(r.point_data.is_empty());
    assert!(r.cell_data.is_empty());
}

#[test]
fn test_two_triangles_roundtrip() {
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
    let r = parse_vtu(&s).expect("two triangles must parse");
    assert_eq!(r.n_cells(), 2);
    assert_eq!(r.cells[0], vec![0u32, 1, 2], "cell 0 connectivity");
    assert_eq!(r.cells[1], vec![1u32, 2, 3], "cell 1 connectivity");
    assert_eq!(r.cell_types[0], VtkCellType::Triangle);
    assert_eq!(r.cell_types[1], VtkCellType::Triangle);
    assert_eq!(r.cell_types[0].to_u8(), 5, "Triangle VTK code must be 5");
}

#[test]
fn test_point_data_scalars_roundtrip() {
    let mut g = tetra();
    g.point_data.insert(
        "pressure".to_string(),
        AttributeArray::Scalars {
            values: vec![1.0, 2.0, 3.0, 4.0],
            num_components: 1,
        },
    );
    let r = parse_vtu(&write_vtu_str(&g)).expect("scalars roundtrip");
    match r.point_data.get("pressure").expect("pressure attr") {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
            assert_eq!(*num_components, 1);
            assert_eq!(values.len(), 4);
            assert!((values[0] - 1.0).abs() < 1e-5, "values[0] = {}", values[0]);
            assert!((values[3] - 4.0).abs() < 1e-5, "values[3] = {}", values[3]);
        }
        other => panic!("expected Scalars, got {:?}", other),
    }
}

#[test]
fn test_cell_data_scalars_roundtrip() {
    let mut g = tetra();
    g.cell_data.insert(
        "stress".to_string(),
        AttributeArray::Scalars {
            values: vec![42.0],
            num_components: 1,
        },
    );
    let r = parse_vtu(&write_vtu_str(&g)).expect("cell scalars roundtrip");
    match r.cell_data.get("stress").expect("stress attr") {
        AttributeArray::Scalars { values, .. } => {
            assert_eq!(values.len(), 1);
            assert!((values[0] - 42.0).abs() < 1e-5, "values[0] = {}", values[0]);
        }
        other => panic!("expected Scalars, got {:?}", other),
    }
}

#[test]
fn test_vectors_roundtrip() {
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
    let r = parse_vtu(&write_vtu_str(&g)).expect("vectors roundtrip");
    match r.point_data.get("vel").expect("vel attr") {
        AttributeArray::Vectors { values } => {
            assert_eq!(values.len(), 1);
            assert!((values[0][0] - 1.0).abs() < 1e-5, "v.x = {}", values[0][0]);
            assert!((values[0][1] - 2.0).abs() < 1e-5, "v.y = {}", values[0][1]);
            assert!((values[0][2] - 3.0).abs() < 1e-5, "v.z = {}", values[0][2]);
        }
        other => panic!("expected Vectors, got {:?}", other),
    }
}

#[test]
fn test_normals_roundtrip() {
    let mut g = VtkUnstructuredGrid::new();
    g.points = vec![[0.0, 0.0, 0.0]];
    g.cells = vec![vec![0u32]];
    g.cell_types = vec![VtkCellType::Vertex];
    g.point_data.insert(
        "normals".to_string(),
        AttributeArray::Normals {
            values: vec![[0.0, 0.0, 1.0]],
        },
    );
    let r = parse_vtu(&write_vtu_str(&g)).expect("normals roundtrip");
    match r.point_data.get("normals").expect("normals attr") {
        AttributeArray::Normals { values } => {
            assert_eq!(values.len(), 1);
            assert!((values[0][2] - 1.0).abs() < 1e-5, "n.z = {}", values[0][2]);
        }
        other => panic!("expected Normals, got {:?}", other),
    }
}

#[test]
fn test_cell_types_variety_roundtrip() {
    // VTK codes: Vertex=1, Triangle=5, Tetra=10, Hexahedron=12.
    let mut g = VtkUnstructuredGrid::new();
    g.points = vec![[0.0; 3]; 8];
    g.cells = vec![
        vec![0u32],
        vec![0u32, 1, 2],
        vec![0u32, 1, 2, 3],
        vec![0u32, 1, 2, 3, 4, 5, 6, 7],
    ];
    g.cell_types = vec![
        VtkCellType::Vertex,
        VtkCellType::Triangle,
        VtkCellType::Tetra,
        VtkCellType::Hexahedron,
    ];
    let r = parse_vtu(&write_vtu_str(&g)).expect("variety roundtrip");
    assert_eq!(r.cell_types[0], VtkCellType::Vertex);
    assert_eq!(r.cell_types[1], VtkCellType::Triangle);
    assert_eq!(r.cell_types[2], VtkCellType::Tetra);
    assert_eq!(r.cell_types[3], VtkCellType::Hexahedron);
    // Canonical code verification.
    assert_eq!(r.cell_types[0].to_u8(), 1);
    assert_eq!(r.cell_types[1].to_u8(), 5);
    assert_eq!(r.cell_types[2].to_u8(), 10);
    assert_eq!(r.cell_types[3].to_u8(), 12);
}

#[test]
fn test_point_data_and_cell_data_both_present_roundtrip() {
    let mut g = tetra();
    g.point_data.insert(
        "temperature".to_string(),
        AttributeArray::Scalars {
            values: vec![100.0, 200.0, 300.0, 400.0],
            num_components: 1,
        },
    );
    g.cell_data.insert(
        "volume".to_string(),
        AttributeArray::Scalars {
            values: vec![0.1667],
            num_components: 1,
        },
    );
    let r = parse_vtu(&write_vtu_str(&g)).expect("dual attr roundtrip");
    match r.point_data.get("temperature").expect("temperature") {
        AttributeArray::Scalars { values, .. } => {
            assert_eq!(values.len(), 4);
            assert!((values[0] - 100.0).abs() < 1e-3);
            assert!((values[3] - 400.0).abs() < 1e-3);
        }
        other => panic!("expected Scalars, got {:?}", other),
    }
    match r.cell_data.get("volume").expect("volume") {
        AttributeArray::Scalars { values, .. } => {
            assert_eq!(values.len(), 1);
            assert!((values[0] - 0.1667).abs() < 1e-4, "volume = {}", values[0]);
        }
        other => panic!("expected Scalars, got {:?}", other),
    }
}

#[test]
fn test_single_vertex_cell_roundtrip() {
    // Single-point "vertex" cell: connectivity=[0], offsets=[1], types=[1].
    let mut g = VtkUnstructuredGrid::new();
    g.points = vec![[7.0, 8.0, 9.0]];
    g.cells = vec![vec![0u32]];
    g.cell_types = vec![VtkCellType::Vertex];
    let r = parse_vtu(&write_vtu_str(&g)).expect("single vertex roundtrip");
    assert_eq!(r.n_points(), 1);
    assert_eq!(r.n_cells(), 1);
    assert_eq!(r.cells[0], vec![0u32]);
    assert_eq!(r.cell_types[0], VtkCellType::Vertex);
    assert!((r.points[0][0] - 7.0).abs() < 1e-5);
    assert!((r.points[0][1] - 8.0).abs() < 1e-5);
    assert!((r.points[0][2] - 9.0).abs() < 1e-5);
}

#[test]
fn test_mixed_cell_sizes_connectivity_reconstruction() {
    // Mixing a triangle (3 pts) and a tetra (4 pts) — offsets [3, 7].
    let mut g = VtkUnstructuredGrid::new();
    g.points = vec![[0.0; 3]; 5];
    g.cells = vec![vec![0u32, 1, 2], vec![0u32, 1, 2, 3]];
    g.cell_types = vec![VtkCellType::Triangle, VtkCellType::Tetra];
    let r = parse_vtu(&write_vtu_str(&g)).expect("mixed cells roundtrip");
    assert_eq!(r.n_cells(), 2);
    assert_eq!(r.cells[0].len(), 3, "triangle has 3 indices");
    assert_eq!(r.cells[1].len(), 4, "tetra has 4 indices");
    assert_eq!(r.cells[0], vec![0u32, 1, 2]);
    assert_eq!(r.cells[1], vec![0u32, 1, 2, 3]);
}
