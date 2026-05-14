use super::*;

fn triangle() -> VtkPolyData {
    VtkPolyData {
        points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]],
        polygons: vec![vec![0, 1, 2]],
        ..Default::default()
    }
}

#[test]
fn test_vtk_poly_data_default_empty() {
    let p = VtkPolyData::default();
    assert!(p.points.is_empty());
    assert!(p.polygons.is_empty());
    assert_eq!(p.num_cells(), 0);
    assert!(p.validate().is_ok());
}

#[test]
fn test_vtk_poly_data_validate_ok() {
    assert_eq!(triangle().validate(), Ok(()));
}

#[test]
fn test_vtk_poly_data_validate_out_of_range() {
    let mut p = triangle();
    p.polygons[0].push(99); // index 99 does not exist
    assert!(p.validate().is_err());
    let msg = p.validate().unwrap_err();
    assert!(msg.contains("POLYGONS"), "error must name the cell type");
}

#[test]
fn test_vtk_poly_data_validate_scalar_length() {
    let mut p = triangle();
    // n_points = 3, ncomp = 1, expected length = 3; supply 2 -> error.
    p.point_data.insert(
        "intensity".to_string(),
        AttributeArray::Scalars {
            values: vec![1.0, 2.0],
            num_components: 1,
        },
    );
    let result = p.validate();
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("intensity"));
}

#[test]
fn test_vtk_data_object_polydata_wraps() {
    let obj = VtkDataObject::PolyData(triangle());
    match obj {
        VtkDataObject::PolyData(p) => {
            assert_eq!(p.points.len(), 3);
            assert_eq!(p.polygons.len(), 1);
        }
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_vtk_poly_data_num_cells_multi_type() {
    let p = VtkPolyData {
        points: vec![[0.0; 3], [1.0; 3], [2.0; 3]],
        vertices: vec![vec![0], vec![1]],
        lines: vec![vec![0, 1]],
        polygons: vec![vec![0, 1, 2]],
        triangle_strips: vec![vec![0, 1, 2]],
        ..Default::default()
    };
    assert_eq!(p.num_cells(), 5);
}

#[test]
fn test_vtk_poly_data_validate_cell_data_scalar_ok() {
    let mut p = VtkPolyData {
        points: vec![[0.0; 3], [1.0; 3], [2.0; 3]],
        polygons: vec![vec![0, 1, 2]], // 1 cell
        ..Default::default()
    };
    p.cell_data.insert(
        "pressure".to_string(),
        AttributeArray::Scalars {
            values: vec![42.0],
            num_components: 1,
        },
    );
    assert!(p.validate().is_ok());
}

#[test]
fn test_attribute_array_equality() {
    let a = AttributeArray::Scalars {
        values: vec![1.0, 2.0],
        num_components: 1,
    };
    let b = AttributeArray::Scalars {
        values: vec![1.0, 2.0],
        num_components: 1,
    };
    assert_eq!(a, b);
}

#[test]
fn test_vtk_structured_grid_n_points() {
    assert_eq!(VtkStructuredGrid::new([2, 3, 4]).n_points(), 24);
}

#[test]
fn test_vtk_structured_grid_n_cells() {
    assert_eq!(VtkStructuredGrid::new([2, 3, 4]).n_cells(), 6);
}

#[test]
fn test_vtk_structured_grid_validate_ok() {
    let mut g = VtkStructuredGrid::new([2, 2, 2]);
    g.points = vec![[0.0; 3]; 8];
    assert!(g.validate().is_ok());
}

#[test]
fn test_vtk_structured_grid_validate_wrong_len() {
    let mut g = VtkStructuredGrid::new([2, 2, 2]);
    g.points = vec![[0.0; 3]; 5];
    assert!(g.validate().unwrap_err().contains("n_points"));
}

#[test]
fn test_vtk_unstructured_grid_validate_ok() {
    let mut g = VtkUnstructuredGrid::new();
    g.points = vec![[0.0; 3]; 4];
    g.cells = vec![vec![0, 1, 2, 3]];
    g.cell_types = vec![VtkCellType::Tetra];
    assert!(g.validate().is_ok());
}

#[test]
fn test_vtk_unstructured_grid_cell_type_mismatch() {
    let mut g = VtkUnstructuredGrid::new();
    g.cells = vec![vec![0], vec![1]];
    g.cell_types = vec![VtkCellType::Triangle];
    assert!(g.validate().unwrap_err().contains("cell_types"));
}

#[test]
fn test_vtk_unstructured_grid_index_out_of_range() {
    let mut g = VtkUnstructuredGrid::new();
    g.points = vec![[0.0; 3]; 3];
    g.cells = vec![vec![0, 1, 99]];
    g.cell_types = vec![VtkCellType::Triangle];
    assert!(g.validate().unwrap_err().contains("99"));
}

#[test]
fn test_vtk_data_object_structured_variant() {
    match VtkDataObject::StructuredGrid(VtkStructuredGrid::new([2, 2, 2])) {
        VtkDataObject::StructuredGrid(g) => assert_eq!(g.n_points(), 8),
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_vtk_data_object_unstructured_variant() {
    let mut g = VtkUnstructuredGrid::new();
    g.points = vec![[1.0, 0.0, 0.0]];
    match VtkDataObject::UnstructuredGrid(g) {
        VtkDataObject::UnstructuredGrid(g) => assert_eq!(g.n_points(), 1),
        _ => panic!("wrong variant"),
    }
}

#[test]
fn test_vtk_unstructured_grid_default_is_empty() {
    let g = VtkUnstructuredGrid::default();
    assert_eq!(g.n_points(), 0);
    assert_eq!(g.n_cells(), 0);
    assert!(g.validate().is_ok());
}

#[test]
fn test_vtk_cell_type_round_trip_all_known() {
    let known: &[(u8, VtkCellType)] = &[
        (1, VtkCellType::Vertex),
        (5, VtkCellType::Triangle),
        (10, VtkCellType::Tetra),
        (12, VtkCellType::Hexahedron),
        (21, VtkCellType::QuadraticEdge),
        (34, VtkCellType::BilinearQuadraticWedge),
    ];
    for &(code, variant) in known {
        assert_eq!(
            VtkCellType::from_u8(code),
            Some(variant),
            "from_u8({code}) must return Some({variant:?})"
        );
        assert_eq!(
            variant.to_u8(),
            code,
            "{variant:?}.to_u8() must return {code}"
        );
    }
}

#[test]
fn test_vtk_cell_type_unknown_returns_none() {
    for v in [0u8, 17, 18, 19, 20, 35, 200, 255] {
        assert_eq!(
            VtkCellType::from_u8(v),
            None,
            "from_u8({v}) must return None for unknown code"
        );
    }
}

#[test]
fn test_vtk_image_data_n_points_and_cells() {
    // WholeExtent "0 2 0 3 0 4":
    // n_points = (2-0+1) * (3-0+1) * (4-0+1) = 3 * 4 * 5 = 60
    // n_cells  = max(1,2) * max(1,3) * max(1,4) = 2 * 3 * 4 = 24
    let img = VtkImageData {
        whole_extent: [0, 2, 0, 3, 0, 4],
        ..Default::default()
    };
    assert_eq!(img.n_points(), 60);
    assert_eq!(img.n_cells(), 24);
}

#[test]
fn test_vtk_image_data_validate_ok() {
    // extent [0,1,0,1,0,1] → n_points = 2*2*2 = 8
    let mut img = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        ..Default::default()
    };
    img.point_data.insert(
        "s".to_string(),
        AttributeArray::Scalars {
            values: vec![0.0f32; 8],
            num_components: 1,
        },
    );
    assert_eq!(img.validate(), Ok(()));
}

#[test]
fn test_vtk_image_data_validate_wrong_scalar_len() {
    // extent [0,1,0,1,0,1] → n_points = 8; supplying 5 → Err
    let mut img = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        ..Default::default()
    };
    img.point_data.insert(
        "s".to_string(),
        AttributeArray::Scalars {
            values: vec![0.0f32; 5],
            num_components: 1,
        },
    );
    let r = img.validate();
    assert!(r.is_err(), "mismatched scalar length must return Err");
    assert!(
        r.unwrap_err().contains("s"),
        "error message must name the field"
    );
}

#[test]
fn test_vtk_image_data_data_object_variant() {
    // extent [0,2,0,3,0,4] → n_points = 60
    let img = VtkImageData {
        whole_extent: [0, 2, 0, 3, 0, 4],
        ..Default::default()
    };
    match VtkDataObject::ImageData(img) {
        VtkDataObject::ImageData(g) => assert_eq!(g.n_points(), 60),
        _ => panic!("wrong VtkDataObject variant"),
    }
}
