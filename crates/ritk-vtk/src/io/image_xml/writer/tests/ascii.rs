use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use crate::io::image_xml::writer::{write_vti_image_data, write_vti_str};
use tempfile::NamedTempFile;

fn grid_2x2x2() -> VtkImageData {
    VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        origin: [1.0, 2.0, 3.0],
        spacing: [0.5, 0.5, 0.5],
        point_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "density".to_string(),
                AttributeArray::Scalars {
                    values: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    num_components: 1 },
            );
            m
        },
        cell_data: std::collections::HashMap::new() }
}

#[test]
fn test_write_vti_produces_vtkfile_header() {
    let s = write_vti_str(&VtkImageData::default());
    assert!(
        s.contains("<?xml version=\"1.0\"?>"),
        "must contain XML declaration"
    );
    assert!(s.contains("<VTKFile"), "must contain VTKFile element");
    assert!(
        s.contains("type=\"ImageData\""),
        "VTKFile type must be ImageData"
    );
    assert!(s.contains("</VTKFile>"), "must contain closing VTKFile tag");
}

#[test]
fn test_write_vti_extent_in_output() {
    let grid = VtkImageData {
        whole_extent: [0, 3, 0, 4, 0, 5],
        ..Default::default()
    };
    let s = write_vti_str(&grid);
    assert!(
        s.contains("WholeExtent=\"0 3 0 4 0 5\""),
        "WholeExtent attribute must be '0 3 0 4 0 5'; got:\n{}",
        s
    );
    assert!(
        s.contains("Piece Extent=\"0 3 0 4 0 5\""),
        "Piece Extent must match WholeExtent"
    );
}

#[test]
fn test_write_vti_origin_and_spacing() {
    let grid = VtkImageData {
        whole_extent: [0, 0, 0, 0, 0, 0],
        origin: [1.5, 2.25, 3.125],
        spacing: [0.1, 0.2, 0.4],
        ..Default::default()
    };
    let s = write_vti_str(&grid);
    assert!(
        s.contains("Origin=\"1.500000 2.250000 3.125000\""),
        "Origin must be formatted with 6 d.p.; got:\n{}",
        s
    );
    assert!(
        s.contains("Spacing=\"0.100000 0.200000 0.400000\""),
        "Spacing must be formatted with 6 d.p.; got:\n{}",
        s
    );
}

#[test]
fn test_write_vti_scalar_point_data() {
    let s = write_vti_str(&grid_2x2x2());
    assert!(s.contains("<PointData>"), "must emit PointData section");
    assert!(
        s.contains("Name=\"density\""),
        "must emit density DataArray"
    );
    for expected in &[
        "1.000000", "2.000000", "3.000000", "4.000000", "5.000000", "6.000000", "7.000000",
        "8.000000",
    ] {
        assert!(
            s.contains(expected),
            "DataArray must contain value {}; got:\n{}",
            expected,
            s
        );
    }
}

#[test]
fn test_write_vti_multicomponent() {
    let mut grid = VtkImageData {
        whole_extent: [0, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    grid.point_data.insert(
        "velocity".to_string(),
        AttributeArray::Vectors {
            values: vec![[1.0f32, 2.0, 3.0]] },
    );
    let s = write_vti_str(&grid);
    assert!(
        s.contains("NumberOfComponents=\"3\""),
        "vectors must emit NumberOfComponents=3; got:\n{}",
        s
    );
    assert!(
        s.contains("Name=\"velocity\""),
        "must emit velocity DataArray"
    );
    assert!(s.contains("1.000000"), "must contain x component");
    assert!(s.contains("3.000000"), "must contain z component");
}

#[test]
fn test_write_vti_cell_data() {
    let mut grid = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        ..Default::default()
    };
    grid.cell_data.insert(
        "pressure".to_string(),
        AttributeArray::Scalars {
            values: vec![99.0f32],
            num_components: 1 },
    );
    let s = write_vti_str(&grid);
    assert!(s.contains("<CellData>"), "must emit CellData section");
    assert!(
        s.contains("Name=\"pressure\""),
        "must emit pressure DataArray"
    );
    assert!(s.contains("99.000000"), "must contain cell value 99.0");
}

#[test]
fn test_write_vti_empty_grid() {
    let s = write_vti_str(&VtkImageData::default());
    assert!(
        s.contains("WholeExtent=\"0 0 0 0 0 0\""),
        "empty grid must have zero extent; got:\n{}",
        s
    );
    assert!(
        !s.contains("<PointData>"),
        "empty grid must not emit PointData"
    );
    assert!(
        !s.contains("<CellData>"),
        "empty grid must not emit CellData"
    );
    assert!(s.contains("<VTKFile"), "must still emit VTKFile");
    assert!(s.contains("<Piece"), "must still emit Piece");
}

#[test]
fn test_write_vti_file_roundtrip_via_string() {
    use crate::io::image_xml::reader::parse_vti;

    let grid = grid_2x2x2();
    let xml = write_vti_str(&grid);
    let parsed = parse_vti(&xml).expect("parse_vti must succeed on writer output");

    assert_eq!(
        parsed.whole_extent, grid.whole_extent,
        "extent must round-trip exactly"
    );
    for i in 0..3 {
        assert!(
            (parsed.origin[i] - grid.origin[i]).abs() < 1e-5,
            "origin[{}] mismatch: {} vs {}",
            i,
            parsed.origin[i],
            grid.origin[i]
        );
        assert!(
            (parsed.spacing[i] - grid.spacing[i]).abs() < 1e-5,
            "spacing[{}] mismatch: {} vs {}",
            i,
            parsed.spacing[i],
            grid.spacing[i]
        );
    }
    let orig_vals = match grid.point_data.get("density").unwrap() {
        AttributeArray::Scalars { values, .. } => values.clone(),
        _ => panic!("expected Scalars") };
    let parsed_vals = match parsed.point_data.get("density").unwrap() {
        AttributeArray::Scalars { values, .. } => values.clone(),
        _ => panic!("expected Scalars after parse") };
    assert_eq!(orig_vals.len(), parsed_vals.len(), "value count must match");
    for (i, (o, p)) in orig_vals.iter().zip(parsed_vals.iter()).enumerate() {
        assert!(
            (o - p).abs() < 1e-5,
            "density[{}]: wrote {:.6}, parsed {:.6}",
            i,
            o,
            p
        );
    }
}

#[test]
fn test_write_vti_rejects_invalid_grid() {
    let mut grid = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        ..Default::default()
    };
    grid.point_data.insert(
        "bad".to_string(),
        AttributeArray::Scalars {
            values: vec![0.0f32; 5],
            num_components: 1 },
    );
    let tmp = NamedTempFile::new().expect("temp file");
    let result = write_vti_image_data(tmp.path(), &grid);
    assert!(
        result.is_err(),
        "invalid grid (scalar length mismatch) must return Err"
    );
}

#[test]
fn test_write_vti_to_file_succeeds() {
    let tmp = NamedTempFile::new().expect("temp file");
    write_vti_image_data(tmp.path(), &grid_2x2x2()).expect("write must succeed");
    let bytes = std::fs::read(tmp.path()).expect("must read back file");
    assert!(!bytes.is_empty(), "written file must be non-empty");
    let content = String::from_utf8(bytes).expect("must be valid UTF-8");
    assert!(
        content.contains("<VTKFile"),
        "file content must contain VTKFile"
    );
    assert!(
        content.contains("ImageData"),
        "file content must reference ImageData"
    );
}
