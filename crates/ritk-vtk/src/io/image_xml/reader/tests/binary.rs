use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use crate::io::image_xml::reader::read_vti_binary_appended_bytes;
use crate::io::image_xml::writer::write_vti_binary_appended_bytes;

/// Invariant: the reader correctly parses a CellData array from a
/// binary-appended VTI byte buffer and returns it with exact scalar values.

#[test]
#[allow(clippy::approx_constant)]
fn test_read_vti_binary_appended_cell_data_roundtrip() {
    // extent [0,1,0,1,0,1] â†’ n_cells = 1Ã—1Ã—1 = 1; n_points = 2Ã—2Ã—2 = 8
    let grid = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0, 1.0, 1.0],
        point_data: std::collections::HashMap::new(),
        cell_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "pressure".to_string(),
                AttributeArray::Scalars {
                    values: vec![42.0f32],
                    num_components: 1,
                },
            );
            m
        },
    };

    let bytes = write_vti_binary_appended_bytes(&grid)
        .expect("write_vti_binary_appended_bytes must succeed on cell-data-only grid");
    let parsed = read_vti_binary_appended_bytes(&bytes)
        .expect("read_vti_binary_appended_bytes must succeed on cell-data-only bytes");

    assert!(
        parsed.cell_data.contains_key("pressure"),
        "parsed must contain 'pressure' CellData key"
    );
    let values = match parsed.cell_data.get("pressure").unwrap() {
        AttributeArray::Scalars { values, .. } => values.clone(),
        other => panic!("expected Scalars variant for 'pressure', got {:?}", other),
    };
    assert_eq!(values.len(), 1, "pressure CellData must have 1 value");
    assert!(
        (values[0] - 42.0f32).abs() < 1e-6,
        "pressure[0]: expected 42.0, got {} (diff {})",
        values[0],
        (values[0] - 42.0f32).abs()
    );
}

/// Invariant: the reader preserves both PointData and CellData sections
/// from a mixed binary-appended file with all values intact.
#[test]
#[allow(clippy::approx_constant)]
fn test_read_vti_binary_appended_preserves_both_sections() {
    // extent [0,1,0,1,0,0] â†’ n_points = 2Ã—2Ã—1 = 4; n_cells = 1Ã—1Ã—1 = 1
    let grid = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 0],
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0, 1.0, 1.0],
        point_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "temperature".to_string(),
                AttributeArray::Scalars {
                    values: vec![100.0f32, 200.0, 300.0, 400.0],
                    num_components: 1,
                },
            );
            m
        },
        cell_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "flux".to_string(),
                AttributeArray::Scalars {
                    values: vec![3.14f32],
                    num_components: 1,
                },
            );
            m
        },
    };

    let bytes = write_vti_binary_appended_bytes(&grid)
        .expect("write_vti_binary_appended_bytes must succeed on mixed grid");
    let parsed = read_vti_binary_appended_bytes(&bytes)
        .expect("read_vti_binary_appended_bytes must succeed on mixed bytes");

    assert!(
        parsed.point_data.contains_key("temperature"),
        "parsed must contain 'temperature' PointData key"
    );
    let temp_vals = match parsed.point_data.get("temperature").unwrap() {
        AttributeArray::Scalars { values, .. } => values.clone(),
        other => panic!("expected Scalars for 'temperature', got {:?}", other),
    };
    assert_eq!(temp_vals.len(), 4, "temperature must have 4 values");
    assert!(
        (temp_vals[3] - 400.0f32).abs() < 1e-6,
        "temperature[3]: expected 400.0, got {} (diff {})",
        temp_vals[3],
        (temp_vals[3] - 400.0f32).abs()
    );

    assert!(
        parsed.cell_data.contains_key("flux"),
        "parsed must contain 'flux' CellData key"
    );
    let flux_vals = match parsed.cell_data.get("flux").unwrap() {
        AttributeArray::Scalars { values, .. } => values.clone(),
        other => panic!("expected Scalars for 'flux', got {:?}", other),
    };
    assert_eq!(flux_vals.len(), 1, "flux must have 1 value");
    assert!(
        (flux_vals[0] - 3.14f32).abs() < 1e-5,
        "flux[0]: expected 3.14, got {} (diff {})",
        flux_vals[0],
        (flux_vals[0] - 3.14f32).abs()
    );
}
