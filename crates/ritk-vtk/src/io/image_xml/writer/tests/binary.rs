use crate::domain::vtk_data_object::{AttributeArray, VtkImageData};
use crate::io::image_xml::reader::read_vti_binary_appended_bytes;
use crate::io::image_xml::writer::{
    write_vti_binary_appended_bytes, write_vti_binary_appended_to_file,
};

/// Build a 2Ã—2Ã—2-point grid (extent [0,1,0,1,0,1], 8 points, 1 cell)
/// with a scalar point-data field named "density".
fn grid_2x2x2_binary() -> VtkImageData {
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
                    num_components: 1,
                },
            );
            m
        },
        cell_data: std::collections::HashMap::new(),
    }
}

/// Invariant: binary-appended output contains `format="appended"` in the
/// XML header and `AppendedData encoding="raw"` in the header, and the
/// total byte count exceeds the XML-only length (binary data is present).
#[test]
fn test_write_vti_binary_appended_header_contains_appended_format() {
    let grid = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 0],
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0, 1.0, 1.0],
        point_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "values".to_string(),
                AttributeArray::Scalars {
                    values: vec![1.0f32, 2.0, 3.0, 4.0],
                    num_components: 1,
                },
            );
            m
        },
        cell_data: std::collections::HashMap::new(),
    };

    let bytes = write_vti_binary_appended_bytes(&grid)
        .expect("write_vti_binary_appended_bytes must succeed on valid grid");

    let ad_start = bytes
        .windows(b"<AppendedData".len())
        .position(|w| w == b"<AppendedData")
        .expect("<AppendedData tag must be present");
    let gt_rel = bytes[ad_start..]
        .iter()
        .position(|&b| b == b'>')
        .expect("<AppendedData tag must have closing >");
    let us_rel = bytes[ad_start + gt_rel + 1..]
        .iter()
        .position(|&b| b == b'_')
        .expect("'_' marker must be present after AppendedData tag");
    let underscore_abs = ad_start + gt_rel + 1 + us_rel;

    let header = std::str::from_utf8(&bytes[..underscore_abs]).expect("header must be valid UTF-8");

    assert!(
        header.contains("format=\"appended\""),
        "header must contain format=\"appended\"; header:\n{}",
        header
    );
    assert!(
        header.contains("AppendedData encoding=\"raw\""),
        "header must contain AppendedData encoding=\"raw\"; header:\n{}",
        header
    );
    assert!(
        bytes.len() > underscore_abs + 20,
        "total bytes must exceed header_end+20 (binary data must be present); \
         total={}, header_end={}",
        bytes.len(),
        underscore_abs
    );
}

/// Invariant: binary-appended round-trip preserves whole_extent, origin,
/// spacing, and all scalar values within f32 representation error (< 1e-6).
#[test]
fn test_write_vti_binary_appended_roundtrip() {
    let expected_values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let grid = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        origin: [1.0, 2.0, 3.0],
        spacing: [0.5, 0.5, 0.5],
        point_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "scalars".to_string(),
                AttributeArray::Scalars {
                    values: expected_values.clone(),
                    num_components: 1,
                },
            );
            m
        },
        cell_data: std::collections::HashMap::new(),
    };

    let bytes = write_vti_binary_appended_bytes(&grid)
        .expect("write_vti_binary_appended_bytes must succeed");
    let parsed = read_vti_binary_appended_bytes(&bytes)
        .expect("read_vti_binary_appended_bytes must parse writer output");

    assert_eq!(
        parsed.whole_extent, grid.whole_extent,
        "whole_extent must round-trip exactly"
    );
    for i in 0..3 {
        assert!(
            (parsed.origin[i] - grid.origin[i]).abs() < 1e-5,
            "origin[{}]: expected {}, got {}",
            i,
            grid.origin[i],
            parsed.origin[i]
        );
        assert!(
            (parsed.spacing[i] - grid.spacing[i]).abs() < 1e-5,
            "spacing[{}]: expected {}, got {}",
            i,
            grid.spacing[i],
            parsed.spacing[i]
        );
    }
    let parsed_vals = match parsed.point_data.get("scalars") {
        Some(AttributeArray::Scalars { values, .. }) => values.clone(),
        other => panic!(
            "expected Scalars variant for 'scalars' key, got {:?}",
            other
        ),
    };
    assert_eq!(
        parsed_vals.len(),
        expected_values.len(),
        "scalar value count must match after round-trip"
    );
    for (i, (e, g)) in expected_values.iter().zip(parsed_vals.iter()).enumerate() {
        assert!(
            (e - g).abs() < 1e-6,
            "scalars[{}]: expected {}, got {} (diff {})",
            i,
            e,
            g,
            (e - g).abs()
        );
    }
}

/// Invariant: offset[0] = 0; offset[1] = 4 + n_values[0] * 4.
#[test]
fn test_write_vti_binary_appended_offset_correctness() {
    let mut grid = VtkImageData {
        whole_extent: [0, 1, 0, 0, 0, 0],
        ..Default::default()
    };
    grid.point_data.insert(
        "A".to_string(),
        AttributeArray::Scalars {
            values: vec![1.0f32, 2.0],
            num_components: 1,
        },
    );
    grid.point_data.insert(
        "B".to_string(),
        AttributeArray::Scalars {
            values: vec![3.0f32, 4.0, 5.0, 6.0],
            num_components: 2,
        },
    );

    let bytes = write_vti_binary_appended_bytes(&grid).expect("write must succeed on valid grid");

    let ad_start = bytes
        .windows(b"<AppendedData".len())
        .position(|w| w == b"<AppendedData")
        .expect("<AppendedData tag must be present");
    let gt_rel = bytes[ad_start..]
        .iter()
        .position(|&b| b == b'>')
        .expect("<AppendedData tag must close");
    let us_rel = bytes[ad_start + gt_rel + 1..]
        .iter()
        .position(|&b| b == b'_')
        .expect("'_' marker must be present");
    let underscore_abs = ad_start + gt_rel + 1 + us_rel;

    let header = std::str::from_utf8(&bytes[..underscore_abs]).expect("header must be valid UTF-8");

    assert!(
        header.contains("offset=\"0\""),
        "array 'A' must have offset=0; header:\n{}",
        header
    );
    assert!(
        header.contains("offset=\"12\""),
        "array 'B' must have offset=12 (=4+2*4); header:\n{}",
        header
    );
}

/// Invariant: a CellData-only binary-appended grid round-trips to an
/// identical grid with no PointData and exactly the original CellData values.
#[test]
fn test_write_vti_binary_appended_cell_data_only_roundtrip() {
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
        .expect("write_vti_binary_appended_bytes must succeed on valid cell-data-only grid");
    let parsed = read_vti_binary_appended_bytes(&bytes)
        .expect("read_vti_binary_appended_bytes must succeed on cell-data-only bytes");

    assert_eq!(
        parsed.whole_extent, grid.whole_extent,
        "whole_extent must match exactly"
    );
    assert!(
        parsed.point_data.is_empty(),
        "cell-only grid must have no PointData after parse"
    );
    assert!(
        parsed.cell_data.contains_key("pressure"),
        "must have 'pressure' CellData key"
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

/// Invariant: both PointData and CellData survive binary-appended round-trip
/// simultaneously with all values preserved within f32 representation error.
#[test]
fn test_write_vti_binary_appended_mixed_point_and_cell_data() {
    let grid = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 1],
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0, 1.0, 1.0],
        point_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "density".to_string(),
                AttributeArray::Scalars {
                    values: vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                    num_components: 1,
                },
            );
            m
        },
        cell_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "material".to_string(),
                AttributeArray::Scalars {
                    values: vec![7.0f32],
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

    assert_eq!(
        parsed.whole_extent, grid.whole_extent,
        "whole_extent must match exactly"
    );
    assert!(
        parsed.point_data.contains_key("density"),
        "parsed must contain 'density' PointData key"
    );
    let pd_vals = match parsed.point_data.get("density").unwrap() {
        AttributeArray::Scalars { values, .. } => values.clone(),
        other => panic!("expected Scalars for 'density', got {:?}", other),
    };
    assert_eq!(pd_vals.len(), 8, "density must have 8 values");
    let expected_density = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    for (i, (&got, &exp)) in pd_vals.iter().zip(expected_density.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-6,
            "density[{}]: expected {}, got {} (diff {})",
            i,
            exp,
            got,
            (got - exp).abs()
        );
    }
    assert!(
        parsed.cell_data.contains_key("material"),
        "parsed must contain 'material' CellData key"
    );
    let cd_vals = match parsed.cell_data.get("material").unwrap() {
        AttributeArray::Scalars { values, .. } => values.clone(),
        other => panic!("expected Scalars for 'material', got {:?}", other),
    };
    assert_eq!(cd_vals.len(), 1, "material must have 1 value");
    assert!(
        (cd_vals[0] - 7.0f32).abs() < 1e-6,
        "material[0]: expected 7.0, got {} (diff {})",
        cd_vals[0],
        (cd_vals[0] - 7.0f32).abs()
    );
}

/// Invariant: non-scalar point data is emitted from component storage without
/// changing component order; vectors remain vectors and normal-named arrays
/// remain normals after a binary-appended round-trip.
#[test]
fn test_write_vti_binary_appended_vector_and_normal_roundtrip() {
    let grid = VtkImageData {
        whole_extent: [0, 1, 0, 0, 0, 0],
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0, 1.0, 1.0],
        point_data: {
            let mut m = std::collections::HashMap::new();
            m.insert(
                "Normals".to_string(),
                AttributeArray::Normals {
                    values: vec![[0.0f32, 1.0, 0.0], [0.0, 0.0, 1.0]],
                },
            );
            m.insert(
                "velocity".to_string(),
                AttributeArray::Vectors {
                    values: vec![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
                },
            );
            m
        },
        cell_data: std::collections::HashMap::new(),
    };

    let bytes =
        write_vti_binary_appended_bytes(&grid).expect("write must succeed for vector point data");
    let parsed = read_vti_binary_appended_bytes(&bytes).expect("read must parse vector point data");

    let velocity = match parsed.point_data.get("velocity") {
        Some(AttributeArray::Vectors { values }) => values,
        other => panic!("expected velocity vectors, got {:?}", other),
    };
    assert_eq!(
        velocity,
        &vec![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]],
        "velocity vectors must round-trip in component order"
    );

    let normals = match parsed.point_data.get("Normals") {
        Some(AttributeArray::Normals { values }) => values,
        other => panic!("expected normal vectors, got {:?}", other),
    };
    assert_eq!(
        normals,
        &vec![[0.0f32, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "normal vectors must round-trip in component order"
    );
}

/// Invariant: when both PointData and CellData are present, the CellData
/// array's offset equals the total byte size of all PointData blocks.
#[test]
fn test_write_vti_binary_appended_cell_data_offset_after_point_data() {
    let mut grid = VtkImageData {
        whole_extent: [0, 1, 0, 0, 0, 0],
        origin: [0.0, 0.0, 0.0],
        spacing: [1.0, 1.0, 1.0],
        ..Default::default()
    };
    grid.point_data.insert(
        "pd".to_string(),
        AttributeArray::Scalars {
            values: vec![1.0f32, 2.0],
            num_components: 1,
        },
    );
    grid.cell_data.insert(
        "cd".to_string(),
        AttributeArray::Scalars {
            values: vec![9.0f32],
            num_components: 1,
        },
    );

    let bytes = write_vti_binary_appended_bytes(&grid).expect("write must succeed on valid grid");

    let ad_start = bytes
        .windows(b"<AppendedData".len())
        .position(|w| w == b"<AppendedData")
        .expect("<AppendedData tag must be present");
    let gt_rel = bytes[ad_start..]
        .iter()
        .position(|&b| b == b'>')
        .expect("<AppendedData tag must close");
    let us_rel = bytes[ad_start + gt_rel + 1..]
        .iter()
        .position(|&b| b == b'_')
        .expect("'_' marker must be present");
    let underscore_abs = ad_start + gt_rel + 1 + us_rel;

    let header = std::str::from_utf8(&bytes[..underscore_abs]).expect("header must be valid UTF-8");

    assert!(
        header.contains("Name=\"pd\""),
        "header must contain Name=\"pd\"; header:\n{}",
        header
    );
    assert!(
        header.contains("Name=\"cd\""),
        "header must contain Name=\"cd\"; header:\n{}",
        header
    );
    assert!(
        header.contains("offset=\"0\""),
        "PointData array 'pd' must have offset=0; header:\n{}",
        header
    );
    assert!(
        header.contains("offset=\"12\""),
        "CellData array 'cd' must have offset=12 (=4+2*4); header:\n{}",
        header
    );
}

#[test]
fn test_write_vti_binary_appended_to_file_succeeds() {
    use tempfile::NamedTempFile;
    let grid = grid_2x2x2_binary();
    let tmp = NamedTempFile::new().expect("temp file");
    write_vti_binary_appended_to_file(tmp.path(), &grid).expect("write must succeed");
    let bytes = std::fs::read(tmp.path()).expect("must read back file");
    assert!(!bytes.is_empty(), "written file must be non-empty");
}
