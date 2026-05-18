use super::*;
use crate::domain::vtk_data_object::{AttributeArray, VtkStructuredGrid};
use tempfile::NamedTempFile;

#[test]
fn test_structured_grid_roundtrip_identity() {
    let dims = [2usize, 3, 2];
    let mut points = Vec::with_capacity(12);
    for iz in 0..2usize {
        for iy in 0..3usize {
            for ix in 0..2usize {
                points.push([ix as f32, iy as f32, iz as f32]);
            }
        }
    }
    let scalars = AttributeArray::Scalars {
        values: (0..12).map(|i| i as f32).collect(),
        num_components: 1,
    };
    let mut grid = VtkStructuredGrid::new(dims);
    grid.points = points.clone();
    grid.point_data.insert("intensity".to_string(), scalars);

    let tmp = NamedTempFile::new().expect("temp file");
    write_vtk_structured_grid(tmp.path(), &grid).expect("write");
    let result = read_vtk_structured_grid(tmp.path()).expect("read");

    assert_eq!(result.dimensions, [2, 3, 2]);
    assert_eq!(result.n_points(), 12);
    for (i, (expected, got)) in points.iter().zip(result.points.iter()).enumerate() {
        for c in 0..3 {
            assert!(
                (expected[c] - got[c]).abs() < 1e-5,
                "point[{}][{}]: expected {} got {}",
                i,
                c,
                expected[c],
                got[c]
            );
        }
    }
    match result
        .point_data
        .get("intensity")
        .expect("intensity attribute")
    {
        AttributeArray::Scalars {
            values,
            num_components,
        } => {
            assert_eq!(*num_components, 1);
            assert_eq!(values.len(), 12);
            for i in 0..12 {
                assert!(
                    (values[i] - i as f32).abs() < 1e-5,
                    "scalar[{}]: expected {} got {}",
                    i,
                    i as f32,
                    values[i]
                );
            }
        }
        other => panic!("expected Scalars, got {:?}", other),
    }
}

#[test]
fn test_structured_grid_validate_rejects_wrong_point_count() {
    let mut grid = VtkStructuredGrid::new([2, 2, 2]);
    grid.points = vec![[0.0f32; 3]; 5];
    let tmp = NamedTempFile::new().expect("temp file");
    let result = write_vtk_structured_grid(tmp.path(), &grid);
    assert!(
        result.is_err(),
        "write must fail when point count mismatches dimensions"
    );
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("n_points") || msg.contains("points"),
        "error message must reference point count: got {}",
        msg
    );
}

#[test]
fn test_structured_grid_roundtrip_vectors() {
    let dims = [2usize, 2, 1];
    let mut points = Vec::with_capacity(4);
    for iz in 0..1usize {
        for iy in 0..2usize {
            for ix in 0..2usize {
                points.push([ix as f32, iy as f32, iz as f32]);
            }
        }
    }
    let expected_vecs: Vec<[f32; 3]> = vec![
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ];
    let mut grid = VtkStructuredGrid::new(dims);
    grid.points = points;
    grid.point_data.insert(
        "velocity".to_string(),
        AttributeArray::Vectors {
            values: expected_vecs.clone(),
        },
    );

    let tmp = NamedTempFile::new().expect("temp file");
    write_vtk_structured_grid(tmp.path(), &grid).expect("write");
    let result = read_vtk_structured_grid(tmp.path()).expect("read");

    assert_eq!(result.n_points(), 4);
    match result
        .point_data
        .get("velocity")
        .expect("velocity attribute")
    {
        AttributeArray::Vectors { values } => {
            assert_eq!(values.len(), 4);
            for (i, (exp, got)) in expected_vecs.iter().zip(values.iter()).enumerate() {
                for c in 0..3 {
                    assert!(
                        (exp[c] - got[c]).abs() < 1e-5,
                        "velocity[{}][{}]: expected {} got {}",
                        i,
                        c,
                        exp[c],
                        got[c]
                    );
                }
            }
        }
        other => panic!("expected Vectors, got {:?}", other),
    }
}
