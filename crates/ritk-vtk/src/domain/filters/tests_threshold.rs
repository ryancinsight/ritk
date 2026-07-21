use super::*;
use crate::domain::vtk_data_object::{AttributeArray, VtkDataObject, VtkImageData};

fn image_2x2x1(values: [f32; 4]) -> VtkImageData {
    let mut img = VtkImageData {
        whole_extent: [0, 1, 0, 1, 0, 0],
        spacing: [1.0, 1.0, 1.0],
        ..Default::default()
    };
    img.point_data.insert(
        "scalars".to_string(),
        AttributeArray::Scalars {
            values: values.to_vec(),
            num_components: 1,
        },
    );
    img
}

#[test]
fn all_below_lower_bound_gives_empty_output() {
    let f = ThresholdFilter::new("scalars", 10.0, 20.0);
    let img = image_2x2x1([0.1, 0.2, 0.3, 0.4]);
    let out = f
        .execute(VtkDataObject::ImageData(img))
        .expect("infallible: validated precondition");
    let VtkDataObject::UnstructuredGrid(ug) = out else {
        panic!()
    };
    assert_eq!(ug.points.len(), 0, "all values below lower → empty output");
    assert_eq!(ug.n_cells(), 0);
}

#[test]
fn all_above_upper_bound_gives_empty_output() {
    let f = ThresholdFilter::new("scalars", -20.0, -10.0);
    let img = image_2x2x1([0.1, 0.2, 0.3, 0.4]);
    let out = f
        .execute(VtkDataObject::ImageData(img))
        .expect("infallible: validated precondition");
    let VtkDataObject::UnstructuredGrid(ug) = out else {
        panic!()
    };
    assert_eq!(ug.points.len(), 0, "all values above upper → empty output");
}

#[test]
fn all_in_range_passes_all_points() {
    let f = ThresholdFilter::new("scalars", 0.0, 1.0);
    let img = image_2x2x1([0.1, 0.2, 0.3, 0.4]);
    let out = f
        .execute(VtkDataObject::ImageData(img))
        .expect("infallible: validated precondition");
    let VtkDataObject::UnstructuredGrid(ug) = out else {
        panic!()
    };
    assert_eq!(ug.points.len(), 4, "all values in range → 4 points");
    assert_eq!(ug.n_cells(), 4);
}

#[test]
fn boundary_values_are_inclusive() {
    // values = [0.1, 0.5, 0.8, 1.2]; threshold [0.5, 0.8] → passes indices 1 and 2
    let f = ThresholdFilter::new("scalars", 0.5, 0.8);
    let img = image_2x2x1([0.1, 0.5, 0.8, 1.2]);
    let out = f
        .execute(VtkDataObject::ImageData(img))
        .expect("infallible: validated precondition");
    let VtkDataObject::UnstructuredGrid(ug) = out else {
        panic!()
    };
    assert_eq!(
        ug.points.len(),
        2,
        "exactly lower and upper boundary values pass: got {} points",
        ug.points.len()
    );
    let AttributeArray::Scalars { values, .. } = ug.cell_data.get("scalars").expect("valid index")
    else {
        panic!()
    };
    // Both passing scalars must be within [0.5, 0.8]
    for &v in values {
        assert!(
            (0.5 - 1e-5..=0.8 + 1e-5).contains(&v),
            "output scalar {} must be in [0.5, 0.8]",
            v
        );
    }
}

#[test]
fn threshold_on_unstructured_grid_filters_cells() {
    // Build a UG with 3 cells having scalars [1.0, 5.0, 9.0]
    let mut ug = VtkUnstructuredGrid::new();
    ug.points = vec![[0.0; 3]; 3];
    ug.cells = vec![vec![0], vec![1], vec![2]];
    ug.cell_types = vec![VtkCellType::Vertex; 3];
    ug.cell_data.insert(
        "pressure".to_string(),
        AttributeArray::Scalars {
            values: vec![1.0, 5.0, 9.0],
            num_components: 1,
        },
    );
    let f = ThresholdFilter::new("pressure", 4.0, 6.0);
    let out = f
        .execute(VtkDataObject::UnstructuredGrid(ug))
        .expect("infallible: validated precondition");
    let VtkDataObject::UnstructuredGrid(result) = out else {
        panic!()
    };
    assert_eq!(result.n_cells(), 1, "only cell with scalar=5.0 must pass");
    let AttributeArray::Scalars { values, .. } =
        result.cell_data.get("pressure").expect("valid index")
    else {
        panic!()
    };
    assert_eq!(values.len(), 1);
    assert!((values[0] - 5.0).abs() < 1e-5, "passing scalar must be 5.0");
}

#[test]
fn missing_scalar_name_returns_err() {
    let f = ThresholdFilter::new("nonexistent_field", 0.0, 1.0);
    let img = image_2x2x1([0.1, 0.2, 0.3, 0.4]);
    let result = f.execute(VtkDataObject::ImageData(img));
    assert!(result.is_err(), "missing scalar field must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(
        msg.contains("nonexistent_field"),
        "error must name the missing field"
    );
}

#[test]
fn wrong_input_type_returns_err() {
    use crate::domain::vtk_data_object::VtkPolyData;
    let f = ThresholdFilter::new("s", 0.0, 1.0);
    let result = f.execute(VtkDataObject::PolyData(VtkPolyData::default()));
    assert!(result.is_err(), "PolyData input must return Err");
}

#[test]
fn test_threshold_filter_range_change_triggers_rerun() {
    let mut tf = ThresholdFilter::new("scalars", 0.0, 1.0);
    let mtime_before = tf.get_mtime();

    tf.set_range(0.5, 0.8);
    let mtime_after_range = tf.get_mtime();
    assert!(
        mtime_after_range > mtime_before,
        "set_range must bump mtime: before={}, after={}",
        mtime_before.value(),
        mtime_after_range.value()
    );

    tf.set_scalar_name("pressure");
    let mtime_after_name = tf.get_mtime();
    assert!(
        mtime_after_name > mtime_after_range,
        "set_scalar_name must bump mtime: before={}, after={}",
        mtime_after_range.value(),
        mtime_after_name.value()
    );
}
