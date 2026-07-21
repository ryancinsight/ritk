use super::*;
use crate::domain::vtk_data_object::{VtkDataObject, VtkPolyData};

/// Triangle in the XY plane: [0,0,0], [1,0,0], [0,1,0].
fn xy_triangle() -> VtkPolyData {
    VtkPolyData {
        points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        polygons: vec![vec![0, 1, 2]],
        ..Default::default()
    }
}

#[test]
fn flat_xy_triangle_normals_are_positive_z() {
    let f = ComputeNormalsFilter;
    let out = f.execute(VtkDataObject::PolyData(xy_triangle())).expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!("expected PolyData")
    };
    let AttributeArray::Normals { values } = p.point_data.get("Normals").expect("valid index") else {
        panic!("expected Normals attribute")
    };
    for n in values {
        assert!((n[0]).abs() < 1e-5, "nx must be ~0: got {}", n[0]);
        assert!((n[1]).abs() < 1e-5, "ny must be ~0: got {}", n[1]);
        assert!((n[2] - 1.0).abs() < 1e-5, "nz must be ~1: got {}", n[2]);
    }
}

#[test]
fn computed_normals_are_unit_length() {
    let f = ComputeNormalsFilter;
    let out = f.execute(VtkDataObject::PolyData(xy_triangle())).expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!("expected PolyData")
    };
    let AttributeArray::Normals { values } = p.point_data.get("Normals").expect("valid index") else {
        panic!("expected Normals")
    };
    for n in values {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 1e-5,
            "normal must be unit length: got len={}",
            len
        );
    }
}

#[test]
fn normals_stored_in_point_data_with_correct_count() {
    let f = ComputeNormalsFilter;
    let poly = xy_triangle();
    let n_points = poly.points.len();
    let out = f.execute(VtkDataObject::PolyData(poly)).expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!()
    };
    let AttributeArray::Normals { values } = p.point_data.get("Normals").expect("valid index") else {
        panic!()
    };
    assert_eq!(values.len(), n_points);
}

#[test]
fn xz_triangle_normals_are_negative_y() {
    // Triangle in XZ plane: normal should be [0, -1, 0] or [0, 1, 0]
    // points [0,0,0], [1,0,0], [0,0,1] → e1=[1,0,0], e2=[0,0,1]
    // cross = [0*1-0*0, 0*0-1*1, 1*0-0*0] = [0,-1,0]
    let poly = VtkPolyData {
        points: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        polygons: vec![vec![0, 1, 2]],
        ..Default::default()
    };
    let f = ComputeNormalsFilter;
    let out = f.execute(VtkDataObject::PolyData(poly)).expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!()
    };
    let AttributeArray::Normals { values } = p.point_data.get("Normals").expect("valid index") else {
        panic!()
    };
    for n in values {
        assert!(n[1].abs() > 0.99, "Y component must dominate: {:?}", n);
    }
}

#[test]
fn wrong_input_type_returns_err() {
    use crate::domain::vtk_data_object::VtkImageData;
    let f = ComputeNormalsFilter;
    let result = f.execute(VtkDataObject::ImageData(VtkImageData::default()));
    assert!(result.is_err(), "non-PolyData input must return Err");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("ImageData"), "error must name the actual type");
}

#[test]
fn degenerate_polygon_skipped_without_panic() {
    // polygon with < 3 vertices — must not panic
    let poly = VtkPolyData {
        points: vec![[0.0; 3], [1.0; 3]],
        polygons: vec![vec![0, 1]], // only 2 points — degenerate
        ..Default::default()
    };
    let f = ComputeNormalsFilter;
    let out = f.execute(VtkDataObject::PolyData(poly)).expect("infallible: validated precondition");
    let VtkDataObject::PolyData(p) = out else {
        panic!()
    };
    // accumulator remains [0,0,0] → fallback [0,0,1]
    let AttributeArray::Normals { values } = p.point_data.get("Normals").expect("valid index") else {
        panic!()
    };
    for n in values {
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!((len - 1.0).abs() < 1e-5, "fallback must be unit: {:?}", n);
    }
}
