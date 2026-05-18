use super::transform::transform_point;
use super::*;
use ndarray::Array2;

#[test]
fn test_kabsch_identity() {
    let fixed = Array2::from_shape_vec((3, 3), vec![0., 0., 0., 1., 0., 0., 0., 1., 0.]).unwrap();
    let rotation = kabsch_algorithm(&fixed, &fixed).unwrap();

    let expected = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    for (r, e) in rotation.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-10, "rotation[i]={r} expected {e}");
    }
}

#[test]
fn test_build_homogeneous_matrix() {
    let rotation = [1., 0., 0., 0., 1., 0., 0., 0., 1.];
    let translation = [1., 2., 3.];
    let matrix = build_homogeneous_matrix(&rotation, &translation);

    assert_eq!(matrix[3], 1.);
    assert_eq!(matrix[7], 2.);
    assert_eq!(matrix[11], 3.);
    assert_eq!(matrix[15], 1.);
}

#[test]
fn test_transform_point() {
    let point = [1., 0., 0.];
    let transform = [
        1., 0., 0., 10., 0., 1., 0., 20., 0., 0., 1., 30., 0., 0., 0., 1.,
    ];
    let result = transform_point(&point, &transform);

    assert!((result[0] - 11.).abs() < 1e-10);
    assert!((result[1] - 20.).abs() < 1e-10);
    assert!((result[2] - 30.).abs() < 1e-10);
}
