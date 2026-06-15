use super::*;
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

/// Tolerance for near-zero assertions in identity and translation tests.
const NEAR_ZERO: f32 = 1e-6;

/// Tolerance for the rigid-seeded affine round-trip test.
/// Euler-angle composition accumulates ~5 ULP; 1e-5 is tight but robust.
const RIGID_AFFINE_CONSISTENCY_TOL: f32 = 1e-5;

#[test]
fn test_affine_transform_identity() {
    let device = Default::default();
    let transform = AffineTransform::<TestBackend, 3>::identity(None, &device);

    let points = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);

    let transformed = transform.transform_points(points);
    let data = transformed.to_data();

    // Identity transform should not change points
    let slice = data.as_slice::<f32>().unwrap();
    assert_eq!(slice[0], 1.0);
    assert_eq!(slice[1], 2.0);
    assert_eq!(slice[2], 3.0);
    assert_eq!(slice[3], 4.0);
    assert_eq!(slice[4], 5.0);
    assert_eq!(slice[5], 6.0);
}

#[test]
fn test_affine_transform_translation_with_center() {
    let device = Default::default();

    // Matrix: Identity
    let matrix = Tensor::<TestBackend, 2>::eye(2, &device);

    // Translation: [1.0, 1.0]
    let translation = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);

    // Center: [10.0, 10.0]
    let center = Tensor::<TestBackend, 1>::from_floats([10.0, 10.0], &device);

    let transform = AffineTransform::<TestBackend, 2>::new(matrix, translation, center);

    // Point at center: [10, 10]
    // T(c) = A(c-c) + c + t = 0 + c + t = c + t
    // Expected: [11, 11]
    let points = Tensor::<TestBackend, 2>::from_floats([[10.0, 10.0]], &device);

    let transformed = transform.transform_points(points);
    let data = transformed.to_data();
    let slice = data.as_slice::<f32>().unwrap();

    assert_eq!(slice[0], 11.0);
    assert_eq!(slice[1], 11.0);
}

#[test]
fn test_affine_transform_scale_with_center() {
    let device = Default::default();

    // Matrix: Scale by 2.0
    // [2, 0]
    // [0, 2]
    let matrix = Tensor::<TestBackend, 2>::eye(2, &device) * 2.0;

    // Translation: 0
    let translation = Tensor::<TestBackend, 1>::zeros([2], &device);

    // Center: [1.0, 1.0]
    let center = Tensor::<TestBackend, 1>::from_floats([1.0, 1.0], &device);

    let transform = AffineTransform::<TestBackend, 2>::new(matrix, translation, center);

    // Point: [2.0, 1.0] (1 unit right of center)
    // T(x) = A(x - c) + c
    // x - c = [1, 0]
    // A(x-c) = [2, 0]
    // + c = [3, 1]
    let points = Tensor::<TestBackend, 2>::from_floats([[2.0, 1.0]], &device);

    let transformed = transform.transform_points(points);
    let data = transformed.to_data();
    let slice = data.as_slice::<f32>().unwrap();

    assert!((slice[0] - 3.0).abs() < NEAR_ZERO);
    assert!((slice[1] - 1.0).abs() < NEAR_ZERO);
}

#[test]
#[should_panic(expected = "expects a [D, D] linear matrix")]
fn new_rejects_homogeneous_matrix() {
    // Seeding a 3-D affine with a [D+1, D+1] = [4, 4] homogeneous matrix
    // (the shape `RigidTransform::matrix()` returns) must fail loudly at
    // construction, not with a cryptic backend matmul panic later.
    let device = Default::default();
    let matrix = Tensor::<TestBackend, 2>::eye(4, &device);
    let translation = Tensor::<TestBackend, 1>::zeros([3], &device);
    let center = Tensor::<TestBackend, 1>::zeros([3], &device);
    let _ = AffineTransform::<TestBackend, 3>::new(matrix, translation, center);
}

#[test]
fn affine_seeded_from_rigid_rotation_reproduces_rigid() {
    use crate::transform::affine::rigid::RigidTransform;
    let device = Default::default();
    // A rigid transform with a non-trivial rotation/translation/center.
    let rigid = RigidTransform::<TestBackend, 3>::new(
        Tensor::from_floats([2.0, -1.0, 3.0], &device), // translation
        Tensor::from_floats([0.3, -0.2, 0.1], &device), // Euler angles
        Tensor::from_floats([5.0, 6.0, 7.0], &device),  // center
    );
    // Correct seeding: A = R (the [D, D] rotation), not the homogeneous matrix.
    let affine = AffineTransform::<TestBackend, 3>::new(
        rigid.build_rotation_matrix(),
        rigid.translation(),
        rigid.center(),
    );
    let pts = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [10.0, -4.0, 0.5]], &device);
    let r = rigid.transform_points(pts.clone()).to_data();
    let a = affine.transform_points(pts).to_data();
    let r = r.as_slice::<f32>().unwrap();
    let a = a.as_slice::<f32>().unwrap();
    for (ri, ai) in r.iter().zip(a.iter()) {
        assert!(
            (ri - ai).abs() < RIGID_AFFINE_CONSISTENCY_TOL,
            "affine {ai} != rigid {ri}"
        );
    }
}
