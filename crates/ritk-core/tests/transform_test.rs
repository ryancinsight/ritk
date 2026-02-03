use burn::tensor::{Tensor, TensorData, Shape};
use burn_ndarray::NdArray;
use ritk_core::transform::{Transform, RigidTransform, DisplacementField, DisplacementFieldTransform, DisplacementFieldTransform2D, DisplacementFieldTransform3D};
use ritk_core::spatial::{Point, Spacing, Direction};
use ritk_core::interpolation::LinearInterpolator;
use std::f32::consts::PI;

type B = NdArray<f32>;

#[test]
fn test_rigid_transform_2d() {
    let device = Default::default();

    // Rotate 90 degrees (PI/2) and translate by (1, 1)
    // Point (1, 0) -> Rotation(90) -> (0, 1) -> Translation(1, 1) -> (1, 2)

    let points_data = TensorData::from([[1.0, 0.0]]);
    let points = Tensor::<B, 2>::from_data(points_data, &device);

    let translation_data = TensorData::from([1.0, 1.0]);
    let translation = Tensor::<B, 1>::from_data(translation_data, &device);

    let rotation_data = TensorData::from([PI / 2.0]);
    let rotation = Tensor::<B, 1>::from_data(rotation_data, &device);

    let center = Tensor::<B, 1>::zeros([2], &device);
    let transform = RigidTransform::<B, 2>::new(translation, rotation, center);

    let transformed = transform.transform_points(points);
    let result = transformed.into_data();

    let expected = [1.0, 2.0]; // Approximately
    let actual = result.as_slice::<f32>().unwrap();

    assert!((actual[0] - expected[0]).abs() < 1e-5, "X mismatch: got {}, expected {}", actual[0], expected[0]);
    assert!((actual[1] - expected[1]).abs() < 1e-5, "Y mismatch: got {}, expected {}", actual[1], expected[1]);
}

#[test]
fn test_rigid_transform_3d() {
    let device = Default::default();

    // Rotate 90 degrees around Z axis (Gamma = PI/2)
    // Point (1, 0, 0) -> (0, 1, 0)
    // Translate by (1, 2, 3) -> (1, 3, 3)

    let points_data = TensorData::from([[1.0, 0.0, 0.0]]);
    let points = Tensor::<B, 2>::from_data(points_data, &device);

    let translation_data = TensorData::from([1.0, 2.0, 3.0]);
    let translation = Tensor::<B, 1>::from_data(translation_data, &device);

    // Euler angles: x, y, z
    let rotation_data = TensorData::from([0.0, 0.0, PI / 2.0]);
    let rotation = Tensor::<B, 1>::from_data(rotation_data, &device);

    let center = Tensor::<B, 1>::zeros([3], &device);
    let transform = RigidTransform::<B, 3>::new(translation, rotation, center);

    let transformed = transform.transform_points(points);
    let result = transformed.into_data();

    let expected = [1.0, 3.0, 3.0];
    let actual = result.as_slice::<f32>().unwrap();

    assert!((actual[0] - expected[0]).abs() < 1e-5, "X mismatch: got {}, expected {}", actual[0], expected[0]);
    assert!((actual[1] - expected[1]).abs() < 1e-5, "Y mismatch: got {}, expected {}", actual[1], expected[1]);
    assert!((actual[2] - expected[2]).abs() < 1e-5, "Z mismatch: got {}, expected {}", actual[2], expected[2]);
}

#[test]
fn test_displacement_field_transform_2d() {
    let device = Default::default();

    // Create a 2x2 displacement field
    // X component: [[1.0, 1.0], [1.0, 1.0]] (Constant shift +1 in X)
    // Y component: [[0.5, 0.5], [0.5, 0.5]] (Constant shift +0.5 in Y)
    
    // Create a 2D displacement field with shape [2, 2] (spatial)
    let x_data = TensorData::from([[1.0, 1.0], [1.0, 1.0]]); // [2, 2]
    let x_tensor = Tensor::<B, 2>::from_data(x_data, &device);
    
    let y_data = TensorData::from([[0.5, 0.5], [0.5, 0.5]]); // [2, 2]
    let y_tensor = Tensor::<B, 2>::from_data(y_data, &device);
    
    let origin = Point::new([0.0, 0.0]);
    let spacing = Spacing::new([1.0, 1.0]);
    let direction = Direction::identity();

    let field = DisplacementField::new(
        vec![x_tensor, y_tensor],
        origin,
        spacing,
        direction,
    );

    let transform = DisplacementFieldTransform2D::new(field, LinearInterpolator::new());
    
    // Test point (0.5, 0.5)
    // Physical (0.5, 0.5) -> Index (0.5, 0.5)
    // Interpolation should be perfect since values are constant
    let points_data = TensorData::from([[0.5, 0.5]]);
    let points = Tensor::<B, 2>::from_data(points_data, &device);
    
    let transformed = transform.transform_points(points);
    let result = transformed.into_data();
    let actual = result.as_slice::<f32>().unwrap();
    
    // Expected: (0.5 + 1.0, 0.5 + 0.5) = (1.5, 1.0)
    // But wait, the interpolation adds displacement to the point.
    // D(x) = (1.0, 0.5)
    // T(x) = x + D(x) = (0.5, 0.5) + (1.0, 0.5) = (1.5, 1.0)
    
    assert!((actual[0] - 1.5).abs() < 1e-5, "X mismatch: got {}, expected 1.5", actual[0]);
    assert!((actual[1] - 1.0).abs() < 1e-5, "Y mismatch: got {}, expected 1.0", actual[1]);
}
