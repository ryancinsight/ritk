use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::transform::{Transform, RigidTransform};
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
