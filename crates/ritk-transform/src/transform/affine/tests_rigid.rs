use super::*;
use coeus_core::SequentialBackend;

type TestBackend = SequentialBackend;

#[test]
fn translation_2d_shifts_point_correctly() {
    let device = Default::default();
    let translation = Tensor::<f32, TestBackend>::from_floats([1.0, 2.0], &device);
    let rotation = Tensor::<f32, TestBackend>::from_floats([0.0], &device); // No rotation
    let center = Tensor::<f32, TestBackend>::zeros([2], &device);
    let transform = RigidTransform::<TestBackend, 2>::new(translation, rotation, center);

    let points = Tensor::<f32, TestBackend>::from_floats([[0.0, 0.0], [1.0, 1.0]], &device);

    let transformed = transform.transform_points(points);
    let data = transformed.to_data();

    // With no rotation, just translation
    assert_eq!(data.as_slice::<f32>().unwrap()[0], 1.0);
    assert_eq!(data.as_slice::<f32>().unwrap()[1], 2.0);
    assert_eq!(data.as_slice::<f32>().unwrap()[2], 2.0);
    assert_eq!(data.as_slice::<f32>().unwrap()[3], 3.0);
}

#[test]
fn translation_3d_shifts_point_correctly() {
    let device = Default::default();
    let translation = Tensor::<f32, TestBackend>::from_floats([1.0, 2.0, 3.0], &device);
    let rotation = Tensor::<f32, TestBackend>::from_floats([0.0, 0.0, 0.0], &device); // No rotation
    let center = Tensor::<f32, TestBackend>::zeros([3], &device);
    let transform = RigidTransform::<TestBackend, 3>::new(translation, rotation, center);

    let points = Tensor::<f32, TestBackend>::from_floats([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], &device);

    let transformed = transform.transform_points(points);
    let data = transformed.to_data();

    // With no rotation, just translation
    assert_eq!(data.as_slice::<f32>().unwrap()[0], 1.0);
    assert_eq!(data.as_slice::<f32>().unwrap()[1], 2.0);
    assert_eq!(data.as_slice::<f32>().unwrap()[2], 3.0);
    assert_eq!(data.as_slice::<f32>().unwrap()[3], 2.0);
    assert_eq!(data.as_slice::<f32>().unwrap()[4], 3.0);
    assert_eq!(data.as_slice::<f32>().unwrap()[5], 4.0);
}

#[test]
fn rotation_2d_maps_axes_correctly() {
    let device = Default::default();
    let translation = Tensor::<f32, TestBackend>::zeros([2], &device);
    let rotation = Tensor::<f32, TestBackend>::from_floats([std::f32::consts::FRAC_PI_2], &device); // 90 degrees
    let center = Tensor::<f32, TestBackend>::zeros([2], &device);
    let transform = RigidTransform::<TestBackend, 2>::new(translation, rotation, center);

    // Point (1, 0)
    let points = Tensor::<f32, TestBackend>::from_floats([[1.0, 0.0]], &device);

    let transformed = transform.transform_points(points);
    let data = transformed.to_data();
    let slice = data.as_slice::<f32>().unwrap();

    // Should be (0, 1) (approximately)
    assert!((slice[0] - 0.0).abs() < 1e-6);
    assert!((slice[1] - 1.0).abs() < 1e-6);
}

#[test]
fn rotation_3d_z_maps_x_to_y() {
    let device = Default::default();
    let translation = Tensor::<f32, TestBackend>::zeros([3], &device);
    // Rotate 90 deg around Z. Euler: x, y, z. So [0, 0, PI/2]
    let rotation =
        Tensor::<f32, TestBackend>::from_floats([0.0, 0.0, std::f32::consts::FRAC_PI_2], &device);
    let center = Tensor::<f32, TestBackend>::zeros([3], &device);
    let transform = RigidTransform::<TestBackend, 3>::new(translation, rotation, center);

    // Point (1, 0, 0) should become (0, 1, 0)
    let points = Tensor::<f32, TestBackend>::from_floats([[1.0, 0.0, 0.0]], &device);

    let transformed = transform.transform_points(points);
    let data = transformed.to_data();
    let slice = data.as_slice::<f32>().unwrap();

    assert!((slice[0] - 0.0).abs() < 1e-6);
    assert!((slice[1] - 1.0).abs() < 1e-6);
    assert!((slice[2] - 0.0).abs() < 1e-6);
}

#[test]
fn translation_1d_shifts_scalar_correctly() {
    let device = Default::default();
    let translation = Tensor::<f32, TestBackend>::from_floats([1.0], &device);
    let rotation = Tensor::<f32, TestBackend>::zeros([0], &device);
    let center = Tensor::<f32, TestBackend>::zeros([1], &device);
    let transform = RigidTransform::<TestBackend, 1>::new(translation, rotation, center);

    let points = Tensor::<f32, TestBackend>::from_floats([[1.0]], &device);
    let transformed = transform.transform_points(points);
    let val = transformed.into_data().as_slice::<f32>().unwrap()[0];
    assert_eq!(val, 2.0);
}

#[test]
fn translation_4d_shifts_four_coords() {
    let device = Default::default();
    let translation = Tensor::<f32, TestBackend>::from_floats([1.0, 1.0, 1.0, 1.0], &device);
    let rotation = Tensor::<f32, TestBackend>::zeros([6], &device); // Usually 6 for 4D but we ignore it
    let center = Tensor::<f32, TestBackend>::zeros([4], &device);
    let transform = RigidTransform::<TestBackend, 4>::new(translation, rotation, center);

    let points = Tensor::<f32, TestBackend>::from_floats([[0.0, 0.0, 0.0, 0.0]], &device);
    let transformed = transform.transform_points(points);
    let result = transformed.into_data().as_slice::<f32>().unwrap().to_vec();

    assert_eq!(result[0], 1.0);
    assert_eq!(result[1], 1.0);
    assert_eq!(result[2], 1.0);
    assert_eq!(result[3], 1.0);
}
