use super::*;
use coeus_core::SequentialBackend;
use ritk_core::transform::Transform;
type TestBackend = SequentialBackend;

#[test]
fn test_bspline_transform_creation() {
    let device = Default::default();
    let grid_size = [4, 4, 4];
    let origin = Tensor::<f32, TestBackend>::zeros([3], &device);
    let spacing = Tensor::<f32, TestBackend>::from_floats([1.0, 1.0, 1.0], &device);
    let direction_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    let direction = Tensor::<f32, TestBackend>::from_data(
        (direction_data, ([3, 3])),
        &device,
    );

    let num_control_points = grid_size.iter().product();
    let coefficients = Tensor::<f32, TestBackend>::zeros([num_control_points, 3], &device);

    let transform = BSplineTransform::<TestBackend, 3>::new(
        grid_size,
        origin,
        spacing,
        direction,
        coefficients,
    );

    assert_eq!(transform.grid_size(), grid_size);
}

#[test]
fn test_bspline_transform_from_spatial() {
    let device = Default::default();
    let grid_size = [4, 4, 4];
    let origin = ritk_core::spatial::Point3::origin();
    let spacing = ritk_core::spatial::Spacing3::uniform(10.0);
    let direction = ritk_core::spatial::Direction3::identity();

    let num_control_points = grid_size.iter().product();
    let coefficients = Tensor::<f32, TestBackend>::zeros([num_control_points, 3], &device);

    let transform = BSplineTransform::<TestBackend, 3>::from_spatial(
        grid_size,
        &origin,
        &spacing,
        &direction,
        coefficients,
        &device,
    );

    assert_eq!(transform.grid_size(), grid_size);
}

#[test]
fn zero_coefficients_is_identity_planar() {
    let device = Default::default();
    // 4x4 grid
    let grid_size = [4, 4];
    let origin = Tensor::<f32, TestBackend>::zeros([2], &device);
    let spacing = Tensor::<f32, TestBackend>::from_floats([10.0, 10.0], &device);
    let direction_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let direction = Tensor::<f32, TestBackend>::from_data(
        (direction_data, ([2, 2])),
        &device,
    );

    let num_control_points = 16; // 4*4

    // Displace the control point at (1, 1) by (1.0, 1.0)
    // Index 5
    let mut coeffs_data = vec![0.0; num_control_points * 2];
    coeffs_data[5 * 2] = 1.0;
    coeffs_data[5 * 2 + 1] = 1.0;

    let coefficients = Tensor::from_floats(
        ritk_image::tensor::(
            coeffs_data,
            ritk_image::tensor::([num_control_points, 2]),
        ),
        &device,
    );

    let transform = BSplineTransform::<TestBackend, 2>::new(
        grid_size,
        origin,
        spacing,
        direction,
        coefficients,
    );

    // Point at (10.0, 10.0) corresponds to index (1.0, 1.0)
    let points = Tensor::<f32, TestBackend>::from_floats([[10.0, 10.0]], &device);
    let transformed = transform.transform_points(points);
    let result = transformed.into_data().as_slice::<f32>().unwrap().to_vec();

    // Expected: 10.0 + 4/9
    let expected_disp = 4.0 / 9.0;
    assert!((result[0] - (10.0 + expected_disp)).abs() < 1e-5);
    assert!((result[1] - (10.0 + expected_disp)).abs() < 1e-5);
}

#[test]
fn test_bspline_transform_out_of_bounds() {
    let device = Default::default();
    let grid_size = [4, 4];
    let origin = Tensor::<f32, TestBackend>::zeros([2], &device);
    let spacing = Tensor::<f32, TestBackend>::from_floats([10.0, 10.0], &device);
    let direction_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let direction = Tensor::<f32, TestBackend>::from_data(
        (direction_data, ([2, 2])),
        &device,
    );

    let num_control_points = 16;
    let coefficients = Tensor::<f32, TestBackend>::zeros([num_control_points, 2], &device);

    let transform = BSplineTransform::<TestBackend, 2>::new(
        grid_size,
        origin,
        spacing,
        direction,
        coefficients,
    );

    // Point outside grid should remain unchanged (zero displacement)
    let points = Tensor::<f32, TestBackend>::from_floats([[100.0, 100.0]], &device);
    let transformed = transform.transform_points(points);
    let result = transformed.into_data().as_slice::<f32>().unwrap().to_vec();

    assert!((result[0] - 100.0).abs() < 1e-5);
    assert!((result[1] - 100.0).abs() < 1e-5);
}

#[test]
fn zero_coefficients_is_identity_scalar() {
    let device = Default::default();
    let grid_size = [4];
    let origin = Tensor::<f32, TestBackend>::zeros([1], &device);
    let spacing = Tensor::<f32, TestBackend>::from_floats([10.0], &device);
    let direction = Tensor::<f32, TestBackend>::eye(1, &device);

    let num_control_points = 4;
    let mut coeffs_data = vec![0.0; num_control_points];
    // Index 1 (x=1) -> 2.0.
    coeffs_data[1] = 2.0;

    let coefficients = Tensor::from_floats(
        ritk_image::tensor::(
            coeffs_data,
            ritk_image::tensor::([num_control_points, 1]),
        ),
        &device,
    );

    let transform = BSplineTransform::<TestBackend, 1>::new(
        grid_size,
        origin,
        spacing,
        direction,
        coefficients,
    );

    // Point at 10.0 corresponds to index 1.0.
    // B-spline basis for index 1.0: B0(1)=1/6(0)=0, B1(1)=...
    // Wait, index u=1.0. u is local.
    // grid coord = 1.0. floor=1. u=0.0.
    // Basis at u=0.0: B0=1/6, B1=4/6, B2=1/6, B3=0.
    // Indices involved: floor-1, floor, floor+1, floor+2 => 0, 1, 2, 3.
    // Coefficients: c0=0, c1=2.0, c2=0, c3=0.
    // Displacement = c0*B0 + c1*B1 + c2*B2 + c3*B3
    // = 0*1/6 + 2.0*4/6 + 0*1/6 + 0
    // = 8/6 = 4/3 = 1.3333...

    let points = Tensor::<f32, TestBackend>::from_floats([[10.0]], &device);
    let transformed = transform.transform_points(points);
    let result = transformed.into_data().as_slice::<f32>().unwrap()[0];

    assert!((result - (10.0 + 4.0 / 3.0)).abs() < 1e-5);
}

#[test]
fn four_dim_grid_constructs_without_error() {
    let device = Default::default();
    let grid_size = [4, 4, 4, 4];
    let origin = Tensor::<f32, TestBackend>::zeros([4], &device);
    let spacing = Tensor::<f32, TestBackend>::ones([4], &device);
    let direction = Tensor::<f32, TestBackend>::eye(4, &device);

    let num_control_points = 256;
    let coefficients = Tensor::<f32, TestBackend>::zeros([num_control_points, 4], &device);

    let transform = BSplineTransform::<TestBackend, 4>::new(
        grid_size,
        origin,
        spacing,
        direction,
        coefficients,
    );

    assert_eq!(transform.grid_size(), grid_size);

    // Basic transform test (identity)
    let points = Tensor::<f32, TestBackend>::from_floats([[1.5, 1.5, 1.5, 1.5]], &device);
    let transformed = transform.transform_points(points.clone());
    let result = transformed.sub(points).abs().sum();
    let diff = result.into_data().as_slice::<f32>().unwrap()[0];
    assert!(diff < 1e-6);
}
