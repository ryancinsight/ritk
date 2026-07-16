use super::*;
use coeus_core::SequentialBackend;
use coeus_tensor::Tensor;
use ritk_image::tensor::{TensorData};

type TestBackend = SequentialBackend;

#[test]
fn test_linear_interpolator_volumetric_axes() {
    let device = Default::default();
    let data_vec = vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2, 2])),
        &device,
    );

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_floats(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        &device,
    );
    let result = interpolator.interpolate(&data, indices);
    let result_data = result.into_data();
    let slice = result_data.as_slice::<f32>().unwrap();

    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 1.0);
    assert_eq!(slice[2], 10.0);
    assert_eq!(slice[3], 100.0);

    let center = Tensor::<f32, TestBackend>::from_floats([[0.5, 0.5, 0.5]], &device);
    let result_center = interpolator.interpolate(&data, center);
    let center_data = result_center.into_data();
    let center_slice = center_data.as_slice::<f32>().unwrap();

    let expected = (0.0 + 1.0 + 10.0 + 11.0 + 100.0 + 101.0 + 110.0 + 111.0) / 8.0;
    assert!(
        (center_slice[0] - expected).abs() < 1e-5,
        "Expected {}, got {}",
        expected,
        center_slice[0]
    );
}

#[test]
fn test_linear_interpolator_planar() {
    let device = Default::default();
    let data_vec = vec![0.0, 1.0, 10.0, 11.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2])),
        &device,
    );

    let interpolator = LinearInterpolator::new();

    let center = Tensor::<f32, TestBackend>::from_floats([[0.5, 0.5]], &device);
    let result = interpolator.interpolate(&data, center);
    let result_data = result.into_data();
    let slice = result_data.as_slice::<f32>().unwrap();

    let expected = (0.0 + 1.0 + 10.0 + 11.0) / 4.0;
    assert!((slice[0] - expected).abs() < 1e-5);
}

#[test]
fn test_linear_interpolation_at_grid_points() {
    let device = Default::default();
    let data_vec = vec![0.0, 1.0, 2.0, 3.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2])),
        &device,
    );

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_floats(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        &device,
    );
    let result = interpolator.interpolate(&data, indices);
    let result_data = result.into_data();
    let slice = result_data.as_slice::<f32>().unwrap();

    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 1.0);
    assert_eq!(slice[2], 2.0);
    assert_eq!(slice[3], 3.0);
}

#[test]
fn test_linear_interpolator_out_of_bounds() {
    let device = Default::default();
    let data_vec = vec![0.0, 1.0, 2.0, 3.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2])),
        &device,
    );

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_floats([[-1.0, -1.0], [5.0, 5.0]], &device);
    let result = interpolator.interpolate(&data, indices);
    let result_data = result.into_data();
    let slice = result_data.as_slice::<f32>().unwrap();

    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 3.0);
}

#[test]
fn test_linear_interpolator_zero_pad_out_of_bounds() {
    let device = Default::default();
    let data_vec = vec![0.0_f32, 1.0, 2.0, 3.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2])),
        &device,
    );

    let interp = LinearInterpolator::new_zero_pad();

    // Out-of-bounds samples must return 0.0
    let oob = Tensor::<f32, TestBackend>::from_floats([[-1.0, -1.0], [5.0, 5.0]], &device);
    let result = interp.interpolate(&data, oob);
    let slice = result.into_data().as_slice::<f32>().unwrap().to_vec();
    assert!(
        slice[0].abs() < 1e-6,
        "ZeroPad OOB at (-1,-1) should give 0.0, got {}",
        slice[0]
    );
    assert!(
        slice[1].abs() < 1e-6,
        "ZeroPad OOB at (5,5) should give 0.0, got {}",
        slice[1]
    );

    // In-bounds sample at center must match bilinear interpolation
    let center = Tensor::<f32, TestBackend>::from_floats([[0.5, 0.5]], &device);
    let center_val = interp.interpolate(&data, center).into_data();
    let cv = center_val.as_slice::<f32>().unwrap()[0];
    let expected = (0.0_f32 + 1.0 + 2.0 + 3.0) / 4.0;
    assert!(
        (cv - expected).abs() < 1e-5,
        "ZeroPad in-bounds at (0.5,0.5) should give {}, got {}",
        expected,
        cv
    );
}

#[test]
fn test_linear_interpolator_volumetric_zero_pad_out_of_bounds() {
    let device = Default::default();
    let data_vec = vec![0.0_f32, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2, 2])),
        &device,
    );
    let interp = LinearInterpolator::new_zero_pad();

    // Out-of-bounds: far outside volume
    let oob =
        Tensor::<f32, TestBackend>::from_floats([[-5.0, -5.0, -5.0], [10.0, 10.0, 10.0]], &device);
    let result = interp.interpolate(&data, oob);
    let s = result.into_data().as_slice::<f32>().unwrap().to_vec();
    assert!(
        s[0].abs() < 1e-6,
        "3D ZeroPad OOB should give 0.0, got {}",
        s[0]
    );
    assert!(
        s[1].abs() < 1e-6,
        "3D ZeroPad OOB should give 0.0, got {}",
        s[1]
    );

    // In-bounds corner at (0,0,0) should return 0.0 (first element of data)
    let corner = Tensor::<f32, TestBackend>::from_floats([[0.0, 0.0, 0.0]], &device);
    let corner_val = interp.interpolate(&data, corner).into_data();
    let cv = corner_val.as_slice::<f32>().unwrap()[0];
    assert!(
        (cv - 0.0_f32).abs() < 1e-6,
        "3D ZeroPad in-bounds corner should give 0.0, got {}",
        cv
    );
}

#[test]
fn test_linear_interpolator_line() {
    let device = Default::default();
    let data_vec = vec![0.0, 10.0, 20.0, 30.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([4])),
        &device,
    );

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_floats([[0.5]], &device);
    let result = interpolator.interpolate(&data, indices);
    let result_data = result.into_data();
    let slice = result_data.as_slice::<f32>().unwrap();

    assert!((slice[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_linear_interpolator_volumetric_typed() {
    // Const-generic shape specialization (audit §8 351-01).
    // Verifies that `interpolate_3d_typed::<B, 2, 2, 2>` matches the
    // runtime `interpolate_3d` for a 2×2×2 cube with the same query
    // points. The typed version takes the shape as const generics,
    // skipping the runtime `data.shape()` read and constant-folding
    // the bounds checks.
    use super::dim3::interpolate_3d_typed;

    let device = Default::default();
    let data_vec = vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2, 2])),
        &device,
    );

    let indices = Tensor::<f32, TestBackend>::from_floats(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        &device,
    );

    let result = interpolate_3d_typed::<TestBackend, 2, 2, 2>(
        &data,
        indices,
        crate::interpolation::shared::OutOfBoundsMode::Clamp,
    );
    let slice = result.into_data().as_slice::<f32>().unwrap().to_vec();

    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 1.0);
    assert_eq!(slice[2], 10.0);
    assert_eq!(slice[3], 100.0);

    // Center sample: average of all 8 corners.
    let center = Tensor::<f32, TestBackend>::from_floats([[0.5, 0.5, 0.5]], &device);
    let center_val = interpolate_3d_typed::<TestBackend, 2, 2, 2>(
        &data,
        center,
        crate::interpolation::shared::OutOfBoundsMode::Clamp,
    );
    let cv = center_val.into_data().as_slice::<f32>().unwrap()[0];
    let expected = (0.0 + 1.0 + 10.0 + 11.0 + 100.0 + 101.0 + 110.0 + 111.0) / 8.0;
    assert!(
        (cv - expected).abs() < 1e-5,
        "Typed 3-D center should give {}, got {}",
        expected,
        cv
    );
}

#[test]
fn test_linear_interpolator_4d() {
    let device = Default::default();
    let mut data_vec = vec![0.0; 16];
    data_vec[15] = 100.0;

    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2, 2, 2])),
        &device,
    );

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
    let result = interpolator.interpolate(&data, indices);
    let val = result.into_data().as_slice::<f32>().unwrap()[0];
    assert_eq!(val, 100.0);

    let center = Tensor::<f32, TestBackend>::from_floats([[0.5, 0.5, 0.5, 0.5]], &device);
    let result_center = interpolator.interpolate(&data, center);
    let val_center = result_center.into_data().as_slice::<f32>().unwrap()[0];

    assert!((val_center - 6.25).abs() < 1e-5);
}

// ════════════════════════════════════════════════════════════════════
//  Const-generic shape specialization tests (audit §8 351-01)
// ════════════════════════════════════════════════════════════════════

#[test]
fn test_linear_interpolator_line_typed() {
    use super::dim1::interpolate_1d_typed;
    let device = Default::default();
    let data_vec = vec![0.0, 10.0, 20.0, 30.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([4])),
        &device,
    );

    let indices = Tensor::<f32, TestBackend>::from_floats([[0.5]], &device);
    let result = interpolate_1d_typed::<TestBackend, 4>(
        &data,
        indices,
        crate::interpolation::shared::OutOfBoundsMode::Clamp,
    );
    let slice = result.into_data().as_slice::<f32>().unwrap().to_vec();
    assert!(
        (slice[0] - 5.0).abs() < 1e-5,
        "Typed 1-D at 0.5 should give 5.0, got {}",
        slice[0]
    );

    // In-bounds corner at index 2 should return 20.0.
    let corner = Tensor::<f32, TestBackend>::from_floats([[2.0]], &device);
    let corner_val = interpolate_1d_typed::<TestBackend, 4>(
        &data,
        corner,
        crate::interpolation::shared::OutOfBoundsMode::Clamp,
    );
    let cv = corner_val.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (cv - 20.0).abs() < 1e-5,
        "Typed 1-D at index 2 should give 20.0, got {}",
        cv
    );
}

#[test]
fn test_linear_interpolator_planar_typed() {
    use super::dim2::interpolate_2d_typed;
    let device = Default::default();
    let data_vec = vec![0.0, 1.0, 10.0, 11.0];
    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2])),
        &device,
    );

    // Center sample: average of all 4 corners.
    let center = Tensor::<f32, TestBackend>::from_floats([[0.5, 0.5]], &device);
    let center_val = interpolate_2d_typed::<TestBackend, 2, 2>(
        &data,
        center,
        crate::interpolation::shared::OutOfBoundsMode::Clamp,
    );
    let cv = center_val.into_data().as_slice::<f32>().unwrap()[0];
    let expected = (0.0 + 1.0 + 10.0 + 11.0) / 4.0;
    assert!(
        (cv - expected).abs() < 1e-5,
        "Typed 2-D center should give {}, got {}",
        expected,
        cv
    );

    // Grid-point samples.
    let indices = Tensor::<f32, TestBackend>::from_floats(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        &device,
    );
    let result = interpolate_2d_typed::<TestBackend, 2, 2>(
        &data,
        indices,
        crate::interpolation::shared::OutOfBoundsMode::Clamp,
    );
    let slice = result.into_data().as_slice::<f32>().unwrap().to_vec();
    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 1.0);
    assert_eq!(slice[2], 10.0);
    assert_eq!(slice[3], 11.0);
}

#[test]
fn test_linear_interpolator_4d_typed() {
    use super::dim4::interpolate_4d_typed;
    let device = Default::default();
    let mut data_vec = vec![0.0; 16];
    data_vec[15] = 100.0;

    let data = Tensor::<f32, TestBackend>::from_data(
        (data_vec, ritk_image::tensor::([2, 2, 2, 2])),
        &device,
    );

    // Corner at (1,1,1,1) should return 100.0.
    let corner = Tensor::<f32, TestBackend>::from_floats([[1.0, 1.0, 1.0, 1.0]], &device);
    let corner_val = interpolate_4d_typed::<TestBackend, 2, 2, 2, 2>(
        &data,
        corner,
        crate::interpolation::shared::OutOfBoundsMode::Clamp,
    );
    let cv = corner_val.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (cv - 100.0).abs() < 1e-5,
        "Typed 4-D corner (1,1,1,1) should give 100.0, got {}",
        cv
    );

    // Center sample: average of all 16 corners. Only the (1,1,1,1) corner
    // is 100.0, the other 15 are 0.0, so the average is 100.0/16 = 6.25.
    let center = Tensor::<f32, TestBackend>::from_floats([[0.5, 0.5, 0.5, 0.5]], &device);
    let center_val = interpolate_4d_typed::<TestBackend, 2, 2, 2, 2>(
        &data,
        center,
        crate::interpolation::shared::OutOfBoundsMode::Clamp,
    );
    let cv2 = center_val.into_data().as_slice::<f32>().unwrap()[0];
    assert!(
        (cv2 - 6.25).abs() < 1e-5,
        "Typed 4-D center should give 6.25, got {}",
        cv2
    );
}
