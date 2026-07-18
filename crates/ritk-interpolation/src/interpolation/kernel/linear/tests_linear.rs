use super::*;
use coeus_core::MoiraiBackend;
use coeus_tensor::Tensor;

type TestBackend = MoiraiBackend;

#[test]
fn test_linear_interpolator_volumetric_axes() {
    let data_vec = vec![0.0, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
    let data = Tensor::<f32, TestBackend>::from_slice([2, 2, 2], &data_vec);

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_slice(
        [4, 3],
        &[
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
    );
    let result = interpolator.interpolate(&data, indices);
    let slice = result.to_contiguous().as_slice();

    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 1.0);
    assert_eq!(slice[2], 10.0);
    assert_eq!(slice[3], 100.0);

    let center = Tensor::<f32, TestBackend>::from_slice([1, 3], &[0.5, 0.5, 0.5]);
    let result_center = interpolator.interpolate(&data, center);
    let center_slice = result_center.to_contiguous().as_slice();

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
    let data_vec = vec![0.0, 1.0, 10.0, 11.0];
    let data = Tensor::<f32, TestBackend>::from_slice([2, 2], &data_vec);

    let interpolator = LinearInterpolator::new();

    let center = Tensor::<f32, TestBackend>::from_slice([1, 2], &[0.5, 0.5]);
    let result = interpolator.interpolate(&data, center);
    let slice = result.to_contiguous().as_slice();

    let expected = (0.0 + 1.0 + 10.0 + 11.0) / 4.0;
    assert!((slice[0] - expected).abs() < 1e-5);
}

#[test]
fn test_linear_interpolation_at_grid_points() {
    let data_vec = vec![0.0, 1.0, 2.0, 3.0];
    let data = Tensor::<f32, TestBackend>::from_slice([2, 2], &data_vec);

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_slice(
        [4, 2],
        &[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    );
    let result = interpolator.interpolate(&data, indices);
    let slice = result.to_contiguous().as_slice();

    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 1.0);
    assert_eq!(slice[2], 2.0);
    assert_eq!(slice[3], 3.0);
}

#[test]
fn test_linear_interpolator_out_of_bounds() {
    let data_vec = vec![0.0, 1.0, 2.0, 3.0];
    let data = Tensor::<f32, TestBackend>::from_slice([2, 2], &data_vec);

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_slice([2, 2], &[-1.0, -1.0, 5.0, 5.0]);
    let result = interpolator.interpolate(&data, indices);
    let slice = result.to_contiguous().as_slice();

    assert_eq!(slice[0], 0.0);
    assert_eq!(slice[1], 3.0);
}

#[test]
fn test_linear_interpolator_zero_pad_out_of_bounds() {
    let data_vec = vec![0.0_f32, 1.0, 2.0, 3.0];
    let data = Tensor::<f32, TestBackend>::from_slice([2, 2], &data_vec);

    let interp = LinearInterpolator::new_zero_pad();

    // Out-of-bounds samples must return 0.0
    let oob = Tensor::<f32, TestBackend>::from_slice([2, 2], &[-1.0, -1.0, 5.0, 5.0]);
    let result = interp.interpolate(&data, oob);
    let slice = result.to_contiguous().as_slice();
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
    let center = Tensor::<f32, TestBackend>::from_slice([1, 2], &[0.5, 0.5]);
    let center_val = interp.interpolate(&data, center);
    let cv = center_val.to_contiguous().as_slice()[0];
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
    let data_vec = vec![0.0_f32, 1.0, 10.0, 11.0, 100.0, 101.0, 110.0, 111.0];
    let data = Tensor::<f32, TestBackend>::from_slice([2, 2, 2], &data_vec);
    let interp = LinearInterpolator::new_zero_pad();

    // Out-of-bounds: far outside volume
    let oob = Tensor::<f32, TestBackend>::from_slice(
        [2, 3],
        &[-5.0, -5.0, -5.0, 10.0, 10.0, 10.0],
    );
    let result = interp.interpolate(&data, oob);
    let s = result.to_contiguous().as_slice();
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
    let corner = Tensor::<f32, TestBackend>::from_slice([1, 3], &[0.0, 0.0, 0.0]);
    let corner_val = interp.interpolate(&data, corner);
    let cv = corner_val.to_contiguous().as_slice()[0];
    assert!(
        (cv - 0.0_f32).abs() < 1e-6,
        "3D ZeroPad in-bounds corner should give 0.0, got {}",
        cv
    );
}

#[test]
fn test_linear_interpolator_line() {
    let data_vec = vec![0.0, 10.0, 20.0, 30.0];
    let data = Tensor::<f32, TestBackend>::from_slice([4], &data_vec);

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_slice([1, 1], &[0.5]);
    let result = interpolator.interpolate(&data, indices);
    let slice = result.to_contiguous().as_slice();

    assert!((slice[0] - 5.0).abs() < 1e-5);
}

#[test]
fn test_linear_interpolator_4d() {
    let mut data_vec = vec![0.0; 16];
    data_vec[15] = 100.0;

    let data = Tensor::<f32, TestBackend>::from_slice([2, 2, 2, 2], &data_vec);

    let interpolator = LinearInterpolator::new();

    let indices = Tensor::<f32, TestBackend>::from_slice([1, 4], &[1.0, 1.0, 1.0, 1.0]);
    let result = interpolator.interpolate(&data, indices);
    let val = result.to_contiguous().as_slice()[0];
    assert_eq!(val, 100.0);

    let center = Tensor::<f32, TestBackend>::from_slice([1, 4], &[0.5, 0.5, 0.5, 0.5]);
    let result_center = interpolator.interpolate(&data, center);
    let val_center = result_center.to_contiguous().as_slice()[0];

    assert!((val_center - 6.25).abs() < 1e-5);
}
