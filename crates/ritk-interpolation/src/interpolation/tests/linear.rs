use crate::interpolation::kernel::bspline::cubic_bspline;
use crate::interpolation::kernel::BoundsPolicy;
use crate::interpolation::BSplineInterpolator;
use burn::tensor::{ElementConversion, Tensor};
use burn_ndarray::NdArray;
use ritk_core::interpolation::Interpolator;

type TestBackend = NdArray<f32>;

#[test]
fn test_bspline_3d() {
    let device = Default::default();
    // Create a simple 3D volume
    let data = Tensor::<TestBackend, 3>::from_floats(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &device,
    );
    let interpolator = BSplineInterpolator::new();

    // Test at exact grid point
    // Note: Without B-spline pre-filtering, the interpolated value at grid points
    // may differ from original data due to the convolution with the B-spline kernel
    let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0, 0.0]], &device);
    let result = interpolator.interpolate(&data, indices);
    let val = result.into_scalar().elem::<f32>();
    // Value should be within reasonable range (cubic B-spline center coefficient is 2/3)
    assert!(
        (0.0..=8.0).contains(&val),
        "Interpolated value {} out of range",
        val
    );

    // Test at interpolated point
    let indices = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
    let result = interpolator.interpolate(&data, indices);
    let val = result.into_scalar().elem::<f32>();
    // Value should be between min and max
    assert!(
        (0.0..=8.0).contains(&val),
        "Interpolated value {} out of range",
        val
    );
}

#[test]
fn test_bspline_2d() {
    let device = Default::default();
    // Create a simple 2D image
    let data = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0], [3.0, 4.0]], &device);
    let interpolator = BSplineInterpolator::new();

    // Test at exact grid point
    // Note: Without B-spline pre-filtering, the interpolated value at grid points
    // may differ from original data due to the convolution with the B-spline kernel
    let indices = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0]], &device);
    let result = interpolator.interpolate(&data, indices);
    let val = result.into_scalar().elem::<f32>();
    // Value should be within reasonable range
    assert!(
        (0.0..=5.0).contains(&val),
        "Interpolated value {} out of range",
        val
    );
}

#[test]
fn test_bspline_basis() {
    // Test B-Spline basis properties
    assert!((cubic_bspline(0.0) - 2.0 / 3.0).abs() < 1e-6);
    assert!(cubic_bspline(1.0) > 0.0);
    assert_eq!(cubic_bspline(2.0), 0.0);
    assert_eq!(cubic_bspline(-2.0), 0.0);
    assert_eq!(cubic_bspline(3.0), 0.0);
    // Symmetry
    assert!((cubic_bspline(0.5) - cubic_bspline(-0.5)).abs() < 1e-6);
}

// ---- zero_pad tests ------------------------------------------------

#[test]
fn test_bspline_zero_pad_3d_oob_returns_zero() {
    let device = Default::default();
    let data = Tensor::<TestBackend, 3>::from_floats(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &device,
    );
    let interp = BSplineInterpolator::new_zero_pad();

    // Clearly out-of-bounds queries in each direction.
    let oob = Tensor::<TestBackend, 2>::from_floats(
        [
            [-5.0, 0.0, 0.0], // dim0 OOB negative
            [10.0, 0.0, 0.0], // dim0 OOB positive
            [0.0, -5.0, 0.0], // dim1 OOB
            [0.0, 0.0, 10.0], // dim2 OOB
        ],
        &device,
    );
    let result = interp.interpolate(&data, oob);
    let s = result.into_data().as_slice::<f32>().unwrap().to_vec();
    for (i, v) in s.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "OOB 3D sample {} should give 0.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn test_bspline_zero_pad_3d_inbounds_matches_no_pad() {
    // In-bounds queries should produce the same result regardless of zero_pad flag.
    let device = Default::default();
    let data = Tensor::<TestBackend, 3>::from_floats(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &device,
    );
    let interp_pad = BSplineInterpolator::new_zero_pad();
    let interp_nop = BSplineInterpolator::new();

    // Interior point; floor coords are (0,0,0) which is in-bounds.
    let pt = Tensor::<TestBackend, 2>::from_floats([[0.5, 0.5, 0.5]], &device);
    let val_pad = interp_pad
        .interpolate(&data, pt.clone())
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    let val_nop = interp_nop
        .interpolate(&data, pt)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(
        (val_pad - val_nop).abs() < 1e-5,
        "In-bounds zero_pad {} vs no-pad {} should match",
        val_pad,
        val_nop
    );
    assert!(
        (0.0..=8.0).contains(&val_pad),
        "In-bounds value {} out of range",
        val_pad
    );
}

#[test]
fn test_bspline_zero_pad_2d_oob_returns_zero() {
    let device = Default::default();
    let data = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    let interp = BSplineInterpolator::new_zero_pad();

    // OOB in both dimensions.
    let oob = Tensor::<TestBackend, 2>::from_floats(
        [[-1.0, 0.0], [0.0, -1.0], [10.0, 0.0], [0.0, 10.0]],
        &device,
    );
    let result = interp.interpolate(&data, oob);
    let s = result.into_data().as_slice::<f32>().unwrap().to_vec();
    for (i, v) in s.iter().enumerate() {
        assert!(
            v.abs() < 1e-6,
            "OOB 2D sample {} should give 0.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn test_bspline_no_zero_pad_oob_gives_finite_value() {
    // Without zero_pad, a query just outside the boundary should still
    // produce a finite (non-panic) value thanks to weight renormalization
    // of the in-bounds neighborhood samples.
    let device = Default::default();
    let data = Tensor::<TestBackend, 3>::from_floats(
        [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]],
        &device,
    );
    let interp = BSplineInterpolator::new(); // zero_pad = false

    // Query just outside: floor(-0.1) = -1 (OOB in dim0).
    // The kernel neighbourhood still touches in-bounds samples at indices 0,1,2
    // (clipped from the 4-wide support), so weight_sum > 0 and result is finite.
    let pt = Tensor::<TestBackend, 2>::from_floats([[-0.1, 0.5, 0.5]], &device);
    let val = interp
        .interpolate(&data, pt)
        .into_data()
        .as_slice::<f32>()
        .unwrap()[0];
    assert!(
        val.is_finite(),
        "No-zero-pad OOB should return finite value, got {}",
        val
    );
}

#[test]
fn test_bspline_with_zero_pad_builder() {
    let interp_extend = BSplineInterpolator::new().with_bounds_policy(BoundsPolicy::Extend);
    let interp_zero = BSplineInterpolator::new().with_bounds_policy(BoundsPolicy::ZeroPad);
    assert_eq!(interp_extend.bounds_policy, BoundsPolicy::Extend);
    assert_eq!(interp_zero.bounds_policy, BoundsPolicy::ZeroPad);

    let interp_default = BSplineInterpolator::default();
    assert_eq!(
        interp_default.bounds_policy,
        BoundsPolicy::Extend,
        "Default should have bounds_policy=Extend"
    );
}

/// Performance regression test: verify B-spline interpolation completes
/// within expected time for a reasonable volume size.
/// This guards against unintentional performance regressions from
/// the flat-data optimization (which reduced allocations by ~64x for 3D).
#[test]
fn test_bspline_performance_regression_3d() {
    let device = Default::default();
    // 64x64x64 volume - large enough to be meaningful, small enough to be fast
    let size = 64usize;
    let n_voxels = size * size * size;
    let data: Vec<f32> = (0..n_voxels).map(|i| (i % 256) as f32).collect();
    let data_tensor = Tensor::<TestBackend, 3>::from_data(
        burn::tensor::TensorData::new(data, [size, size, size]),
        &device,
    );

    let interpolator = BSplineInterpolator::new();

    // Sample at 1000 random points
    let n_points = 1000;
    let mut indices_data = vec![0.0f32; n_points * 3];
    for i in 0..n_points {
        indices_data[i * 3] = (rand::random::<f32>() * size as f32) - 0.5;
        indices_data[i * 3 + 1] = (rand::random::<f32>() * size as f32) - 0.5;
        indices_data[i * 3 + 2] = (rand::random::<f32>() * size as f32) - 0.5;
    }
    let indices = Tensor::<TestBackend, 2>::from_data(
        burn::tensor::TensorData::new(indices_data, [n_points, 3]),
        &device,
    );

    // Time the interpolation
    use std::time::Instant;
    let start = Instant::now();
    let result = interpolator.interpolate(&data_tensor, indices);
    let duration = start.elapsed();

    // Result should have n_points values
    assert_eq!(result.dims()[0], n_points);

    // Performance assertion: 1000 points on 64^3 volume should complete
    // in under 1 second on typical hardware. This guards against regressions.
    // The flat-data optimization made this ~64x faster than the original
    // clone().slice().reshape() approach.
    // Allow 5 seconds for very slow CI environments.
    assert!(
        duration.as_secs_f32() < 5.0,
        "B-spline interpolation performance regression: {:.2}s for {} points on {}x{}x{} volume",
        duration.as_secs_f32(),
        n_points,
        size,
        size,
        size
    );
}
