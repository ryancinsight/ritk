use super::*;
use burn::tensor::{ElementConversion, Tensor};
use burn_ndarray::NdArray;

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
    let interp = BSplineInterpolator::new().with_bounds_policy(BoundsPolicy::ZeroPad);
    assert_eq!(interp.bounds_policy, BoundsPolicy::ZeroPad);
    let interp2 = BSplineInterpolator::new_zero_pad().with_bounds_policy(BoundsPolicy::Extend);
    assert_eq!(interp2.bounds_policy, BoundsPolicy::Extend);
}

// ---- batch correctness + performance smoke tests -------------------------

/// Batched interpolation of interior integer grid points on a linear ramp.
/// B-spline without pre-filtering reproduces linear fields exactly when all
/// 4 support samples in each axis are in-bounds (coord ≥ 1).
#[test]
fn test_bspline_3d_batch_correctness() {
    let device = Default::default();
    let n = 8usize;
    let mut data_vec: Vec<f32> = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                data_vec.push((i + j + k) as f32);
            }
        }
    }
    let data = Tensor::<TestBackend, 3>::from_data(TensorData::new(data_vec, [n, n, n]), &device);
    let interp = BSplineInterpolator::new();

    // Interior coords only (c ≥ 1) to avoid boundary renormalization.
    let test_coords: &[(usize, usize, usize)] =
        &[(1, 1, 1), (2, 1, 1), (1, 2, 1), (1, 1, 2), (3, 3, 3)];
    let mut pts: Vec<f32> = Vec::new();
    for &(i, j, k) in test_coords {
        pts.extend_from_slice(&[i as f32, j as f32, k as f32]);
    }
    let n_pts = test_coords.len();
    let indices = Tensor::<TestBackend, 2>::from_data(TensorData::new(pts, [n_pts, 3]), &device);

    let result = interp.interpolate(&data, indices);
    let vals = result.into_data().as_slice::<f32>().unwrap().to_vec();
    for (idx, &(i, j, k)) in test_coords.iter().enumerate() {
        let expected = (i + j + k) as f32;
        assert!(
            (vals[idx] - expected).abs() < 1e-3,
            "At ({},{},{}) expected {}, got {}",
            i,
            j,
            k,
            expected,
            vals[idx]
        );
    }
}

/// 2D version of the linear ramp batch test.
#[test]
fn test_bspline_2d_batch_correctness() {
    let device = Default::default();
    let n = 6usize;
    let mut data_vec: Vec<f32> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            data_vec.push((i + j) as f32);
        }
    }
    let data = Tensor::<TestBackend, 2>::from_data(TensorData::new(data_vec, [n, n]), &device);
    let interp = BSplineInterpolator::new();

    // Interior coords only (c ≥ 1).
    let test_coords: &[(usize, usize)] = &[(1, 1), (2, 1), (1, 2), (2, 3)];
    let mut pts: Vec<f32> = Vec::new();
    for &(i, j) in test_coords {
        pts.extend_from_slice(&[i as f32, j as f32]);
    }
    let n_pts = test_coords.len();
    let indices = Tensor::<TestBackend, 2>::from_data(TensorData::new(pts, [n_pts, 2]), &device);

    let result = interp.interpolate(&data, indices);
    let vals = result.into_data().as_slice::<f32>().unwrap().to_vec();
    for (idx, &(i, j)) in test_coords.iter().enumerate() {
        let expected = (i + j) as f32;
        assert!(
            (vals[idx] - expected).abs() < 1e-3,
            "At ({},{}) expected {}, got {}",
            i,
            j,
            expected,
            vals[idx]
        );
    }
}

/// Empty index batch must return an empty tensor without panic.
#[test]
fn test_bspline_empty_indices() {
    let device = Default::default();
    let data = Tensor::<TestBackend, 3>::from_floats(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        &device,
    );
    let interp = BSplineInterpolator::new();
    let indices =
        Tensor::<TestBackend, 2>::from_data(TensorData::new(Vec::<f32>::new(), [0, 3]), &device);
    let result = interp.interpolate(&data, indices);
    assert_eq!(result.dims()[0], 0);
}

/// Performance regression guard — 1000 points on a 64³ volume must complete
/// in under 5 s in debug mode (the original implementation took ~33 s).
/// Run with `cargo test -- bspline_perf --ignored --nocapture` to see timing.
#[test]
#[ignore = "performance measurement; run explicitly"]
fn test_bspline_3d_perf_regression() {
    use std::time::Instant;

    let device = Default::default();
    let n = 64usize;
    let mut data_vec: Vec<f32> = Vec::with_capacity(n * n * n);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                data_vec.push((i + j + k) as f32);
            }
        }
    }
    let data = Tensor::<TestBackend, 3>::from_data(TensorData::new(data_vec, [n, n, n]), &device);

    // Build 1000 random-ish interior points.
    let n_pts = 1000usize;
    let mut pts: Vec<f32> = Vec::with_capacity(n_pts * 3);
    for p in 0..n_pts {
        let c = (p % (n - 2) + 1) as f32;
        pts.extend_from_slice(&[c, c, c]);
    }
    let indices = Tensor::<TestBackend, 2>::from_data(TensorData::new(pts, [n_pts, 3]), &device);

    let interp = BSplineInterpolator::new();
    let t0 = Instant::now();
    let result = interp.interpolate(&data, indices);
    let elapsed = t0.elapsed();

    // Consume result to prevent dead-code elimination.
    let sum: f32 = result.into_data().as_slice::<f32>().unwrap().iter().sum();
    println!(
        "1000-point 64³ BSpline (debug): {:.3}s sum={sum:.1}",
        elapsed.as_secs_f32()
    );
    assert!(
        elapsed.as_secs_f32() < 5.0,
        "BSpline 1000-pt 64³ took {:.2}s (regression threshold: 5s in debug mode)",
        elapsed.as_secs_f32()
    );
}
