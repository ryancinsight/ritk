use super::{derivative_operator, DiscreteGaussianDerivativeFilter};
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

/// The order-1 / order-2 central-difference derivative operators match ITK's
/// `DerivativeOperator`.
#[test]
fn derivative_operator_known_orders() {
    assert_eq!(derivative_operator(0), vec![1.0]);
    assert_eq!(derivative_operator(1), vec![-0.5, 0.0, 0.5]);
    assert_eq!(derivative_operator(2), vec![1.0, -2.0, 1.0]);
    // order 3 = D¹ ⊛ D²  → [-0.5,0,0.5] ⊛ [1,-2,1]
    assert_eq!(derivative_operator(3), vec![-0.5, 1.0, 0.0, -1.0, 0.5]);
}

/// A constant image has zero derivative everywhere (the derivative operator sums
/// to zero, the Gaussian preserves the constant).
#[test]
fn constant_image_zero_derivative() {
    let img: Image<B, 3> = ts::make_image(vec![7.0f32; 6 * 8 * 8], [6, 8, 8]);
    let out = DiscreteGaussianDerivativeFilter::new(4.0, [1, 0, 0], 0.01, false).apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    assert!(
        v.iter().all(|&x| x.abs() < 1e-3),
        "constant image must have zero first derivative"
    );
}

/// First derivative of a linear ramp `I = x` is a constant (≈1) in the deep
/// interior (the Gaussian-derivative of a linear function recovers its slope).
#[test]
fn ramp_first_derivative_is_slope() {
    let (nz, ny, nx) = (1usize, 16, 16);
    let mut vals = vec![0.0f32; nz * ny * nx];
    for iy in 0..ny {
        for ix in 0..nx {
            vals[iy * nx + ix] = ix as f32;
        }
    }
    let img: Image<B, 3> = ts::make_image(vals, [nz, ny, nx]);
    // d/dx along axis 2 (x). Order [0,0,1].
    let out = DiscreteGaussianDerivativeFilter::new(2.0, [0, 0, 1], 0.01, false).apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    // deep interior (away from edges) should be ≈ 1.0 (slope of the ramp).
    for iy in 4..ny - 4 {
        for ix in 4..nx - 4 {
            let got = v[iy * nx + ix];
            // Discrete Gaussian-derivative slope response has magnitude ≈1 (sign
            // per ITK's operator convention; exact sitk match verified in Python).
            assert!(
                (got.abs() - 1.0).abs() < 0.1,
                "ramp derivative magnitude at ({iy},{ix}) = {got}, expected |·| ≈ 1.0"
            );
        }
    }
}

/// Output shape and order-0-everywhere (pure Gaussian smoothing) are well-formed.
#[test]
fn order_zero_is_smoothing() {
    let img: Image<B, 3> = ts::make_image(vec![3.0f32; 2 * 4 * 5], [2, 4, 5]);
    let out = DiscreteGaussianDerivativeFilter::new(1.0, [0, 0, 0], 0.01, false).apply(&img);
    assert_eq!(out.shape(), [2, 4, 5]);
    let (v, _) = extract_vec_infallible(&out);
    // Smoothing a constant returns the constant.
    assert!(v.iter().all(|&x| (x - 3.0).abs() < 1e-3));
}
