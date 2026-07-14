//! Differential + analytical coverage for the Coeus-native recursive-Gaussian path.
//!
//! The native wrapper shares the exact `filter_vals` host core (order 0/1/2 plus
//! scale normalization) the Burn path calls, so the differential assertion is
//! bitwise-exact (shared harness). Analytical oracles pin each order's contract
//! directly on the native path.

use crate::native_support::{assert_native_matches_burn, make_native_image, native_vals};
use crate::recursive_gaussian::{DerivativeOrder, RecursiveGaussianFilter};

fn ramp(dims: [usize; 3]) -> Vec<f32> {
    let n = dims[0] * dims[1] * dims[2];
    (0..n)
        .map(|i| (i as f32) * 0.29 - (i % 6) as f32 * 0.8 + 1.0)
        .collect()
}

fn filter(order: DerivativeOrder) -> RecursiveGaussianFilter {
    RecursiveGaussianFilter::new(1.5).with_derivative_order(order)
}

#[test]
fn matches_burn_order_zero() {
    let dims = [7, 6, 5];
    assert_native_matches_burn(
        ramp(dims),
        dims,
        |img| filter(DerivativeOrder::Zero).apply(img).expect("burn"),
        |img, _b| filter(DerivativeOrder::Zero).apply_native(img),
    );
}

#[test]
fn matches_burn_order_first() {
    let dims = [6, 6, 6];
    assert_native_matches_burn(
        ramp(dims),
        dims,
        |img| filter(DerivativeOrder::First).apply(img).expect("burn"),
        |img, _b| filter(DerivativeOrder::First).apply_native(img),
    );
}

#[test]
fn matches_burn_order_second() {
    let dims = [6, 5, 7];
    assert_native_matches_burn(
        ramp(dims),
        dims,
        |img| filter(DerivativeOrder::Second).apply(img).expect("burn"),
        |img, _b| filter(DerivativeOrder::Second).apply_native(img),
    );
}

/// Order-0 smoothing preserves a constant field (unit-gain DC-normalized IIR).
#[test]
fn smoothing_preserves_constant() {
    let dims = [10, 10, 10];
    let c = 6.0_f32;
    let img = make_native_image(vec![c; dims[0] * dims[1] * dims[2]], dims);
    let out = filter(DerivativeOrder::Zero).apply_native(&img).unwrap();
    for v in native_vals(&out) {
        assert!(
            (v - c).abs() < 1e-3,
            "smoothing must preserve a constant, got {v}"
        );
    }
}

/// Order-1 gradient magnitude of a constant field is 0.
#[test]
fn gradient_of_constant_is_zero() {
    let dims = [10, 10, 10];
    let img = make_native_image(vec![9.0_f32; dims[0] * dims[1] * dims[2]], dims);
    let out = filter(DerivativeOrder::First).apply_native(&img).unwrap();
    for v in native_vals(&out) {
        assert!(
            v.abs() < 1e-3,
            "gradient magnitude of a constant must be ~0, got {v}"
        );
    }
}

/// Order-2 Laplacian of a constant field is 0.
#[test]
fn laplacian_of_constant_is_zero() {
    let dims = [10, 10, 10];
    let img = make_native_image(vec![-4.0_f32; dims[0] * dims[1] * dims[2]], dims);
    let out = filter(DerivativeOrder::Second).apply_native(&img).unwrap();
    for v in native_vals(&out) {
        assert!(
            v.abs() < 1e-3,
            "Laplacian of a constant must be ~0, got {v}"
        );
    }
}
