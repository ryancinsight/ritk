//! Tests for trig
//! Extracted to keep the 500-line structural limit.

use super::*;
use ritk_core::filter::ops::extract_vec_infallible;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(vals, Shape::new(dims));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0, 0.0]),
        Spacing::new([1.0_f64, 1.0, 1.0]),
        Direction::identity(),
    )
}

// ── AtanImageFilter ───────────────────────────────────────────────────────

/// Proof: atan(0) = 0 exactly; atan(1) = π/4; atan(−1) = −π/4.
#[test]
fn atan_zero_is_zero() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = AtanImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert_eq!(x, 0.0f32, "atan(0) must equal 0 exactly");
    }
}

/// atan(1) = π/4 ≈ 0.7854.
#[test]
fn atan_one_is_pi_over_four() {
    let img = make_image(vec![1.0f32; 8], [2, 2, 2]);
    let out = AtanImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    let expected = (1.0f32).atan();
    for &x in &v {
        assert!(
            (x - expected).abs() < 1e-6,
            "atan(1) = {x}, expected {expected}"
        );
    }
}

/// Odd function: atan(−x) = −atan(x).
#[test]
fn atan_odd_function() {
    let img = make_image(vec![2.5f32; 8], [2, 2, 2]);
    let pos = AtanImageFilter::new().apply(&img);
    let img_neg = make_image(vec![-2.5f32; 8], [2, 2, 2]);
    let neg = AtanImageFilter::new().apply(&img_neg);
    let (pv, _) = extract_vec_infallible(&pos);
    let (nv, _) = extract_vec_infallible(&neg);
    for (&p, &n) in pv.iter().zip(nv.iter()) {
        assert!((p + n).abs() < 1e-6, "atan not odd: {p} + {n} ≠ 0");
    }
}

/// Range is strictly within (−π/2, π/2).
#[test]
fn atan_range_bounded() {
    let vals: Vec<f32> = (-100..=100).map(|i| i as f32).collect();
    let n = vals.len();
    let img = make_image(vals, [1, 1, n]);
    let out = AtanImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    let pi_half = std::f32::consts::FRAC_PI_2;
    for &x in &v {
        assert!(x > -pi_half && x < pi_half, "atan out of range: {x}");
    }
}

/// Spatial metadata is preserved exactly.
#[test]
fn atan_preserves_metadata() {
    let img = make_image(vec![1.0f32; 27], [3, 3, 3]);
    let out = AtanImageFilter::new().apply(&img);
    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.direction(), img.direction());
    assert_eq!(out.shape(), img.shape());
}

// ── SinImageFilter ────────────────────────────────────────────────────────

/// sin(0) = 0 exactly.
#[test]
fn sin_zero_is_zero() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = SinImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!((x - 0.0f32).abs() < 1e-6, "sin(0) must equal 0");
    }
}

/// sin(π/2) = 1.0 exactly.
#[test]
fn sin_pi_over_2_is_one() {
    let pi_half = std::f32::consts::FRAC_PI_2;
    let img = make_image(vec![pi_half; 8], [2, 2, 2]);
    let out = SinImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!((x - 1.0f32).abs() < 1e-6, "sin(π/2) = {x}");
    }
}

/// Output always in [−1, 1] for real inputs.
#[test]
fn sin_range_bounded() {
    let vals: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.1).collect();
    let n = vals.len();
    let img = make_image(vals, [1, 1, n]);
    let out = SinImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!(
            (-1.0 - 1e-6..=1.0 + 1e-6).contains(&x),
            "sin out of [−1,1]: {x}"
        );
    }
}

// ── CosImageFilter ────────────────────────────────────────────────────────

/// cos(0) = 1.0 exactly.
#[test]
fn cos_zero_is_one() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = CosImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!((x - 1.0f32).abs() < 1e-6, "cos(0) = {x}");
    }
}

/// cos(π) = −1.0.
#[test]
fn cos_pi_is_minus_one() {
    let pi = std::f32::consts::PI;
    let img = make_image(vec![pi; 8], [2, 2, 2]);
    let out = CosImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!((x - (-1.0f32)).abs() < 1e-5, "cos(π) = {x}");
    }
}

/// sin²(x) + cos²(x) = 1 (Pythagorean identity).
#[test]
fn sin_cos_pythagorean_identity() {
    let vals: Vec<f32> = (0..8)
        .map(|i| i as f32 * std::f32::consts::PI / 8.0)
        .collect();
    let img = make_image(vals.clone(), [2, 2, 2]);
    let img2 = make_image(vals, [2, 2, 2]);
    let sins = SinImageFilter::new().apply(&img);
    let coss = CosImageFilter::new().apply(&img2);
    let (sv, _) = extract_vec_infallible(&sins);
    let (cv, _) = extract_vec_infallible(&coss);
    for (&s, &c) in sv.iter().zip(cv.iter()) {
        let sum = s * s + c * c;
        assert!((sum - 1.0f32).abs() < 1e-5, "sin²+cos²={sum}");
    }
}

// ── TanImageFilter ────────────────────────────────────────────────────────

/// tan(0) = 0 exactly.
#[test]
fn tan_zero_is_zero() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = TanImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!((x - 0.0f32).abs() < 1e-6, "tan(0) = {x}");
    }
}

/// tan(π/4) = 1.0.
#[test]
fn tan_pi_over_4_is_one() {
    let pi_4 = std::f32::consts::FRAC_PI_4;
    let img = make_image(vec![pi_4; 8], [2, 2, 2]);
    let out = TanImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!((x - 1.0f32).abs() < 1e-5, "tan(π/4) = {x}");
    }
}

// ── AsinImageFilter ───────────────────────────────────────────────────────

/// asin(0) = 0, asin(1) = π/2, asin(−1) = −π/2.
#[test]
fn asin_boundary_values() {
    let pi_half = std::f32::consts::FRAC_PI_2;
    let img = make_image(
        vec![0.0f32, 1.0, -1.0, 0.5, -0.5, 0.0, 1.0, -1.0],
        [2, 2, 2],
    );
    let out = AsinImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    assert!((v[0] - 0.0f32).abs() < 1e-6, "asin(0) = {}", v[0]);
    assert!((v[1] - pi_half).abs() < 1e-5, "asin(1) = {}", v[1]);
    assert!((v[2] - (-pi_half)).abs() < 1e-5, "asin(-1) = {}", v[2]);
}

/// asin + acos = π/2 (complement identity).
#[test]
fn asin_acos_complement_identity() {
    let pi_half = std::f32::consts::FRAC_PI_2;
    let vals = vec![0.0f32, 0.5, -0.5, 0.8, -0.8, 1.0, -1.0, 0.0];
    let img = make_image(vals.clone(), [2, 2, 2]);
    let img2 = make_image(vals, [2, 2, 2]);
    let asins = AsinImageFilter::new().apply(&img);
    let acoss = AcosImageFilter::new().apply(&img2);
    let (asv, _) = extract_vec_infallible(&asins);
    let (acv, _) = extract_vec_infallible(&acoss);
    for (&a, &b) in asv.iter().zip(acv.iter()) {
        assert!((a + b - pi_half).abs() < 1e-5, "asin+acos={} ≠ π/2", a + b);
    }
}

// ── AcosImageFilter ───────────────────────────────────────────────────────

/// acos(1) = 0, acos(0) = π/2, acos(−1) = π.
#[test]
fn acos_boundary_values() {
    let pi = std::f32::consts::PI;
    let pi_half = std::f32::consts::FRAC_PI_2;
    let img = make_image(vec![1.0f32, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0], [2, 2, 2]);
    let out = AcosImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    assert!((v[0] - 0.0f32).abs() < 1e-5, "acos(1) = {}", v[0]);
    assert!((v[1] - pi_half).abs() < 1e-5, "acos(0) = {}", v[1]);
    assert!((v[2] - pi).abs() < 1e-5, "acos(-1) = {}", v[2]);
}

// ── BoundedReciprocalImageFilter ──────────────────────────────────────────

/// At zero: out = 1.0.
#[test]
fn bounded_reciprocal_zero_is_one() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = BoundedReciprocalImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert_eq!(x, 1.0f32, "1/(1+|0|) must equal 1.0");
    }
}

/// At x=1: out = 0.5. At x=−1: out = 0.5 (symmetric in |x|).
#[test]
fn bounded_reciprocal_at_one_is_half() {
    let img = make_image(
        vec![1.0f32, -1.0f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        [2, 2, 2],
    );
    let out = BoundedReciprocalImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!((x - 0.5f32).abs() < 1e-6, "1/(1+1) = {x}");
    }
}

/// Output strictly in (0, 1] for arbitrary inputs.
#[test]
fn bounded_reciprocal_range() {
    let vals: Vec<f32> = (-50..=50).map(|i| i as f32).collect();
    let n = vals.len();
    let img = make_image(vals, [1, 1, n]);
    let out = BoundedReciprocalImageFilter::new().apply(&img);
    let (v, _) = extract_vec_infallible(&out);
    for &x in &v {
        assert!(x > 0.0 && x <= 1.0, "bounded reciprocal out of (0,1]: {x}");
    }
}

/// Monotone decreasing: |x| < |y| ⟹ out(x) > out(y).
#[test]
fn bounded_reciprocal_monotone() {
    let img_small = make_image(vec![1.0f32; 8], [2, 2, 2]);
    let img_large = make_image(vec![10.0f32; 8], [2, 2, 2]);
    let out_small = BoundedReciprocalImageFilter::new().apply(&img_small);
    let out_large = BoundedReciprocalImageFilter::new().apply(&img_large);
    let (sv, _) = extract_vec_infallible(&out_small);
    let (lv, _) = extract_vec_infallible(&out_large);
    for (&s, &l) in sv.iter().zip(lv.iter()) {
        assert!(s > l, "bounded_reciprocal not monotone: {s} <= {l}");
    }
}

/// Spatial metadata preserved by BoundedReciprocal.
#[test]
fn bounded_reciprocal_preserves_metadata() {
    let img = make_image(vec![3.0f32; 27], [3, 3, 3]);
    let out = BoundedReciprocalImageFilter::new().apply(&img);
    assert_eq!(out.origin(), img.origin());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.direction(), img.direction());
    assert_eq!(out.shape(), img.shape());
}
