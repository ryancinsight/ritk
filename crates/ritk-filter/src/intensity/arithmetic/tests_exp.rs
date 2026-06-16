use super::*;
use crate::intensity::arithmetic::LogImageFilter;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_core::image::Image;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(data, shape)
}

fn vals(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// exp(0) = 1.
#[test]
fn exp_of_zero_is_one() {
    let img = make_image(vec![0.0, 0.0], [1, 1, 2]);
    let out = ExpImageFilter::new().apply(&img);
    for &v in vals(&out).iter() {
        assert!((v - 1.0).abs() < 1e-6, "exp(0) = 1; got {v}");
    }
}

/// exp(1) ≈ e.
#[test]
fn exp_of_one_is_e() {
    let e = std::f32::consts::E;
    let img = make_image(vec![1.0], [1, 1, 1]);
    let out = ExpImageFilter::new().apply(&img);
    let v = vals(&out)[0];
    assert!((v - e).abs() < 1e-5, "exp(1) must be ≈ e = {e}; got {v}");
}

/// exp(2) ≈ e².
#[test]
fn exp_of_two_is_e_squared() {
    let e2 = std::f32::consts::E * std::f32::consts::E;
    let img = make_image(vec![2.0], [1, 1, 1]);
    let out = ExpImageFilter::new().apply(&img);
    let v = vals(&out)[0];
    assert!(
        (v - e2).abs() < 1e-3,
        "exp(2) must be ≈ e² ≈ {e2:.4}; got {v}"
    );
}

/// Output is always positive for finite inputs.
#[test]
fn exp_output_always_positive() {
    let img = make_image(vec![-5.0, -1.0, 0.0, 1.0, 5.0], [1, 1, 5]);
    let out = ExpImageFilter::new().apply(&img);
    for &v in vals(&out).iter() {
        assert!(v > 0.0, "exp(x) must be > 0 for finite x; got {v}");
    }
}

/// Spatial metadata is preserved.
#[test]
fn exp_preserves_metadata() {
    let sp = Spacing::new([0.5, 0.5, 0.5]);
    let device: burn_ndarray::NdArrayDevice = Default::default();
    let td = TensorData::new(vec![0.0_f32], Shape::new([1usize, 1, 1]));
    let t = Tensor::<B, 3>::from_data(td, &device);
    let img = Image::new(t, Point::new([0.0, 0.0, 0.0]), sp, Direction::identity());
    let out = ExpImageFilter::new().apply(&img);
    assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
}

/// Log ∘ Exp ≈ identity (round-trip within f32 precision).
#[test]
fn log_exp_roundtrip() {
    let vals_in = vec![0.5_f32, 1.0, 2.0];
    let img = make_image(vals_in.clone(), [1, 1, 3]);
    let exp_out = ExpImageFilter::new().apply(&img);
    let log_out = LogImageFilter::new().apply(&exp_out);
    let v = vals(&log_out);
    for (a, b) in v.iter().zip(vals_in.iter()) {
        assert!(
            (a - b).abs() < 1e-5,
            "ln(exp(x)) round-trip: got {a}, expected {b}"
        );
    }
}
