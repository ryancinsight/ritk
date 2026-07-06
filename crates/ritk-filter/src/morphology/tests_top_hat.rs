//! Tests for top_hat
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};
type B = NdArray<f32>;
fn img(v: Vec<f32>, d: [usize; 3]) -> Image<B, 3> {
    let t = Tensor::<B, 3>::from_data(TensorData::new(v, Shape::new(d)), &Default::default());
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}
fn vv(i: &Image<B, 3>) -> Vec<f32> {
    i.data_slice().into_owned()
}

#[test]
fn test_wth_constant_zero() {
    let d = [8, 8, 8];
    let n = d[0] * d[1] * d[2];
    let out = vv(&WhiteTopHatFilter::new(1)
        .apply(&img(vec![5.0; n], d))
        .unwrap());
    for &v in &out {
        assert!(v.abs() < 1e-5, "WTH(const)={v}");
    }
}
#[test]
fn test_wth_bright_spike() {
    let d = [9, 9, 9];
    let [nz, ny, nx] = d;
    let n = nz * ny * nx;
    let bg = 2.0_f32;
    let mut v = vec![bg; n];
    let c = 4 * ny * nx + 4 * nx + 4;
    v[c] = 10.0;
    let out = vv(&WhiteTopHatFilter::new(1).apply(&img(v, d)).unwrap());
    assert!(out[c] > 1.0, "WTH spike not detected: {}", out[c]);
    for &x in &out {
        assert!(x >= 0.0, "WTH non-negative");
    }
}
#[test]
fn test_wth_radius_zero() {
    let d = [6, 6, 6];
    let n = d[0] * d[1] * d[2];
    let v: Vec<f32> = (0..n).map(|i| (i % 10) as f32).collect();
    let out = vv(&WhiteTopHatFilter::new(0).apply(&img(v, d)).unwrap());
    for &x in &out {
        assert!(x.abs() < 1e-5, "WTH(r=0)={x}");
    }
}
#[test]
fn test_wth_metadata() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; n], Shape::new(d)),
        &Default::default(),
    );
    let o = Point::new([1.0, 2.0, 3.0]);
    let s = Spacing::new([0.5, 0.5, 0.5]);
    let r = WhiteTopHatFilter::new(1)
        .apply(&Image::new(t, o, s, Direction::identity()))
        .unwrap();
    assert_eq!(*r.origin(), o);
    assert_eq!(*r.spacing(), s);
}
#[test]
fn test_wth_non_negative() {
    let d = [8, 8, 8];
    let n = d[0] * d[1] * d[2];
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.7 + 1.0) % 20.0).collect();
    let out = vv(&WhiteTopHatFilter::new(1).apply(&img(v, d)).unwrap());
    for &x in &out {
        assert!(x >= -1e-5, "WTH neg: {x}");
    }
}
#[test]
fn test_bth_constant_zero() {
    let d = [8, 8, 8];
    let n = d[0] * d[1] * d[2];
    let out = vv(&BlackTopHatFilter::new(1)
        .apply(&img(vec![5.0; n], d))
        .unwrap());
    for &v in &out {
        assert!(v.abs() < 1e-5, "BTH(const)={v}");
    }
}
#[test]
fn test_bth_dark_hole() {
    let d = [9, 9, 9];
    let [nz, ny, nx] = d;
    let n = nz * ny * nx;
    let bg = 10.0_f32;
    let mut v = vec![bg; n];
    let c = 4 * ny * nx + 4 * nx + 4;
    v[c] = 2.0;
    let out = vv(&BlackTopHatFilter::new(1).apply(&img(v, d)).unwrap());
    assert!(out[c] > 1.0, "BTH hole not detected: {}", out[c]);
    for &x in &out {
        assert!(x >= 0.0, "BTH non-negative");
    }
}
#[test]
fn test_bth_radius_zero() {
    let d = [6, 6, 6];
    let n = d[0] * d[1] * d[2];
    let v: Vec<f32> = (0..n).map(|i| (i % 10) as f32).collect();
    let out = vv(&BlackTopHatFilter::new(0).apply(&img(v, d)).unwrap());
    for &x in &out {
        assert!(x.abs() < 1e-5, "BTH(r=0)={x}");
    }
}
#[test]
fn test_bth_metadata() {
    let d = [5, 5, 5];
    let n = d[0] * d[1] * d[2];
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; n], Shape::new(d)),
        &Default::default(),
    );
    let o = Point::new([1.0, 2.0, 3.0]);
    let s = Spacing::new([0.5, 0.5, 0.5]);
    let r = BlackTopHatFilter::new(1)
        .apply(&Image::new(t, o, s, Direction::identity()))
        .unwrap();
    assert_eq!(*r.origin(), o);
    assert_eq!(*r.spacing(), s);
}
#[test]
fn test_bth_non_negative() {
    let d = [8, 8, 8];
    let n = d[0] * d[1] * d[2];
    let v: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 + 1.0) % 15.0).collect();
    let out = vv(&BlackTopHatFilter::new(1).apply(&img(v, d)).unwrap());
    for &x in &out {
        assert!(x >= -1e-5, "BTH neg: {x}");
    }
}
