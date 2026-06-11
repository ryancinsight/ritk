use super::*;
use crate::filter::ops::extract_vec_infallible;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0, 0.0]),
        Spacing::new([1.0_f64, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

// ── ConstantPadImageFilter tests ──────────────────────────────────────────

/// Zero-padding: padded voxels filled with 0.
#[test]
fn constant_pad_zero() {
    // 1×1×2 image [3,7], pad by 1 on each side → 1×1×4 [0,3,7,0].
    let img = make_image(vec![3.0, 7.0], [1, 1, 2]);
    let out = ConstantPadImageFilter::new(Padding::new([0, 0, 1]), Padding::new([0, 0, 1]), 0.0)
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 4]);
    let v = voxels(&out);
    assert_eq!(v, vec![0.0, 3.0, 7.0, 0.0]);
}

/// Custom constant pad value.
#[test]
fn constant_pad_custom_value() {
    let img = make_image(vec![5.0], [1, 1, 1]);
    let out = ConstantPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 2]), -1.0)
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 5]);
    let v = voxels(&out);
    assert_eq!(v, vec![-1.0, -1.0, 5.0, -1.0, -1.0]);
}

/// Constant pad preserves spacing, updates origin.
#[test]
fn constant_pad_origin_updated() {
    let device: <B as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(vec![1.0f32], Shape::new([1, 1, 1]));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    // Origin at [0, 0, 10], spacing [1, 1, 2] — pad 1 voxel on lower X.
    let img2 = Image::new(
        tensor,
        Point::new([0.0_f64, 0.0, 10.0]),
        Spacing::new([1.0_f64, 1.0, 2.0]),
        Direction::identity(),
    );
    let out = ConstantPadImageFilter::new(Padding::new([0, 0, 1]), Padding::new([0, 0, 0]), 0.0)
        .apply(&img2)
        .unwrap();
    // Origin x (axis 2) shifts by -1 * spacing[2] = -1 * 2.0 = -2.0 → new origin[2] = 10 - 2 = 8.
    let ox = out.origin()[2];
    assert!((ox - 8.0).abs() < 1e-10, "origin[2]={ox}");
    // Origin z (axis 0) unchanged (pad_lower[0] = 0).
    assert!((out.origin()[0]).abs() < 1e-10, "origin[0] should be 0");
}

// ── MirrorPadImageFilter tests ────────────────────────────────────────────

/// Mirror pad: 1×1×3 = [1,2,3], pad 2 on each side → [3,2,1,2,3,2,1].
#[test]
fn mirror_pad_1d() {
    let img = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
    let out = MirrorPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 2]))
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 7]);
    let v = voxels(&out);
    // Expected: period=4, mirror extension:
    // index -2 → mirror_index(-2,3): i=-2, period=4, r=((-2%4)+4)%4=2, r<3 → 2 → val=3
    // index -1 → r=3, 3>=3 → 4-3=1 → val=2
    // index 0..2 → original [1,2,3]
    // index 3 → r=3, 3>=3 → 4-3=1 → val=2
    // index 4 → r=0 → val=1
    assert!((v[0] - 3.0).abs() < 1e-5, "v[0]={}", v[0]);
    assert!((v[1] - 2.0).abs() < 1e-5, "v[1]={}", v[1]);
    assert!((v[2] - 1.0).abs() < 1e-5, "v[2]={}", v[2]);
    assert!((v[3] - 2.0).abs() < 1e-5, "v[3]={}", v[3]);
    assert!((v[4] - 3.0).abs() < 1e-5, "v[4]={}", v[4]);
    assert!((v[5] - 2.0).abs() < 1e-5, "v[5]={}", v[5]);
    assert!((v[6] - 1.0).abs() < 1e-5, "v[6]={}", v[6]);
}

/// Mirror index formula for n=1 always returns 0.
#[test]
fn mirror_index_n1() {
    for i in -5i64..=5 {
        assert_eq!(super::mirror_index(i, 1), 0);
    }
}

// ── WrapPadImageFilter tests ──────────────────────────────────────────────

/// Wrap pad: 1×1×3 = [A,B,C], pad 2 on each side → [B,C,A,B,C,A,B].
#[test]
fn wrap_pad_1d() {
    let img = make_image(vec![10.0, 20.0, 30.0], [1, 1, 3]);
    let out = WrapPadImageFilter::new(Padding::new([0, 0, 2]), Padding::new([0, 0, 2]))
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 7]);
    let v = voxels(&out);
    // index shifts: output i → input wrap(i-2, 3)
    // i=0: wrap(-2,3)=1 → 20
    // i=1: wrap(-1,3)=2 → 30
    // i=2: wrap(0,3)=0 → 10
    // i=3: wrap(1,3)=1 → 20
    // i=4: wrap(2,3)=2 → 30
    // i=5: wrap(3,3)=0 → 10
    // i=6: wrap(4,3)=1 → 20
    assert!((v[0] - 20.0).abs() < 1e-5, "v[0]={}", v[0]);
    assert!((v[2] - 10.0).abs() < 1e-5, "v[2]={}", v[2]);
    assert!((v[5] - 10.0).abs() < 1e-5, "v[5]={}", v[5]);
}

/// Output shape correct for wrap pad.
#[test]
fn wrap_pad_shape() {
    let img = make_image(vec![0.0f32; 24], [2, 3, 4]);
    let out = WrapPadImageFilter::new(Padding::new([1, 2, 3]), Padding::new([1, 2, 3]))
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [4, 7, 10]);
}
