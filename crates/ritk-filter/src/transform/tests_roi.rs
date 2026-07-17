//! Tests for roi
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = LegacyBurnBackend;

fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::burn_compat::make_image::<B, 3>(vals, shape)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data().clone().into_data().into_vec::<f32>().unwrap()
}

#[test]
fn roi_full_image_is_identity() {
    // ROI spanning the whole volume = identical copy
    let vals: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let img = make_image(vals.clone(), [3, 3, 3]);
    let out = RegionOfInterestImageFilter::new([0, 0, 0], [3, 3, 3])
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [3, 3, 3]);
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip(vals.iter()).enumerate() {
        assert_eq!(a, b, "voxel {}: identity violation", i);
    }
}

#[test]
fn roi_extracts_correct_sub_volume() {
    // 3×3×3 with values 1..27; extract [1..3, 1..3, 1..3] (2×2×2)
    let vals: Vec<f32> = (1..=27).map(|x| x as f32).collect();
    let img = make_image(vals, [3, 3, 3]);
    let out = RegionOfInterestImageFilter::new([1, 1, 1], [2, 2, 2])
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [2, 2, 2]);
    let v = voxels(&out);
    // Voxel (1,1,1) in 3×3×3 = index 1*9 + 1*3 + 1 = 13 → value 14
    assert_eq!(v[0], 14.0, "first voxel of sub-volume mismatch");
    // Voxel (1,1,2) = 1*9 + 1*3 + 2 = 14 → value 15
    assert_eq!(v[1], 15.0);
    // Voxel (2,2,2) = 2*9 + 2*3 + 2 = 26 → value 27
    assert_eq!(v[7], 27.0, "last voxel of sub-volume mismatch");
}

#[test]
fn roi_single_voxel_extract() {
    let vals: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let img = make_image(vals, [2, 2, 2]);
    // Extract voxel (1,1,1) — index 7, value 8
    let out = RegionOfInterestImageFilter::new([1, 1, 1], [1, 1, 1])
        .apply(&img)
        .unwrap();
    assert_eq!(out.shape(), [1, 1, 1]);
    let v = voxels(&out);
    assert_eq!(v[0], 8.0, "expected voxel (1,1,1) = 8");
}

#[test]
fn roi_updates_origin_with_identity_direction() {
    // With identity direction and spacing=[2,3,4], start=[1,1,1]:
    // new_origin[0] = 0 + 1*2*1 = 2
    // new_origin[1] = 0 + 1*3*1 = 3
    // new_origin[2] = 0 + 1*4*1 = 4
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    let device: <B as ritk_image::tensor::Backend>::Device = Default::default();
    let td = TensorData::new(vec![0.0f32; 27], Shape::new([3, 3, 3]));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    let img = Image::new(
        tensor,
        Point::new([0.0f64, 0.0, 0.0]),
        Spacing::new([2.0f64, 3.0, 4.0]),
        Direction::identity(),
    );
    let out = RegionOfInterestImageFilter::new([1, 1, 1], [2, 2, 2])
        .apply(&img)
        .unwrap();
    let o = out.origin();
    assert!(
        (o[0] - 2.0).abs() < 1e-9,
        "origin[0] expected 2.0 got {}",
        o[0]
    );
    assert!(
        (o[1] - 3.0).abs() < 1e-9,
        "origin[1] expected 3.0 got {}",
        o[1]
    );
    assert!(
        (o[2] - 4.0).abs() < 1e-9,
        "origin[2] expected 4.0 got {}",
        o[2]
    );
}

#[test]
fn roi_out_of_bounds_returns_error() {
    let img = make_image(vec![1.0; 8], [2, 2, 2]);
    // start=[0,0,0], size=[3,2,2] → Z range [0..3) exceeds depth 2
    let result = RegionOfInterestImageFilter::new([0, 0, 0], [3, 2, 2]).apply(&img);
    assert!(result.is_err(), "out-of-bounds ROI must return Err");
}

#[test]
fn native_roi_extracts_value_and_translates_origin() {
    use coeus_core::SequentialBackend;
    use ritk_image::native::Image as NativeImage;

    let image = NativeImage::from_flat_on(
        (1..=8).map(|value| value as f32).collect(),
        [2, 2, 2],
        Point::new([0.0; 3]),
        Spacing::new([2.0, 3.0, 4.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = RegionOfInterestImageFilter::new([1, 1, 1], [1, 1, 1])
        .apply_native(&image, &SequentialBackend)
        .expect("native ROI succeeds");
    assert_eq!(output.data_slice().expect("contiguous output"), &[8.0]);
    assert_eq!(
        [output.origin()[0], output.origin()[1], output.origin()[2]],
        [2.0, 3.0, 4.0]
    );
}
