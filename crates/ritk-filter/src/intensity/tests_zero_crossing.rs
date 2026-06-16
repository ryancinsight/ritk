//! Tests for zero_crossing
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, shape)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

#[test]
fn zero_crossing_detects_sign_change_across_boundary() {
    // 1×1×3: [-1, +1, +1] — crossing between ix=0 and ix=1
    let img = make_image(vec![-1.0, 1.0, 1.0], [1, 1, 3]);
    let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
    let v = voxels(&out);
    // ix=0: neighbour ix=1 is +1, -1 * +1 < 0 → crossing
    assert_eq!(v[0], 1.0, "ix=0 should be zero-crossing");
    // ix=1: neighbour ix=0 is -1, +1 * -1 < 0 → crossing
    assert_eq!(v[1], 1.0, "ix=1 should be zero-crossing");
    // ix=2: only neighbour ix=1 is +1, +1 * +1 > 0 → not crossing
    assert_eq!(v[2], 0.0, "ix=2 should be background");
}

#[test]
fn zero_crossing_exact_zero_is_foreground() {
    // 1×1×3: [1, 0, 1] — middle voxel is exactly 0 → crossing
    let img = make_image(vec![1.0, 0.0, 1.0], [1, 1, 3]);
    let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
    let v = voxels(&out);
    assert_eq!(v[1], 1.0, "exact-zero voxel must be foreground");
    assert_eq!(
        v[0], 0.0,
        "positive voxel with only positive neighbours is background"
    );
    assert_eq!(
        v[2], 0.0,
        "positive voxel with only positive neighbours is background"
    );
}

#[test]
fn zero_crossing_uniform_positive_no_crossings() {
    let img = make_image(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2, 2, 2]);
    let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
    let v = voxels(&out);
    for (i, &x) in v.iter().enumerate() {
        assert_eq!(x, 0.0, "voxel {} should be background, got {}", i, x);
    }
}

#[test]
fn zero_crossing_custom_foreground_background_values() {
    // 1×1×2: [-1, +1] — both voxels are crossings
    let img = make_image(vec![-1.0, 1.0], [1, 1, 2]);
    let out = ZeroCrossingImageFilter::new()
        .with_foreground(255.0)
        .with_background(-1.0)
        .apply(&img)
        .unwrap();
    let v = voxels(&out);
    assert_eq!(v[0], 255.0);
    assert_eq!(v[1], 255.0);
}

#[test]
fn zero_crossing_preserves_spatial_metadata() {
    let img = make_image(vec![-1.0, 1.0, -1.0, 1.0], [1, 2, 2]);
    let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
    assert_eq!(out.shape(), img.shape());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.origin(), img.origin());
}

#[test]
fn zero_crossing_boundary_voxel_no_oob_crossing() {
    // 1×1×2: [1, 2] — both positive, no sign change
    // Boundary voxels only consider in-bounds neighbours
    let img = make_image(vec![1.0, 2.0], [1, 1, 2]);
    let out = ZeroCrossingImageFilter::new().apply(&img).unwrap();
    let v = voxels(&out);
    assert_eq!(
        v[0], 0.0,
        "boundary voxel with no sign-change neighbour is background"
    );
    assert_eq!(
        v[1], 0.0,
        "boundary voxel with no sign-change neighbour is background"
    );
}
