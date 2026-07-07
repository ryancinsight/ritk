use super::*;
use crate::native_support::{make_native_image, native_vals};
use coeus_core::SequentialBackend;

/// Non-negative image: abs is identity.
#[test]
fn abs_nonneg_is_identity() {
    let img = make_native_image(vec![0.0, 1.0, 2.5, 10.0], [1, 2, 2]);
    let out = AbsImageFilter::new()
        .apply_native(&img, &SequentialBackend)
        .unwrap();
    let v = native_vals(&out);
    assert_eq!(
        v,
        vec![0.0_f32, 1.0, 2.5, 10.0],
        "non-negative input must be unchanged by abs"
    );
}

/// Negative values become positive.
#[test]
fn abs_negates_negative_voxels() {
    let img = make_native_image(vec![-3.0, -1.0, 0.0, 2.0], [1, 2, 2]);
    let out = AbsImageFilter::new()
        .apply_native(&img, &SequentialBackend)
        .unwrap();
    let v = native_vals(&out);
    assert_eq!(
        v,
        vec![3.0_f32, 1.0, 0.0, 2.0],
        "abs must negate each negative voxel: [-3,-1,0,2] → [3,1,0,2]"
    );
}

/// All-negative image: every output is the negation of input.
#[test]
fn abs_all_negative_all_positive() {
    let img = make_native_image(vec![-5.0, -2.0, -0.5], [1, 1, 3]);
    let out = AbsImageFilter::new()
        .apply_native(&img, &SequentialBackend)
        .unwrap();
    for &v in native_vals(&out).iter() {
        assert!(v >= 0.0, "abs output must be non-negative; got {v}");
    }
}

/// Spatial metadata is preserved.
#[test]
fn abs_preserves_metadata() {
    use ritk_image::native::Image;
    use ritk_spatial::{Direction, Point, Spacing};
    let sp = Spacing::new([2.0, 3.0, 4.0]);
    let img = Image::from_flat_on(
        vec![1.0_f32, -1.0],
        [1, 1, 2],
        Point::new([0.0, 0.0, 0.0]),
        sp,
        Direction::identity(),
        &SequentialBackend,
    )
    .unwrap();
    let out = AbsImageFilter::new()
        .apply_native(&img, &SequentialBackend)
        .unwrap();
    assert_eq!(out.spacing(), img.spacing(), "spacing must be preserved");
}

/// Constant positive image: unchanged.
#[test]
fn abs_constant_positive_unchanged() {
    let img = make_native_image(vec![7.0, 7.0, 7.0], [1, 1, 3]);
    let out = AbsImageFilter::new()
        .apply_native(&img, &SequentialBackend)
        .unwrap();
    for &v in native_vals(&out).iter() {
        assert_eq!(v, 7.0_f32, "constant positive image unchanged by abs");
    }
}
