//! Tests for zero_crossing
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::native_support::{make_native_image, native_vals};
use coeus_core::SequentialBackend;

#[test]
fn zero_crossing_detects_sign_change_across_boundary() {
    // 1×1×3: [-1, +1, +1] — crossing between ix=0 and ix=1, equal magnitude.
    // ITK marks only the near-zero side; on a |·|-tie it resolves toward the
    // forward voxel, so ix=0 (whose +neighbour crosses) is marked, ix=1 is not.
    let img = make_native_image(vec![-1.0, 1.0, 1.0], [1, 1, 3]);
    let out = ZeroCrossingImageFilter::new().apply_native(&img, &SequentialBackend)
        .expect("apply_native should succeed");
    let v = native_vals(&out);
    // ix=0: forward neighbour ix=1 = +1, sign change, |−1| <= |1| → crossing.
    assert_eq!(v[0], 1.0, "ix=0 should be zero-crossing");
    // ix=1: backward neighbour ix=0 = -1, |1| < |−1| is false (tie) → background.
    assert_eq!(
        v[1], 0.0,
        "ix=1 should be background (tie resolved to ix=0)"
    );
    // ix=2: only neighbour ix=1 is +1, no sign change → background.
    assert_eq!(v[2], 0.0, "ix=2 should be background");
}

#[test]
fn zero_crossing_exact_zero_is_foreground() {
    // 1×1×3: [1, 0, 1] — middle voxel is exactly 0 → crossing
    let img = make_native_image(vec![1.0, 0.0, 1.0], [1, 1, 3]);
    let out = ZeroCrossingImageFilter::new().apply_native(&img, &SequentialBackend)
        .expect("apply_native should succeed");
    let v = native_vals(&out);
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
    let img = make_native_image(vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], [2, 2, 2]);
    let out = ZeroCrossingImageFilter::new().apply_native(&img, &SequentialBackend)
        .expect("apply_native should succeed");
    let v = native_vals(&out);
    for (i, &x) in v.iter().enumerate() {
        assert_eq!(x, 0.0, "voxel {} should be background, got {}", i, x);
    }
}

#[test]
fn zero_crossing_custom_foreground_background_values() {
    // 1×1×2: [-1, +1] — equal-magnitude crossing; ITK marks only the forward
    // side (ix=0), leaving ix=1 as background.
    let img = make_native_image(vec![-1.0, 1.0], [1, 1, 2]);
    let out = ZeroCrossingImageFilter::new()
        .with_foreground(255.0)
        .with_background(-1.0)
        .apply_native(&img, &SequentialBackend)
        .expect("apply_native should succeed");
    let v = native_vals(&out);
    assert_eq!(v[0], 255.0, "forward side of the crossing is foreground");
    assert_eq!(v[1], -1.0, "backward side of the tie is background");
}

#[test]
fn zero_crossing_preserves_spatial_metadata() {
    let img = make_native_image(vec![-1.0, 1.0, -1.0, 1.0], [1, 2, 2]);
    let out = ZeroCrossingImageFilter::new().apply_native(&img, &SequentialBackend)
        .expect("apply_native should succeed");
    assert_eq!(out.shape(), img.shape());
    assert_eq!(out.spacing(), img.spacing());
    assert_eq!(out.origin(), img.origin());
}

#[test]
fn zero_crossing_boundary_voxel_no_oob_crossing() {
    // 1×1×2: [1, 2] — both positive, no sign change
    // Boundary voxels only consider in-bounds neighbours
    let img = make_native_image(vec![1.0, 2.0], [1, 1, 2]);
    let out = ZeroCrossingImageFilter::new().apply_native(&img, &SequentialBackend)
        .expect("apply_native should succeed");
    let v = native_vals(&out);
    assert_eq!(
        v[0], 0.0,
        "boundary voxel with no sign-change neighbour is background"
    );
    assert_eq!(
        v[1], 0.0,
        "boundary voxel with no sign-change neighbour is background"
    );
}
