//! Tests for paste
//! Extracted to keep the 500-line structural limit.
#![allow(clippy::identity_op, clippy::erasing_op)]
use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_tensor_ops::extract_vec_infallible;

type B = LegacyBurnBackend;

fn make_image(vals: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    ts::burn_compat::make_image::<B, 3>(vals, shape)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

#[test]
fn paste_writes_source_into_destination() {
    // dest: 3×3×3 all zeros; source: 1×1×1 = [99]; paste at [1,1,1]
    let dest = make_image(vec![0.0f32; 27], [3, 3, 3]);
    let src = make_image(vec![99.0f32], [1, 1, 1]);
    let out = PasteImageFilter::new([1, 1, 1]).apply(&dest, &src).unwrap();
    let v = voxels(&out);
    let pasted_idx = 1 * 9 + 1 * 3 + 1;
    assert_eq!(v[pasted_idx], 99.0, "pasted voxel value");
    // All other voxels remain 0
    for (i, &x) in v.iter().enumerate() {
        if i != pasted_idx {
            assert_eq!(x, 0.0, "voxel {} should be 0, got {}", i, x);
        }
    }
}

#[test]
fn paste_at_origin_replaces_corner() {
    let dest = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let src = make_image(vec![5.0, 6.0, 7.0, 8.0], [1, 2, 2]);
    let out = PasteImageFilter::new([0, 0, 0]).apply(&dest, &src).unwrap();
    let v = voxels(&out);
    assert_eq!(v[0], 5.0);
    assert_eq!(v[1], 6.0);
    assert_eq!(v[2], 7.0);
    assert_eq!(v[3], 8.0);
    // Second z-slice unchanged
    assert_eq!(v[4], 0.0);
    assert_eq!(v[5], 0.0);
    assert_eq!(v[6], 0.0);
    assert_eq!(v[7], 0.0);
}

#[test]
fn paste_preserves_dest_spatial_metadata() {
    let dest = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let src = make_image(vec![1.0f32], [1, 1, 1]);
    let out = PasteImageFilter::new([0, 0, 0]).apply(&dest, &src).unwrap();
    assert_eq!(out.shape(), dest.shape());
    assert_eq!(out.origin(), dest.origin());
    assert_eq!(out.spacing(), dest.spacing());
}

#[test]
fn paste_out_of_bounds_returns_error() {
    let dest = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let src = make_image(vec![1.0f32; 4], [1, 2, 2]);
    // dest_start=[1,0,0]: Z extent [1..2) OK, but source is [1,2,2] → Z [1..2) OK
    // Increase to [1,1,0] → Y extent [1..3) exceeds height 2 → error
    let r = PasteImageFilter::new([1, 1, 0]).apply(&dest, &src);
    assert!(r.is_err(), "out-of-bounds paste must return Err");
}

#[test]
fn paste_full_source_into_full_dest_replaces_all() {
    let dest = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let src_vals: Vec<f32> = (1..=8).map(|x| x as f32).collect();
    let src = make_image(src_vals.clone(), [2, 2, 2]);
    let out = PasteImageFilter::new([0, 0, 0]).apply(&dest, &src).unwrap();
    let v = voxels(&out);
    for (i, (&a, &b)) in v.iter().zip(src_vals.iter()).enumerate() {
        assert_eq!(a, b, "voxel {}: expected {} got {}", i, b, a);
    }
}
