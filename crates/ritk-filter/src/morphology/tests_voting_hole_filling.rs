use super::*;
use ritk_image::test_support as ts;

type B = burn_ndarray::NdArray<f32>;

fn filt() -> VotingBinaryHoleFillingImageFilter {
    VotingBinaryHoleFillingImageFilter::new([1, 1, 1], 1, 1.0, 0.0)
}

/// A single-voxel hole surrounded by foreground is filled (z=1: 8 in-plane fg →
/// 24 clamped fg ≥ threshold 14).
#[test]
fn fills_surrounded_hole() {
    let mut v = vec![1.0f32; 25];
    v[2 * 5 + 2] = 0.0; // hole at (y=2, x=2)
    let img = ts::make_image::<B, 3>(v, [1, 5, 5]);
    let out = filt().apply(&img);
    assert_eq!(out.data_slice().into_owned()[2 * 5 + 2], 1.0);
}

/// A lone background voxel with too few foreground neighbours is not filled.
#[test]
fn sparse_background_not_filled() {
    // Only 3 foreground voxels near (2,2): 3·3(z) = 9 < threshold 14.
    let mut v = vec![0.0f32; 25];
    v[1 * 5 + 1] = 1.0;
    v[1 * 5 + 2] = 1.0;
    v[2 * 5 + 1] = 1.0;
    let img = ts::make_image::<B, 3>(v, [1, 5, 5]);
    let out = filt().apply(&img);
    assert_eq!(out.data_slice().into_owned()[2 * 5 + 2], 0.0);
}

/// Foreground is never removed (even an isolated foreground voxel survives).
#[test]
fn foreground_always_survives() {
    let mut v = vec![0.0f32; 25];
    v[2 * 5 + 2] = 1.0;
    let img = ts::make_image::<B, 3>(v, [1, 5, 5]);
    let out = filt().apply(&img);
    assert_eq!(out.data_slice().into_owned()[2 * 5 + 2], 1.0);
}

/// Corner background voxel fills under clamp (replicate) boundary: in-bounds fg
/// neighbours give a clamped count of 15 ≥ 14 (a zero boundary would give 9).
#[test]
fn corner_fills_under_clamp_boundary() {
    let mut v = vec![1.0f32; 16];
    v[0] = 0.0; // corner (0,0)
    let img = ts::make_image::<B, 3>(v, [1, 4, 4]);
    let out = filt().apply(&img);
    assert_eq!(out.data_slice().into_owned()[0], 1.0);
}
