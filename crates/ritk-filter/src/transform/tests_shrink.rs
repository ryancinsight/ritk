//! Tests for shrink
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec_infallible;

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, shape: [usize; 3], spacing: [f64; 3]) -> Image<B, 3> {
    ts::make_image_with_spacing::<B, 3>(data, shape, spacing)
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

/// Factor [1,1,1] → identity (same shape and values).
#[test]
fn factor_one_is_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = make_image(data.clone(), [2, 3, 4], [1.0, 1.0, 1.0]);
    let out = ShrinkImageFilter::new([1, 1, 1]).apply(&img).unwrap();
    assert_eq!(out.shape(), [2, 3, 4]);
    let v = voxels(&out);
    for (a, b) in v.iter().zip(data.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

/// 1-D shrink by 2: mean of pairs.
/// [0, 2, 4, 6] with factor [1,1,2] → [1, 5] (means of [0,2] and [4,6]).
#[test]
fn shrink_x_by_2() {
    let img = make_image(vec![0.0, 2.0, 4.0, 6.0], [1, 1, 4], [1.0, 1.0, 1.0]);
    let out = ShrinkImageFilter::new([1, 1, 2]).apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 2]);
    let v = voxels(&out);
    assert!((v[0] - 1.0).abs() < 1e-5, "v[0]={}", v[0]);
    assert!((v[1] - 5.0).abs() < 1e-5, "v[1]={}", v[1]);
}

/// Output spacing scales by factor.
#[test]
fn output_spacing_scales() {
    let img = make_image(vec![1.0f32; 8], [2, 2, 2], [1.0, 2.0, 3.0]);
    let out = ShrinkImageFilter::new([2, 2, 2]).apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 1]);
    let s = out.spacing();
    assert!((s[0] - 2.0).abs() < 1e-10, "sz={}", s[0]);
    assert!((s[1] - 4.0).abs() < 1e-10, "sy={}", s[1]);
    assert!((s[2] - 6.0).abs() < 1e-10, "sx={}", s[2]);
}

/// Shrink 4×4×4 by [2,2,2] → 2×2×2, each output voxel = mean of 2×2×2 tile of constant image.
#[test]
fn constant_image_mean_preserved() {
    let img = make_image(vec![5.0f32; 64], [4, 4, 4], [1.0, 1.0, 1.0]);
    let out = ShrinkImageFilter::new([2, 2, 2]).apply(&img).unwrap();
    assert_eq!(out.shape(), [2, 2, 2]);
    let v = voxels(&out);
    for &x in &v {
        assert!((x - 5.0).abs() < 1e-5, "expected 5.0 got {x}");
    }
}

/// Odd input size with even factor: ceil division.
/// 1×1×5 with factor [1,1,2] → shape [1,1,3]: means of [0,1], [2,3], [4].
#[test]
fn odd_size_ceil_division() {
    let img = make_image(vec![0.0, 1.0, 2.0, 3.0, 4.0], [1, 1, 5], [1.0, 1.0, 1.0]);
    let out = ShrinkImageFilter::new([1, 1, 2]).apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 3]);
    let v = voxels(&out);
    assert!((v[0] - 0.5).abs() < 1e-5, "v[0]={}", v[0]); // mean(0,1)
    assert!((v[1] - 2.5).abs() < 1e-5, "v[1]={}", v[1]); // mean(2,3)
    assert!((v[2] - 4.0).abs() < 1e-5, "v[2]={}", v[2]); // only 4
}
