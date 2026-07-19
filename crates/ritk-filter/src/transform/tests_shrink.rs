//! Tests for shrink
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_image::test_support as ts;
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3], spacing: [f64; 3]) -> Image<f32, B, 3> {
    ts::make_image_with_spacing::<f32, B, 3>(data, shape, spacing)
}

fn voxels(img: &Image<f32, B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

// ── TileMeanShrinkFilter (tile-averaging downsample) ─────────────────────────

/// Factor [1,1,1] → identity (same shape and values).
#[test]
fn tile_mean_factor_one_is_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = make_image(data.clone(), [2, 3, 4], [1.0, 1.0, 1.0]);
    let out = TileMeanShrinkFilter::new([1, 1, 1]).apply(&img).unwrap();
    assert_eq!(out.shape(), [2, 3, 4]);
    let v = voxels(&out);
    for (a, b) in v.iter().zip(data.iter()) {
        assert!((a - b).abs() < 1e-5);
    }
}

/// 1-D tile-mean by 2: mean of pairs.
/// [0, 2, 4, 6] with factor [1,1,2] → [1, 5] (means of [0,2] and [4,6]).
#[test]
fn tile_mean_x_by_2() {
    let img = make_image(vec![0.0, 2.0, 4.0, 6.0], [1, 1, 4], [1.0, 1.0, 1.0]);
    let out = TileMeanShrinkFilter::new([1, 1, 2]).apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 2]);
    let v = voxels(&out);
    assert!((v[0] - 1.0).abs() < 1e-5, "v[0]={}", v[0]);
    assert!((v[1] - 5.0).abs() < 1e-5, "v[1]={}", v[1]);
}

/// Output spacing scales by factor.
#[test]
fn tile_mean_output_spacing_scales() {
    let img = make_image(vec![1.0f32; 8], [2, 2, 2], [1.0, 2.0, 3.0]);
    let out = TileMeanShrinkFilter::new([2, 2, 2]).apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 1]);
    let s = out.spacing();
    assert!((s[0] - 2.0).abs() < 1e-10, "sz={}", s[0]);
    assert!((s[1] - 4.0).abs() < 1e-10, "sy={}", s[1]);
    assert!((s[2] - 6.0).abs() < 1e-10, "sx={}", s[2]);
}

/// Odd input size with even factor: ceil division (trailing partial tile averaged).
/// 1×1×5 with factor [1,1,2] → shape [1,1,3]: means of [0,1], [2,3], [4].
#[test]
fn tile_mean_odd_size_ceil_division() {
    let img = make_image(vec![0.0, 1.0, 2.0, 3.0, 4.0], [1, 1, 5], [1.0, 1.0, 1.0]);
    let out = TileMeanShrinkFilter::new([1, 1, 2]).apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 3]);
    let v = voxels(&out);
    assert!((v[0] - 0.5).abs() < 1e-5, "v[0]={}", v[0]); // mean(0,1)
    assert!((v[1] - 2.5).abs() < 1e-5, "v[1]={}", v[1]); // mean(2,3)
    assert!((v[2] - 4.0).abs() < 1e-5, "v[2]={}", v[2]); // only 4
}

#[test]
fn native_tile_mean_preserves_origin_and_scales_spacing() {
    use coeus_core::SequentialBackend;
    use ritk_image::Image as NativeImage;
    use ritk_spatial::{Direction, Point};

    let image = NativeImage::from_flat_on(
        vec![0.0, 2.0, 4.0, 6.0],
        [1, 1, 4],
        Point::new([5.0, 7.0, 11.0]),
        Spacing::new([1.0, 2.0, 3.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = TileMeanShrinkFilter::new([1, 1, 2])
        .apply_native(&image, &SequentialBackend)
        .expect("native tile mean succeeds");

    assert_eq!(output.shape(), [1, 1, 2]);
    assert_eq!(output.data_slice().expect("contiguous output"), &[1.0, 5.0]);
    assert_eq!(
        [output.origin()[0], output.origin()[1], output.origin()[2]],
        [5.0, 7.0, 11.0]
    );
    assert_eq!(
        [
            output.spacing()[0],
            output.spacing()[1],
            output.spacing()[2]
        ],
        [1.0, 2.0, 6.0]
    );
}

// ── ShrinkImageFilter (ITK subsampling) ──────────────────────────────────────

/// Factor [1,1,1] → identity.
#[test]
fn subsample_factor_one_is_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = make_image(data.clone(), [2, 3, 4], [1.0, 1.0, 1.0]);
    let out = ShrinkImageFilter::new([1, 1, 1]).apply(&img).unwrap();
    assert_eq!(out.shape(), [2, 3, 4]);
    assert_eq!(voxels(&out), data);
}

/// ITK subsampling matches the hand-computed sitk.Shrink result:
/// 4×6 (z=1) by [fx=2, fy=2] → 2×3, sampling index o·f + f/2.
/// Input row-major value = y*6 + x. Kept (y,x) = (1,1),(1,3),(1,5),(3,1),(3,3),(3,5)
/// → [7, 9, 11, 19, 21, 23].
#[test]
fn subsample_matches_itk_offset() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = make_image(data, [1, 4, 6], [1.0, 1.0, 1.0]);
    let out = ShrinkImageFilter::new([1, 2, 2]).apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 2, 3]);
    assert_eq!(voxels(&out), vec![7.0, 9.0, 11.0, 19.0, 21.0, 23.0]);
}

/// Non-dividing axis: ITK centers the kept samples. N=12, f=5 → out=2,
/// offset = (12 mod 5 + 5)/2 = (2+5)/2 = 3 → samples at index 3 and 8.
#[test]
fn subsample_centers_samples_on_remainder() {
    let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let img = make_image(data, [1, 1, 12], [1.0, 1.0, 1.0]);
    let out = ShrinkImageFilter::new([1, 1, 5]).apply(&img).unwrap();
    assert_eq!(out.shape(), [1, 1, 2]);
    assert_eq!(voxels(&out), vec![3.0, 8.0]);
    // Origin shift = (12 mod 5 + 5 − 1)/2 = (2+4)/2 = 3.0 voxels.
    assert!(
        (out.origin()[2] - 3.0).abs() < 1e-10,
        "ox={}",
        out.origin()[2]
    );
}

/// Spacing scales by factor; origin shifts to the first tile centroid
/// (in_origin + spacing·(f−1)/2 under identity direction).
#[test]
fn subsample_spacing_and_origin() {
    let img = make_image(vec![3.0f32; 24], [1, 4, 6], [1.0, 1.0, 1.0]);
    let out = ShrinkImageFilter::new([1, 2, 2]).apply(&img).unwrap();
    let s = out.spacing();
    assert!((s[1] - 2.0).abs() < 1e-10 && (s[2] - 2.0).abs() < 1e-10);
    let o = out.origin();
    // z factor 1 → no shift; y,x factor 2 → +0.5.
    assert!((o[0] - 0.0).abs() < 1e-10, "oz={}", o[0]);
    assert!((o[1] - 0.5).abs() < 1e-10, "oy={}", o[1]);
    assert!((o[2] - 0.5).abs() < 1e-10, "ox={}", o[2]);
}
