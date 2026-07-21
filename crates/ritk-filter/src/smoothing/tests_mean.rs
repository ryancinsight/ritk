//! Tests for mean
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;
use ritk_spatial::{Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, shape)
}

fn voxels(img: &Image<f32, B, 3>) -> Vec<f32> {
    let (v, _) = extract_vec_infallible(img);
    v
}

/// Constant image → mean = constant for any radius.
#[test]
fn constant_image_identity() {
    let img = make_image(vec![7.0f32; 27], [3, 3, 3]);
    let out = MeanImageFilter::new(1)
        .apply(&img)
        .expect("infallible: validated precondition");
    let v = voxels(&out);
    for &x in &v {
        assert!((x - 7.0).abs() < 1e-5, "expected 7.0 got {x}");
    }
}

/// radius=0 → exact identity.
#[test]
fn radius_zero_is_identity() {
    let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
    let img = make_image(data.clone(), [3, 3, 3]);
    let out = MeanImageFilter::new(0)
        .apply(&img)
        .expect("infallible: validated precondition");
    let v = voxels(&out);
    for (a, b) in v.iter().zip(data.iter()) {
        assert!((a - b).abs() < 1e-6, "{a} != {b}");
    }
}

/// Single-voxel image: radius > 0 → returns that voxel unchanged.
#[test]
fn single_voxel_returns_itself() {
    let img = make_image(vec![42.0], [1, 1, 1]);
    let out = MeanImageFilter::new(2)
        .apply(&img)
        .expect("infallible: validated precondition");
    assert!((voxels(&out)[0] - 42.0).abs() < 1e-5);
}

/// Step-edge smoothing: 3×3×3 volume, left half = 0, right half = 10.
/// Center voxel mean over a 3×3×3 (no boundary cut) should be 5.0.
#[test]
fn step_edge_center_mean() {
    // 1×1×4 image: [0, 0, 10, 10]. With r=1, voxel at index 1:
    // neighbourhood = [0,0,10] → mean = 10/3 ≈ 3.333
    let img = make_image(vec![0.0, 0.0, 10.0, 10.0], [1, 1, 4]);
    let out = MeanImageFilter::new(1)
        .apply(&img)
        .expect("infallible: validated precondition");
    let v = voxels(&out);
    // voxel 1 (0-indexed): neighbourhood [0,0,10] → 10/3
    assert!((v[1] - 10.0 / 3.0).abs() < 1e-4, "v[1]={}", v[1]);
    // voxel 2: neighbourhood [0,10,10] → 20/3
    assert!((v[2] - 20.0 / 3.0).abs() < 1e-4, "v[2]={}", v[2]);
}

/// Spatial metadata is preserved.
#[test]
fn preserves_spatial_metadata() {
    use ritk_spatial::Direction;
    let tensor = Tensor::<f32, B>::from_slice([2, 2, 2], &[1.0f32; 8]);
    let origin = Point::new([3.0_f64, 5.0, 7.0]);
    let spacing = Spacing::new([2.0_f64, 3.0, 4.0]);
    let dir = Direction::identity();
    let img = Image::new(tensor, origin, spacing, dir)
        .expect("invariant: fixture tensor has the declared rank");
    let out = MeanImageFilter::new(1)
        .apply(&img)
        .expect("infallible: validated precondition");
    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
}

/// Shape is unchanged after filtering.
#[test]
fn output_shape_matches_input() {
    let img = make_image(vec![1.0f32; 60], [3, 4, 5]);
    let out = MeanImageFilter::new(2)
        .apply(&img)
        .expect("infallible: validated precondition");
    assert_eq!(out.shape(), [3, 4, 5]);
}

#[test]
fn native_mean_uses_zero_flux_boundary() {
    use coeus_core::SequentialBackend;
    use ritk_image::Image as NativeImage;
    use ritk_spatial::Direction;

    let image = NativeImage::from_flat_on(
        vec![0.0, 0.0, 10.0, 10.0],
        [1, 1, 4],
        Point::new([2.0, 3.0, 5.0]),
        Spacing::new([1.0, 2.0, 4.0]),
        Direction::identity(),
        &SequentialBackend,
    )
    .expect("invariant: valid native image");
    let output = MeanImageFilter::new(1)
        .apply_native(&image, &SequentialBackend)
        .expect("native mean succeeds");

    assert_eq!(output.shape(), [1, 1, 4]);
    let values = output.data_slice().expect("contiguous output");
    assert!((values[1] - 10.0 / 3.0).abs() <= 1e-6);
    assert!((values[2] - 20.0 / 3.0).abs() <= 1e-6);
    assert_eq!(
        [output.origin()[0], output.origin()[1], output.origin()[2]],
        [2.0, 3.0, 5.0]
    );
}
