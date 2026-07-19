use super::*;
use ritk_core::image::Image;
use ritk_image::tensor::Tensor;
use ritk_image::test_support as ts;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_tensor_ops::extract_vec_infallible;

type B = coeus_core::SequentialBackend;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<f32, B, 3> {
    ts::make_image::<f32, B, 3>(data, shape)
}

// ── DownsampleFilter ───────────────────────────────────────────────────────

/// Factor 1 in every dimension: shape, spacing, and voxel values are unchanged.
#[test]
fn downsample_factor_one_is_identity() {
    let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let img = make_image(data.clone(), [2, 3, 4]);
    let out = DownsampleFilter::<B>::new(vec![1, 1, 1]).apply(&img);
    assert_eq!(
        out.shape(),
        img.shape(),
        "shape must be unchanged for factor=1"
    );
    let (got, _) = extract_vec_infallible(&out);
    assert_eq!(got, data, "voxels must be identical for factor=1");
}

/// Factor 2 in every dimension halves the shape (floor division) and doubles the spacing.
///
/// # Derivation
/// Input shape [4, 6, 8], factor=2: indices 0,2,4,6 along each axis.
/// New shape = [ceil(4/2), ceil(6/2), ceil(8/2)] BUT the implementation uses
/// step_by(factor) which gives floor: [4/2=2, 6/2=3, 8/2=4].
#[test]
fn downsample_factor_two_halves_shape_and_doubles_spacing() {
    let n = 4 * 6 * 8;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = Image::<f32, B, 3>::new(
        Tensor::<f32, B>::from_slice([4usize, 6, 8], &data),
        Point::new([0.0; 3]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");
    let out = DownsampleFilter::<B>::new(vec![2, 2, 2]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "dim0: 4 / step_by(2) = 2 elements");
    assert_eq!(s[1], 3, "dim1: 6 / step_by(2) = 3 elements");
    assert_eq!(s[2], 4, "dim2: 8 / step_by(2) = 4 elements");
    // Spacing must double
    assert!(
        (out.spacing()[0] - 2.0).abs() < 1e-9,
        "spacing[0] must double to 2.0"
    );
}

/// Factor vector shorter than D: last factor is broadcast (factors[0]).
///
/// With factors=[2] on a 3D image, all 3 dims use factor=2.
#[test]
fn downsample_scalar_factor_broadcast() {
    let n = 4 * 4 * 4;
    let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let img = make_image(data, [4, 4, 4]);
    let out = DownsampleFilter::<B>::new(vec![2]).apply(&img);
    let s = out.shape();
    assert_eq!(s[0], 2, "broadcast factor must apply to dim 0");
    assert_eq!(s[1], 2, "broadcast factor must apply to dim 1");
    assert_eq!(s[2], 2, "broadcast factor must apply to dim 2");
}

/// Spacing is updated proportionally to the downsampling factor.
#[test]
fn downsample_spacing_updated_proportionally() {
    let n = 6 * 6 * 6;
    let img = Image::<f32, B, 3>::new(
        Tensor::<f32, B>::from_slice([6usize, 6, 6], &vec![0.0_f32; n]),
        Point::new([0.0; 3]),
        Spacing::new([0.5, 1.0, 2.0]),
        Direction::identity(),
    )
    .expect("invariant: fixture tensor has the declared rank");
    let out = DownsampleFilter::<B>::new(vec![3, 2, 1]).apply(&img);
    assert!(
        (out.spacing()[0] - 1.5).abs() < 1e-9,
        "spacing[0]: 0.5*3=1.5"
    );
    assert!(
        (out.spacing()[1] - 2.0).abs() < 1e-9,
        "spacing[1]: 1.0*2=2.0"
    );
    assert!(
        (out.spacing()[2] - 2.0).abs() < 1e-9,
        "spacing[2]: 2.0*1=2.0"
    );
}
