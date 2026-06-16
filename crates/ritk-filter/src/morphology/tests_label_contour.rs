//! Tests for label_contour
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn::tensor::TensorData;
use burn_ndarray::NdArray;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(data: Vec<f32>, shape: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let td = TensorData::new(data, Shape::new(shape));
    let tensor = Tensor::<B, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0_f64, 0.0, 0.0]),
        Spacing::new([1.0_f64, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn voxels(img: &Image<B, 3>) -> Vec<f32> {
    img.data_slice().into_owned()
}

/// All-background image → all background in output.
#[test]
fn all_background_zero() {
    let img = make_image(vec![0.0f32; 27], [3, 3, 3]);
    let out = LabelContourImageFilter::default().apply(&img).unwrap();
    assert!(voxels(&out).iter().all(|&v| v == 0.0));
}

/// Single label region filling entire image: outer shell is border; center is interior.
#[test]
fn single_label_all_border() {
    let img = make_image(vec![2.0f32; 27], [3, 3, 3]);
    let out = LabelContourImageFilter::default().apply(&img).unwrap();
    let v = voxels(&out);
    // Center (1,1,1) = index 13: all 6 neighbors in-bounds and same label → interior → bg.
    assert_eq!(v[13], 0.0, "center of 3×3×3 single-label is interior");
    // All other 26 voxels border out-of-bounds (treated as different label) → contour.
    assert!(
        v.iter()
            .enumerate()
            .filter(|&(i, _)| i != 13)
            .all(|(_, &x)| (x - 2.0).abs() < 1e-5),
        "outer shell of 3×3×3 single-label must be contour"
    );
}

/// Two labels side-by-side (left half = label 1, right half = label 2).
/// Interface voxels should be marked; interior voxels (no different-label neighbour) → 0.
#[test]
fn two_labels_interface_marked() {
    // 1×1×6: [1,1,1,2,2,2]
    let img = make_image(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0], [1, 1, 6]);
    let out = LabelContourImageFilter::default().apply(&img).unwrap();
    let v = voxels(&out);
    // voxel 2 (label 1, right edge of label-1): neighbour voxel 3 = label 2 → contour
    assert!(
        (v[2] - 1.0).abs() < 1e-5,
        "v[2] should be 1 (contour), got {}",
        v[2]
    );
    // voxel 3 (label 2, left edge of label-2): neighbour voxel 2 = label 1 → contour
    assert!(
        (v[3] - 2.0).abs() < 1e-5,
        "v[3] should be 2 (contour), got {}",
        v[3]
    );
    // voxels 0 and 1 are at left image boundary → also border
    // voxels 4 and 5 are at right image boundary → also border
    // voxel 1 borders voxel 0 (same label) and voxel 2 (same label) but image left boundary treated as bg → border
}

/// Labels are preserved in contour voxels (label value, not a binary mask).
#[test]
fn contour_voxels_preserve_label_value() {
    // 1×1×4: [0, 5, 5, 0]
    let img = make_image(vec![0.0, 5.0, 5.0, 0.0], [1, 1, 4]);
    let out = LabelContourImageFilter::default().apply(&img).unwrap();
    let v = voxels(&out);
    // voxel 1 (label 5): neighbour 0 is bg → contour → value = 5
    assert!((v[1] - 5.0).abs() < 1e-5, "v[1]={}", v[1]);
    // voxel 2 (label 5): neighbour 3 is bg → contour → value = 5
    assert!((v[2] - 5.0).abs() < 1e-5, "v[2]={}", v[2]);
}

/// Spatial metadata preserved.
#[test]
fn preserves_metadata() {
    let img = make_image(vec![0.0f32; 8], [2, 2, 2]);
    let out = LabelContourImageFilter::default().apply(&img).unwrap();
    assert_eq!(out.shape(), [2, 2, 2]);
}
