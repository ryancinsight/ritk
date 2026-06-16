//! Tests for binary_erode
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::make_image::<B, 3>(vals, dims)
}

fn flat(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// T1: Radius-0 erosion is identity (single-voxel SE).
#[test]
fn radius_zero_is_identity() {
    let vals = vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    let img = make_image(vals.clone(), [2, 2, 2]);
    let out = BinaryErodeFilter::new(0).apply(&img).unwrap();
    assert_eq!(flat(&out), vals);
}

/// T2: All-foreground 3×3×3 image with r=1 → only centre voxel (1,1,1) survives.
///
/// For r=1, a voxel survives erosion iff ALL 27 SE positions are in-bounds and fg.
/// Only (1,1,1) has all neighbours within [0,2]³ (nz=ny=nx=3).
/// All border/edge/corner voxels have at least one OOB SE position → eroded to bg.
///
/// Flat index of (1,1,1) in 3×3×3: 1*9 + 1*3 + 1 = 13.
#[test]
fn border_voxels_eroded_to_background() {
    let img = make_image(vec![1.0; 27], [3, 3, 3]);
    let out = BinaryErodeFilter::new(1).apply(&img).unwrap();
    let result = flat(&out);
    // Only centre voxel (flat index 13) survives.
    assert_eq!(result[13], 1.0, "centre voxel must survive erosion");
    for (i, &v) in result.iter().enumerate() {
        if i != 13 {
            assert_eq!(v, 0.0, "border/edge voxel {i} must be eroded");
        }
    }
}

/// T3: Background pixel surrounded by foreground is NOT changed to foreground
///     (erosion only removes; it cannot add foreground).
#[test]
fn background_remains_background() {
    // Image: [fg, bg, fg] — bg is isolated, not eroded from fg
    let img = make_image(vec![1.0, 0.0, 1.0], [1, 1, 3]);
    let out = BinaryErodeFilter::new(1).apply(&img).unwrap();
    // Left fg eroded (out-of-bounds left), centre bg stays 0, right fg eroded.
    assert_eq!(flat(&out), vec![0.0, 0.0, 0.0]);
}

/// T4: 3×3×5 all-foreground, r=1 → strips one border layer from all 6 faces.
///
/// Surviving voxels: iz=1, iy=1, ix ∈ {1,2,3}.
/// Flat indices (nz=3, ny=3, nx=5): iz*15 + iy*5 + ix = 21, 22, 23.
#[test]
fn erosion_strips_one_border_layer_r1() {
    let img = make_image(vec![1.0; 45], [3, 3, 5]);
    let out = BinaryErodeFilter::new(1).apply(&img).unwrap();
    let result = flat(&out);
    let mut expected = vec![0.0_f32; 45];
    expected[21] = 1.0; // (1,1,1)
    expected[22] = 1.0; // (1,1,2)
    expected[23] = 1.0; // (1,1,3)
    assert_eq!(result, expected);
}

/// T5: 5×5×7 all-foreground, r=2 → strips two border layers from all faces.
///
/// Surviving voxels: iz=2, iy=2, ix ∈ {2,3,4}.
/// Flat indices (nz=5, ny=5, nx=7): iz*35 + iy*7 + ix = 86, 87, 88.
#[test]
fn erosion_strips_two_border_layers_r2() {
    let img = make_image(vec![1.0; 175], [5, 5, 7]);
    let out = BinaryErodeFilter::new(2).apply(&img).unwrap();
    let result = flat(&out);
    let mut expected = vec![0.0_f32; 175];
    expected[86] = 1.0; // (2,2,2)
    expected[87] = 1.0; // (2,2,3)
    expected[88] = 1.0; // (2,2,4)
    assert_eq!(result, expected);
}

/// T6: Custom foreground value 255.0 — 3×3×5 volume, same geometry as T4.
///
/// Flat indices 21, 22, 23 survive with value 255.0.
#[test]
fn custom_foreground_value() {
    let img = make_image(vec![255.0; 45], [3, 3, 5]);
    let out = BinaryErodeFilter::new(1)
        .with_foreground(255.0)
        .apply(&img)
        .unwrap();
    let result = flat(&out);
    let mut expected = vec![0.0_f32; 45];
    expected[21] = 255.0;
    expected[22] = 255.0;
    expected[23] = 255.0;
    assert_eq!(result, expected);
}

/// T7: Spatial metadata is preserved unchanged.
#[test]
fn spatial_metadata_preserved() {
    let device: burn_ndarray::NdArrayDevice = Default::default();
    let origin = Point::new([3.0, 2.0, 1.0]);
    let spacing = Spacing::new([0.5, 0.5, 1.0]);
    let direction = Direction::identity();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2])),
        &device,
    );
    let img = Image::new(t, origin, spacing, direction);
    let out = BinaryErodeFilter::new(0).apply(&img).unwrap();
    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
    assert_eq!(*out.direction(), direction);
}
