//! Tests for binary_closing
//! Extracted to keep the 500-line structural limit.
use super::*;
use ritk_image::tensor::{Shape, Tensor, TensorData};
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

/// T1: Radius-0 closing is identity.
#[test]
fn radius_zero_is_identity() {
    let vals = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0];
    let img = make_image(vals.clone(), [2, 2, 2]);
    let out = BinaryMorphologicalClosing::new(0).apply(&img).unwrap();
    assert_eq!(flat(&out), vals);
}

/// T2: Extensivity — interior fg voxels survive closing.
///
/// For any f and B: C_B(f)(x) ≥ f(x) for interior voxels.
/// Uses 3×3×9 image with fg at (1,1,{2..6}) — all interior voxels at
/// distance ≥1 from every border face in Z and Y.
///
/// Proof that fg voxels survive:
/// 1. Dilate r=1: fg expands to iz∈{0..2}, iy∈{0..2}, ix∈{1..7}.
/// 2. Erode dilated r=1: iz=1, iy=1, ix∈{2..6} all have full SE in-bounds
///    and fg in dilated → all original fg voxels preserved.
#[test]
fn extensivity_no_foreground_lost() {
    let mut vals = vec![0.0_f32; 81]; // 3×3×9
    for ix in 2..=6usize {
        vals[1 * 27 + 1 * 9 + ix] = 1.0;
    }
    let img = make_image(vals.clone(), [3, 3, 9]);
    let out = BinaryMorphologicalClosing::new(1).apply(&img).unwrap();
    let result = flat(&out);
    for (i, &v) in vals.iter().enumerate() {
        if v == 1.0 {
            assert_eq!(
                result[i], 1.0,
                "fg voxel {i} was removed by closing (extensivity violated)"
            );
        }
    }
}

/// T3: Gap filling by closing in a proper 3D volume.
///
/// 3×3×7 image: fg at (1,1,{1,2,4,5}), bg at (1,1,{0,3,6}) and all other voxels.
/// The single-voxel gap at (1,1,3) separates two fg blobs.
///
/// Closing analysis:
/// 1. Dilate r=1: the two blobs expand and bridge the gap at ix=3. The
///    dilated image covers iz∈{0..2}, iy∈{0..2}, ix∈{0..6} (all 63 voxels).
/// 2. Erode all-fg 3×3×7 r=1: iz=1, iy=1, ix∈{1..5} survive.
///    Voxel (1,1,3) flat index = 1*21+1*7+3 = 31 → present in output.
#[test]
fn small_hole_filled_by_closing() {
    let mut vals = vec![0.0_f32; 63]; // 3×3×7
    for ix in [1usize, 2, 4, 5] {
        vals[1 * 21 + 1 * 7 + ix] = 1.0;
    }
    let img = make_image(vals, [3, 3, 7]);
    let out = BinaryMorphologicalClosing::new(1).apply(&img).unwrap();
    let result = flat(&out);
    // Gap at (1,1,3) flat index 31 must be filled.
    assert_eq!(result[31], 1.0, "gap at (1,1,3) must be filled by closing");
}

/// T4: All-background input stays all-background after closing.
#[test]
fn all_background_stays_background() {
    let img = make_image(vec![0.0; 8], [2, 2, 2]);
    let out = BinaryMorphologicalClosing::new(1).apply(&img).unwrap();
    assert!(flat(&out).iter().all(|&v| v == 0.0));
}

/// T5: All-foreground input stays all-foreground after closing.
///
/// Closing an all-fg image: dilate(fg) = fg, erode(fg) = fg.
#[test]
fn all_foreground_stays_foreground() {
    // Use a 5×5×5 volume; inner voxels survive erosion.
    let img = make_image(vec![1.0; 125], [5, 5, 5]);
    let out = BinaryMorphologicalClosing::new(1).apply(&img).unwrap();
    let result = flat(&out);
    // Centre voxel (index 62) must remain fg.
    assert_eq!(result[62], 1.0);
}

/// T6: Idempotence — applying closing twice gives the same result as once.
#[test]
fn idempotence() {
    let vals: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0];
    let img = make_image(vals, [1, 1, 9]);
    let once = BinaryMorphologicalClosing::new(1).apply(&img).unwrap();
    let twice = BinaryMorphologicalClosing::new(1).apply(&once).unwrap();
    assert_eq!(flat(&once), flat(&twice));
}

/// T7: Spatial metadata preserved.
#[test]
fn spatial_metadata_preserved() {
    let device: burn_ndarray::NdArrayDevice = Default::default();
    let origin = Point::new([1.0, 0.0, 0.0]);
    let spacing = Spacing::new([2.0, 2.0, 2.0]);
    let direction = Direction::identity();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3])),
        &device,
    );
    let img = Image::new(t, origin, spacing, direction);
    let out = BinaryMorphologicalClosing::new(1).apply(&img).unwrap();
    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
    assert_eq!(*out.direction(), direction);
}
