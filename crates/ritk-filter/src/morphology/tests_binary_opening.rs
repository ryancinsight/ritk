//! Tests for binary_opening
//! Extracted to keep the 500-line structural limit.
use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = NdArray<f32>;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    let device = Default::default();
    let t = Tensor::<B, 3>::from_data(TensorData::new(vals, Shape::new(dims)), &device);
    Image::new(
        t,
        Point::new([0.0, 0.0, 0.0]),
        Spacing::new([1.0, 1.0, 1.0]),
        Direction::identity(),
    )
}

fn flat(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// T1: Radius-0 opening is identity.
#[test]
fn radius_zero_is_identity() {
    let vals = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let img = make_image(vals.clone(), [2, 2, 2]);
    let out = BinaryMorphologicalOpening::new(0).apply(&img).unwrap();
    assert_eq!(flat(&out), vals);
}

/// T2: Anti-extensivity — opening does not add foreground voxels.
///
/// For any f and B: O_B(f)(x) ≤ f(x).
/// Equivalently: no background voxel in input becomes foreground in output.
#[test]
fn anti_extensivity_no_new_foreground() {
    let vals: Vec<f32> = vec![0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    let img = make_image(vals.clone(), [1, 1, 9]);
    let out = BinaryMorphologicalOpening::new(1).apply(&img).unwrap();
    let result = flat(&out);
    for (i, &v) in vals.iter().enumerate() {
        if v == 0.0 {
            assert_eq!(result[i], 0.0, "bg voxel {i} became fg after opening");
        }
    }
}

/// T3: Small isolated foreground blob is removed by opening.
///
/// 1×1×7 image: [0, 0, 1, 0, 0, 0, 0] — isolated single fg voxel at index 2.
/// r=1: erode → all background (single voxel cannot survive r=1 erosion).
/// dilate(background) → all background.
#[test]
fn small_spike_removed_by_opening() {
    let img = make_image(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1, 1, 7]);
    let out = BinaryMorphologicalOpening::new(1).apply(&img).unwrap();
    assert!(flat(&out).iter().all(|&v| v == 0.0));
}

/// T4: Large 3D fg block (wider than 2r+1 in all dimensions) survives opening.
///
/// 3×3×9 image: fg at iz∈{0..2}, iy∈{0..2}, ix∈{2..6} — a 3×3×5 block.
///
/// Opening analysis (r=1):
/// 1. Erode: voxels needing ALL SE positions fg.
///    - iz: need iz±1 ∈ [0,2] → iz=1 only.
///    - iy: need iy±1 ∈ [0,2] → iy=1 only.
///    - ix: need ix±1 ∈ [2,6] → ix ∈ {3,4,5}.
///      Surviving erode: (1,1,{3,4,5}).
/// 2. Dilate {(1,1,3),(1,1,4),(1,1,5)} by r=1: expands back to the original
///    3×3×5 block (iz∈{0..2}, iy∈{0..2}, ix∈{2..6}).
///
/// Centre voxels at (1,1,{3,4,5}) survive; flat indices 39, 40, 41 (3×3×9 grid).
#[test]
fn large_region_survives_opening() {
    let mut vals = vec![0.0_f32; 81]; // 3×3×9
    for iz in 0..3usize {
        for iy in 0..3usize {
            for ix in 2..=6usize {
                vals[iz * 27 + iy * 9 + ix] = 1.0;
            }
        }
    }
    let img = make_image(vals, [3, 3, 9]);
    let out = BinaryMorphologicalOpening::new(1).apply(&img).unwrap();
    let result = flat(&out);
    // Centre voxels (1,1,{3,4,5}) must survive.
    assert_eq!(
        result[1 * 27 + 1 * 9 + 3],
        1.0,
        "(1,1,3) must survive opening"
    );
    assert_eq!(
        result[1 * 27 + 1 * 9 + 4],
        1.0,
        "(1,1,4) must survive opening"
    );
    assert_eq!(
        result[1 * 27 + 1 * 9 + 5],
        1.0,
        "(1,1,5) must survive opening"
    );
}

/// T5: All-background stays background after opening.
#[test]
fn all_background_stays_background() {
    let img = make_image(vec![0.0; 8], [2, 2, 2]);
    let out = BinaryMorphologicalOpening::new(1).apply(&img).unwrap();
    assert!(flat(&out).iter().all(|&v| v == 0.0));
}

/// T6: Idempotence — applying opening twice gives the same result as once.
#[test]
fn idempotence() {
    let vals: Vec<f32> = vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    let img = make_image(vals, [1, 1, 9]);
    let once = BinaryMorphologicalOpening::new(1).apply(&img).unwrap();
    let twice = BinaryMorphologicalOpening::new(1).apply(&once).unwrap();
    assert_eq!(flat(&once), flat(&twice));
}

/// T7: Spatial metadata preserved.
#[test]
fn spatial_metadata_preserved() {
    let device: burn_ndarray::NdArrayDevice = Default::default();
    let origin = Point::new([0.0, 1.0, 2.0]);
    let spacing = Spacing::new([1.5, 1.5, 1.5]);
    let direction = Direction::identity();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3])),
        &device,
    );
    let img = Image::new(t, origin, spacing, direction);
    let out = BinaryMorphologicalOpening::new(0).apply(&img).unwrap();
    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
    assert_eq!(*out.direction(), direction);
}
