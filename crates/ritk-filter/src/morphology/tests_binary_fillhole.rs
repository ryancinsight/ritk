//! Tests for binary_fillhole
//! Extracted to keep the 500-line structural limit.
use super::*;
use crate::native_support::LegacyBurnBackend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_image::test_support as ts;
use ritk_image::Image;
use ritk_spatial::{Direction, Point, Spacing};

type B = LegacyBurnBackend;

fn make_image(vals: Vec<f32>, dims: [usize; 3]) -> Image<B, 3> {
    ts::burn_compat::make_image::<B, 3>(vals, dims)
}

fn flat(img: &Image<B, 3>) -> Vec<f32> {
    img.data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// T1: All-foreground image stays all-foreground.
#[test]
fn all_foreground_unchanged() {
    let img = make_image(vec![1.0; 8], [2, 2, 2]);
    let out = BinaryFillholeFilter::new().apply(&img).unwrap();
    assert!(flat(&out).iter().all(|&v| v == 1.0));
}

/// T2: All-background image — all voxels are external bg (reachable from
///     border), so output is all background.
#[test]
fn all_background_stays_background() {
    let img = make_image(vec![0.0; 8], [2, 2, 2]);
    let out = BinaryFillholeFilter::new().apply(&img).unwrap();
    assert!(flat(&out).iter().all(|&v| v == 0.0));
}

/// T3: Enclosed hole is filled.
///
/// 3×3×3 volume: foreground shell (outer voxels) with a single background
/// voxel at the centre (index 13 in ZYX order).
///
/// The centre voxel at (1,1,1) is background and NOT reachable from any
/// border voxel (all 6 face-neighbors are foreground), so it must be filled.
///
/// Construction:
///   All 27 voxels = 1.0 except centre voxel index 13 = 0.0.
#[test]
fn enclosed_hole_filled() {
    let mut vals = vec![1.0_f32; 27];
    vals[13] = 0.0; // centre of 3×3×3 = (1,1,1), index = 1*9+1*3+1 = 13
    let img = make_image(vals, [3, 3, 3]);
    let out = BinaryFillholeFilter::new().apply(&img).unwrap();
    assert_eq!(
        flat(&out)[13],
        1.0,
        "enclosed hole at centre must be filled"
    );
}

/// T4: Interior background region in 5×5×5 volume is filled.
///
/// 5×5×5 volume with fg outer shell and bg interior (iz∈{1..3}, iy∈{1..3}, ix∈{1..3}).
/// The inner 3×3×3 = 27 voxels are bg and not reachable from any border face
/// (all 6 immediate Z/Y/X face-neighbours of the inner region are fg → no bg path).
/// After filling, all inner voxels must be fg.
#[test]
fn interior_bg_region_filled() {
    let mut vals = vec![1.0_f32; 125];
    for iz in 1..=3usize {
        for iy in 1..=3usize {
            for ix in 1..=3usize {
                vals[iz * 25 + iy * 5 + ix] = 0.0;
            }
        }
    }
    let img = make_image(vals, [5, 5, 5]);
    let out = BinaryFillholeFilter::new().apply(&img).unwrap();
    let result = flat(&out);
    for iz in 1..=3usize {
        for iy in 1..=3usize {
            for ix in 1..=3usize {
                let idx = iz * 25 + iy * 5 + ix;
                assert_eq!(
                    result[idx], 1.0,
                    "inner voxel ({iz},{iy},{ix}) must be filled"
                );
            }
        }
    }
}

/// T5: Extensivity — no foreground voxel is removed.
#[test]
fn extensivity_no_foreground_removed() {
    let vals: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];
    // 1×3×3 flat slice, centre bg at index 4.
    let img = make_image(vals.clone(), [1, 3, 3]);
    let out = BinaryFillholeFilter::new().apply(&img).unwrap();
    let result = flat(&out);
    for (i, &v) in vals.iter().enumerate() {
        if v == 1.0 {
            assert_eq!(result[i], 1.0, "fg voxel {i} was incorrectly removed");
        }
    }
}

/// T6: Custom foreground value fills enclosed bg with fg value.
///
/// 3×3×3 shell of fg=255 with interior bg at (1,1,1) — same geometry as T3
/// but with fg=255.  The centre voxel is enclosed and must be filled to 255.
#[test]
fn custom_foreground_value() {
    let mut vals = vec![255.0_f32; 27];
    vals[13] = 0.0; // centre (1,1,1) = bg
    let img = make_image(vals, [3, 3, 3]);
    let out = BinaryFillholeFilter::new()
        .with_foreground(255.0)
        .apply(&img)
        .unwrap();
    assert_eq!(
        flat(&out)[13],
        255.0,
        "enclosed bg centre must be filled to 255"
    );
}

/// T7: Spatial metadata preserved.
#[test]
fn spatial_metadata_preserved() {
    let device = Default::default();
    let origin = Point::new([5.0, 3.0, 1.0]);
    let spacing = Spacing::new([0.8, 0.8, 1.2]);
    let direction = Direction::identity();
    let t = Tensor::<B, 3>::from_data(
        TensorData::new(vec![1.0_f32; 8], Shape::new([2, 2, 2])),
        &device,
    );
    let img = Image::new(t, origin, spacing, direction);
    let out = BinaryFillholeFilter::new().apply(&img).unwrap();
    assert_eq!(*out.origin(), origin);
    assert_eq!(*out.spacing(), spacing);
    assert_eq!(*out.direction(), direction);
}
