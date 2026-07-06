//! 2-D Zhang-Suen thinning tests.
#![allow(clippy::identity_op, clippy::erasing_op)]

use super::*;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_core::image::Image;
use ritk_core::spatial::{Direction, Point, Spacing};

// ── D = 2 tests ──────────────────────────────────────────────────────

#[test]
fn test_2d_empty_stays_empty() {
    let image = make_mask_2d(vec![0.0; 9], [3, 3]);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(count_fg_2d(&result), 0);
}

#[test]
fn test_2d_single_pixel_preserved() {
    let mut vals = vec![0.0_f32; 9];
    vals[4] = 1.0; // center of 3×3
    let image = make_mask_2d(vals, [3, 3]);
    let result = Skeletonization::new().apply(&image);
    let v = values_2d(&result);
    assert_eq!(v[4], 1.0);
    assert_eq!(count_fg_2d(&result), 1);
}

#[test]
fn test_2d_3x3_square_thins_to_center() {
    // Filled 3×3 square → Zhang-Suen thins to center pixel (1,1).
    let image = make_mask_2d(vec![1.0_f32; 9], [3, 3]);
    let result = Skeletonization::new().apply(&image);
    let v = values_2d(&result);
    assert_eq!(count_fg_2d(&result), 1, "3×3 square thins to 1 pixel");
    assert_eq!(v[4], 1.0, "remaining pixel is at center (1,1)");
}

#[test]
fn test_2d_horizontal_line_preserved() {
    // A 1-pixel-wide horizontal line is already a skeleton.
    // 3 rows × 7 cols, middle row is foreground.
    let (ny, nx) = (3, 7);
    let mut vals = vec![0.0_f32; ny * nx];
    for ix in 0..nx {
        vals[1 * nx + ix] = 1.0;
    }
    let image = make_mask_2d(vals.clone(), [ny, nx]);
    let result = Skeletonization::new().apply(&image);
    let v = values_2d(&result);
    // Endpoints (row 1, col 0) and (row 1, col 6) should remain.
    // Interior pixels: each has exactly 2 neighbors (left and right).
    // Zhang-Suen A = 1 check: for a horizontal line interior pixel:
    // P2=0, P3=0, P4=1, P5=0, P6=0, P7=0, P8=1, P9=0
    // B=2, A=2 (0→1 at P3→P4 and P7→P8). A≠1 → not deleted. Line preserved.
    for ix in 0..nx {
        assert_eq!(
            v[1 * nx + ix],
            1.0,
            "horizontal line pixel at x={ix} must be preserved"
        );
    }
}

#[test]
fn test_2d_skeleton_is_subset() {
    // 5×5 filled square.
    let image = make_mask_2d(vec![1.0_f32; 25], [5, 5]);
    let result = Skeletonization::new().apply(&image);
    let orig = [1.0_f32; 25];
    let skel = values_2d(&result);
    for (i, (&s, &o)) in skel.iter().zip(orig.iter()).enumerate() {
        if s > 0.5 {
            assert!(o > 0.5, "skeleton voxel {i} must be within original mask");
        }
    }
    assert!(
        count_fg_2d(&result) < 25,
        "skeleton must be strictly smaller than filled square"
    );
}

#[test]
fn test_2d_binary_output() {
    let image = make_mask_2d(vec![1.0_f32; 25], [5, 5]);
    let result = Skeletonization::new().apply(&image);
    for &v in values_2d(&result).iter() {
        assert!(v == 0.0 || v == 1.0, "output must be binary, got {v}");
    }
}

#[test]
fn test_2d_topology_preserved() {
    // Two separate 3×3 squares in a 3×9 image (with gap between).
    // Each should thin independently; component count preserved.
    let (ny, nx) = (3, 9);
    let mut vals = vec![0.0_f32; ny * nx];
    // Square 1: rows 0-2, cols 0-2
    for iy in 0..3 {
        for ix in 0..3 {
            vals[iy * nx + ix] = 1.0;
        }
    }
    // Square 2: rows 0-2, cols 6-8
    for iy in 0..3 {
        for ix in 6..9 {
            vals[iy * nx + ix] = 1.0;
        }
    }
    let cc_before = count_components_2d(&vals, ny, nx);
    assert_eq!(cc_before, 2, "two components before thinning");
    let image = make_mask_2d(vals, [ny, nx]);
    let result = Skeletonization::new().apply(&image);
    let skel = values_2d(&result);
    let cc_after = count_components_2d(&skel, ny, nx);
    assert_eq!(
        cc_after, cc_before,
        "connected component count must be preserved"
    );
}

#[test]
fn test_2d_no_2x2_block() {
    // The skeleton of a filled rectangle must be thin (no 2×2 blocks).
    let (ny, nx) = (7, 11);
    let image = make_mask_2d(vec![1.0_f32; ny * nx], [ny, nx]);
    let result = Skeletonization::new().apply(&image);
    let skel = values_2d(&result);
    assert!(
        !has_2x2_block(&skel, ny, nx),
        "skeleton must contain no 2×2 foreground block (thinness)"
    );
}

#[test]
fn test_2d_spatial_metadata_preserved() {
    let device: <TestBackend as ritk_image::tensor::Backend>::Device = Default::default();
    let td = TensorData::new(vec![1.0_f32; 9], Shape::new([3, 3]));
    let tensor = Tensor::<TestBackend, 2>::from_data(td, &device);
    let origin = Point::new([1.0, 2.0]);
    let spacing = Spacing::new([0.5, 1.5]);
    let direction = Direction::identity();
    let image = Image::new(tensor, origin, spacing, direction);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(result.origin(), &origin);
    assert_eq!(result.spacing(), &spacing);
    assert_eq!(result.direction(), &direction);
}
