//! 3-D thinning tests and internal predicate / trait tests.

use super::*;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};

// ── D = 3 tests ──────────────────────────────────────────────────────

#[test]
fn test_3d_empty_stays_empty() {
    let image = make_mask_3d(vec![0.0; 27], [3, 3, 3]);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(count_fg_3d(&result), 0);
}

#[test]
fn test_3d_single_voxel_preserved() {
    let mut vals = vec![0.0_f32; 27];
    vals[13] = 1.0; // center of 3×3×3
    let image = make_mask_3d(vals, [3, 3, 3]);
    let result = Skeletonization::new().apply(&image);
    let v = values_3d(&result);
    assert_eq!(v[13], 1.0);
    assert_eq!(count_fg_3d(&result), 1);
}

#[test]
fn test_3d_straight_line_preserved() {
    // 1×1×7 line (all foreground): already 1-voxel wide.
    // Endpoints have 1 neighbor each; interior voxels have 2 neighbors
    // and are NOT simple (T₂₆ = 2 when neighbors are disconnected).
    // The line is preserved.
    let image = make_mask_3d(vec![1.0_f32; 7], [1, 1, 7]);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(
        count_fg_3d(&result),
        7,
        "1-voxel-wide line must be fully preserved"
    );
}

#[test]
fn test_3d_cube_thins_to_smaller() {
    // 5×5×5 filled cube → skeleton is strictly smaller.
    let n = 5 * 5 * 5;
    let image = make_mask_3d(vec![1.0_f32; n], [5, 5, 5]);
    let result = Skeletonization::new().apply(&image);
    let skel_count = count_fg_3d(&result);
    assert!(skel_count > 0, "skeleton must be non-empty");
    assert!(
        skel_count < n,
        "skeleton ({skel_count}) must be strictly smaller than cube ({n})"
    );
}

#[test]
fn test_3d_skeleton_is_subset() {
    let orig = vec![1.0_f32; 125]; // 5×5×5
    let image = make_mask_3d(orig.clone(), [5, 5, 5]);
    let result = Skeletonization::new().apply(&image);
    let skel = values_3d(&result);
    for (i, (&s, &o)) in skel.iter().zip(orig.iter()).enumerate() {
        if s > 0.5 {
            assert!(o > 0.5, "skeleton voxel {i} must be within original mask");
        }
    }
}

#[test]
fn test_3d_binary_output() {
    let image = make_mask_3d(vec![1.0_f32; 27], [3, 3, 3]);
    let result = Skeletonization::new().apply(&image);
    for &v in values_3d(&result).iter() {
        assert!(v == 0.0 || v == 1.0, "output must be binary, got {v}");
    }
}

#[test]
fn test_3d_topology_preserved() {
    // Two separate 3×3×3 cubes in a 3×3×9 image.
    let (nz, ny, nx) = (3, 3, 9);
    let mut vals = vec![0.0_f32; nz * ny * nx];
    // Cube 1: z=0..3, y=0..3, x=0..3
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 0..3 {
                vals[iz * ny * nx + iy * nx + ix] = 1.0;
            }
        }
    }
    // Cube 2: z=0..3, y=0..3, x=6..9
    for iz in 0..3 {
        for iy in 0..3 {
            for ix in 6..9 {
                vals[iz * ny * nx + iy * nx + ix] = 1.0;
            }
        }
    }
    let cc_before = count_components_3d(&vals, nz, ny, nx);
    assert_eq!(cc_before, 2);
    let image = make_mask_3d(vals, [nz, ny, nx]);
    let result = Skeletonization::new().apply(&image);
    let skel = values_3d(&result);
    let cc_after = count_components_3d(&skel, nz, ny, nx);
    assert_eq!(
        cc_after, cc_before,
        "3-D connected component count must be preserved"
    );
}

#[test]
fn test_3d_spatial_metadata_preserved() {
    let device: <TestBackend as burn::tensor::backend::Backend>::Device = Default::default();
    let td = TensorData::new(vec![1.0_f32; 27], Shape::new([3, 3, 3]));
    let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 1.0, 2.0]);
    let direction = Direction::identity();
    let image = Image::new(tensor, origin, spacing, direction);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(result.origin(), &origin);
    assert_eq!(result.spacing(), &spacing);
    assert_eq!(result.direction(), &direction);
}

// ── Internal predicate tests ─────────────────────────────────────────

#[test]
fn test_fg_components_26_single_component() {
    // All 26 neighbors are foreground → 1 component.
    let mut local = [true; 27];
    local[13] = false; // center excluded by convention in the function
    assert_eq!(fg_components_26(&local), 1);
}

#[test]
fn test_fg_components_26_two_components() {
    // Only two opposite corners: (0,0,0)=idx 0 and (2,2,2)=idx 26.
    // Chebyshev distance = 2 → not 26-adjacent → 2 components.
    let mut local = [false; 27];
    local[0] = true; // (0,0,0)
    local[26] = true; // (2,2,2)
    assert_eq!(fg_components_26(&local), 2);
}

#[test]
fn test_fg_components_26_empty() {
    let local = [false; 27];
    assert_eq!(fg_components_26(&local), 0);
}

#[test]
fn test_fg_components_26_adjacent_corners() {
    // (0,0,0) and (1,1,1)=center is excluded, (0,0,1) and (0,1,0).
    // (0,0,0)=0, (0,0,1)=1, (0,1,0)=3.
    // (0,0,0) is 26-adjacent to (0,0,1) (differ by 1 in x) → connected.
    // (0,0,0) is 26-adjacent to (0,1,0) (differ by 1 in y) → connected.
    // All form 1 component.
    let mut local = [false; 27];
    local[0] = true; // (0,0,0)
    local[1] = true; // (0,0,1)
    local[3] = true; // (0,1,0)
    assert_eq!(fg_components_26(&local), 1);
}

// ── Trait / API parity test ──────────────────────────────────────────

#[test]
fn test_morphological_operation_trait() {
    use crate::segmentation::morphology::MorphologicalOperation;
    let image = make_mask_3d(vec![1.0_f32; 27], [3, 3, 3]);
    let via_method = Skeletonization::new().apply(&image);
    let via_trait: Image<TestBackend, 3> = <Skeletonization as MorphologicalOperation<
        TestBackend,
        3,
    >>::apply(&Skeletonization::new(), &image);
    assert_eq!(values_3d(&via_method), values_3d(&via_trait));
}

#[test]
fn test_default_impl() {
    let s = Skeletonization::default();
    let image = make_mask_2d(vec![1.0_f32; 9], [3, 3]);
    let result = s.apply(&image);
    assert_eq!(count_fg_2d(&result), 1);
}
