//! Tests for skeletonization
//! Extracted from the main module to keep the 500-line structural limit.
use super::*;
use crate::image::Image;
use crate::spatial::{Direction, Point, Spacing};
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;

type TestBackend = NdArray<f32>;

// ── 1-D helpers ───────────────────────────────────────────────────────

fn make_mask_1d(values: Vec<f32>, nx: usize) -> Image<TestBackend, 1> {
    let device: <TestBackend as Backend>::Device = Default::default();
    let td = TensorData::new(values, Shape::new([nx]));
    let tensor = Tensor::<TestBackend, 1>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0]),
        Spacing::new([1.0]),
        Direction::identity(),
    )
}

fn values_1d(image: &Image<TestBackend, 1>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn count_fg_1d(image: &Image<TestBackend, 1>) -> usize {
    values_1d(image).iter().filter(|&&v| v > 0.5).count()
}

// ── 2-D helpers ───────────────────────────────────────────────────────

fn make_mask_2d(values: Vec<f32>, shape: [usize; 2]) -> Image<TestBackend, 2> {
    let device: <TestBackend as Backend>::Device = Default::default();
    let td = TensorData::new(values, Shape::new(shape));
    let tensor = Tensor::<TestBackend, 2>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 2]),
        Spacing::new([1.0; 2]),
        Direction::identity(),
    )
}

fn values_2d(image: &Image<TestBackend, 2>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn count_fg_2d(image: &Image<TestBackend, 2>) -> usize {
    values_2d(image).iter().filter(|&&v| v > 0.5).count()
}

// ── 3-D helpers ───────────────────────────────────────────────────────

fn make_mask_3d(values: Vec<f32>, shape: [usize; 3]) -> Image<TestBackend, 3> {
    let device: <TestBackend as Backend>::Device = Default::default();
    let td = TensorData::new(values, Shape::new(shape));
    let tensor = Tensor::<TestBackend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        Point::new([0.0; 3]),
        Spacing::new([1.0; 3]),
        Direction::identity(),
    )
}

fn values_3d(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

fn count_fg_3d(image: &Image<TestBackend, 3>) -> usize {
    values_3d(image).iter().filter(|&&v| v > 0.5).count()
}

// ── Connected component counting (for topology checks) ───────────────

/// Count 8-connected foreground components in a 2-D flat mask.
fn count_components_2d(flat: &[f32], ny: usize, nx: usize) -> usize {
    let n = ny * nx;
    let mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();
    let mut visited = vec![false; n];
    let mut components = 0usize;

    for start in 0..n {
        if !mask[start] || visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(idx) = stack.pop() {
            let iy = idx / nx;
            let ix = idx % nx;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    if dy == 0 && dx == 0 {
                        continue;
                    }
                    let ny_i = iy as isize + dy;
                    let nx_i = ix as isize + dx;
                    if ny_i < 0 || ny_i >= ny as isize || nx_i < 0 || nx_i >= nx as isize {
                        continue;
                    }
                    let ni = ny_i as usize * nx + nx_i as usize;
                    if mask[ni] && !visited[ni] {
                        visited[ni] = true;
                        stack.push(ni);
                    }
                }
            }
        }
    }
    components
}

/// Count 26-connected foreground components in a 3-D flat mask.
fn count_components_3d(flat: &[f32], nz: usize, ny: usize, nx: usize) -> usize {
    let n = nz * ny * nx;
    let mask: Vec<bool> = flat.iter().map(|&v| v > 0.5).collect();
    let mut visited = vec![false; n];
    let mut components = 0usize;

    for start in 0..n {
        if !mask[start] || visited[start] {
            continue;
        }
        components += 1;
        let mut stack = vec![start];
        visited[start] = true;
        while let Some(idx) = stack.pop() {
            let iz = idx / (ny * nx);
            let rem = idx % (ny * nx);
            let iy = rem / nx;
            let ix = rem % nx;
            for dz in -1isize..=1 {
                for dy in -1isize..=1 {
                    for dx in -1isize..=1 {
                        if dz == 0 && dy == 0 && dx == 0 {
                            continue;
                        }
                        let gz = iz as isize + dz;
                        let gy = iy as isize + dy;
                        let gx = ix as isize + dx;
                        if gz < 0
                            || gz >= nz as isize
                            || gy < 0
                            || gy >= ny as isize
                            || gx < 0
                            || gx >= nx as isize
                        {
                            continue;
                        }
                        let ni = gz as usize * ny * nx + gy as usize * nx + gx as usize;
                        if mask[ni] && !visited[ni] {
                            visited[ni] = true;
                            stack.push(ni);
                        }
                    }
                }
            }
        }
    }
    components
}

/// Check that no 2×2 foreground block exists (thinness in 2-D).
fn has_2x2_block(flat: &[f32], ny: usize, nx: usize) -> bool {
    for iy in 0..ny.saturating_sub(1) {
        for ix in 0..nx.saturating_sub(1) {
            if flat[iy * nx + ix] > 0.5
                && flat[iy * nx + ix + 1] > 0.5
                && flat[(iy + 1) * nx + ix] > 0.5
                && flat[(iy + 1) * nx + ix + 1] > 0.5
            {
                return true;
            }
        }
    }
    false
}

// ── D = 1 tests ──────────────────────────────────────────────────────

#[test]
fn test_1d_empty_stays_empty() {
    let image = make_mask_1d(vec![0.0; 5], 5);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(count_fg_1d(&result), 0);
}

#[test]
fn test_1d_single_voxel_preserved() {
    let mut vals = vec![0.0_f32; 7];
    vals[3] = 1.0;
    let image = make_mask_1d(vals, 7);
    let result = Skeletonization::new().apply(&image);
    let v = values_1d(&result);
    assert_eq!(v[3], 1.0);
    assert_eq!(count_fg_1d(&result), 1);
}

#[test]
fn test_1d_run_produces_midpoint() {
    // Run of 5: indices 1..=5 → midpoint = (1+5)/2 = 3.
    let mut vals = vec![0.0_f32; 8];
    for i in 1..=5 {
        vals[i] = 1.0;
    }
    let image = make_mask_1d(vals, 8);
    let result = Skeletonization::new().apply(&image);
    let v = values_1d(&result);
    assert_eq!(count_fg_1d(&result), 1, "single run → single midpoint");
    assert_eq!(v[3], 1.0, "midpoint at index 3");
}

#[test]
fn test_1d_two_runs_two_midpoints() {
    // Two runs: [0,1,1,0,0,1,1,1,0]
    let vals = vec![0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
    let image = make_mask_1d(vals, 9);
    let result = Skeletonization::new().apply(&image);
    let v = values_1d(&result);
    assert_eq!(count_fg_1d(&result), 2, "two runs → two midpoints");
    // Run 1: [1,2] → midpoint 1. Run 2: [5,6,7] → midpoint 6.
    assert_eq!(v[1], 1.0);
    assert_eq!(v[6], 1.0);
}

#[test]
fn test_1d_all_foreground() {
    // Entire image foreground: run [0, nx-1] → midpoint at nx/2.
    let nx = 9;
    let image = make_mask_1d(vec![1.0; nx], nx);
    let result = Skeletonization::new().apply(&image);
    assert_eq!(count_fg_1d(&result), 1);
    let v = values_1d(&result);
    assert_eq!(v[4], 1.0, "midpoint of [0,8] is 4");
}

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
    let orig = vec![1.0_f32; 25];
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
    let device: <TestBackend as Backend>::Device = Default::default();
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
    let device: <TestBackend as Backend>::Device = Default::default();
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
    use super::super::MorphologicalOperation;
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
