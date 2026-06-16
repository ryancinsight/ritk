//! Tests for distance_transform
//! Extracted to keep the 500-line structural limit.
#![allow(clippy::needless_range_loop)]
use super::*;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use ritk_image::test_support::make_image;
use ritk_core::spatial::{Direction, Point, Spacing};

type TestBackend = NdArray<f32>;

fn make_image_3d(data: Vec<f32>, dims: [usize; 3]) -> Image<TestBackend, 3> {
    make_image(data, dims)
}

fn get_values(image: &Image<TestBackend, 3>) -> Vec<f32> {
    image
        .data()
        .clone()
        .into_data()
        .as_slice::<f32>()
        .unwrap()
        .to_vec()
}

/// Helper: index into flat 3D array.
fn at(vals: &[f32], z: usize, y: usize, x: usize, ny: usize, nx: usize) -> f32 {
    vals[z * ny * nx + y * nx + x]
}

// ── Test 1: Single foreground voxel in center of 5×5×5 ────────────────

#[test]
fn test_single_foreground_voxel_center_5x5x5() {
    // 5×5×5 image, all background (0.0) except center voxel (2,2,2) = 1.0 (foreground).
    // New convention: distance from each voxel to nearest foreground voxel.
    // Foreground voxel (2,2,2) is the only seed → EDT²(2,2,2) = 0.
    // Background voxel at (z,y,x): EDT²(z,y,x) = (z-2)²+(y-2)²+(x-2)².
    let dims = [5, 5, 5];
    let total = 125;
    let mut data = vec![0.0f32; total];
    data[idx3(2, 2, 2, 5, 5)] = 1.0; // foreground

    let image = make_image_3d(data, dims);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);

    // Center foreground voxel is the seed: EDT² = 0.
    assert_eq!(
        at(&vals, 2, 2, 2, 5, 5),
        0.0,
        "center foreground voxel EDT² must be 0 (it is the seed)"
    );

    // Every background voxel (z,y,x) ≠ (2,2,2) gets EDT² = (z-2)²+(y-2)²+(x-2)².
    for z in 0..5usize {
        for y in 0..5usize {
            for x in 0..5usize {
                if (z, y, x) != (2, 2, 2) {
                    let expected = ((z as i32 - 2).pow(2)
                        + (y as i32 - 2).pow(2)
                        + (x as i32 - 2).pow(2)) as f32;
                    let actual = at(&vals, z, y, x, 5, 5);
                    assert_eq!(
                        actual, expected,
                        "background voxel ({z},{y},{x}) must have EDT²={expected}, got {actual}"
                    );
                }
            }
        }
    }

    // Verify the non-squared transform: center is seed → EDT = 0.
    let edt = distance_transform(&image, 0.5);
    let edt_vals = get_values(&edt);
    let center_dist = at(&edt_vals, 2, 2, 2, 5, 5);
    assert!(
        (center_dist - 0.0).abs() < 1e-6,
        "center EDT must be 0.0 (seed), got {center_dist}"
    );
    // Adjacent voxel (2,2,3) is background at distance 1 from seed.
    let adj_dist = at(&edt_vals, 2, 2, 3, 5, 5);
    assert!(
        (adj_dist - 1.0).abs() < 1e-6,
        "adjacent voxel (2,2,3) EDT must be 1.0, got {adj_dist}"
    );
}

// ── Test 2: All-foreground image → sentinel distance ──────────────────

#[test]
fn test_all_foreground_image() {
    // 3×3×3, all voxels = 1.0 (foreground). All voxels are seeds.
    // New convention: EDT² = 0 for every voxel (each voxel is its own seed).
    let dims = [3, 3, 3];
    let data = vec![1.0f32; 27];
    let image = make_image_3d(data, dims);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);

    let expected_sq = 0.0f32;

    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(
            v, expected_sq,
            "all-foreground voxel {i} must have EDT²=0 (all are seeds), got {v}"
        );
    }
}

// ── Test 3: All-background image → all distances 0 ────────────────────

#[test]
fn test_all_background_image() {
    // 4×3×5, all background. No foreground seeds.
    // Convention (Sprint 81): EDT(p) = min over empty set → defined as 0.0.
    // Rationale: no seeds exist; returning 0 is the safe sentinel for this degenerate case.
    let dims = [4, 3, 5];
    let data = vec![0.0f32; 60];
    let image = make_image_3d(data, dims);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);

    for (i, &v) in vals.iter().enumerate() {
        assert_eq!(
            v, 0.0,
            "all-background EDT²=0 (empty foreground, no seeds), voxel {i} got {v}"
        );
    }
}

// ── Test 4: Single background voxel at corner → analytical distances ──

#[test]
fn test_single_background_voxel_at_corner() {
    // 5×5×5, all foreground (1.0) except corner (0,0,0) = 0.0 (background).
    // New convention: 124 foreground voxels are seeds → EDT²=0 for all fg.
    // Background (0,0,0): nearest foreground is (1,0,0)/(0,1,0)/(0,0,1) at distance 1 → EDT²=1.
    let dims = [5, 5, 5];
    let total = 125;
    let mut data = vec![1.0f32; total];
    data[idx3(0, 0, 0, 5, 5)] = 0.0; // background

    let image = make_image_3d(data, dims);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);

    // All foreground voxels are seeds: EDT²=0.
    for z in 0..5usize {
        for y in 0..5usize {
            for x in 0..5usize {
                if (z, y, x) != (0, 0, 0) {
                    let actual = at(&vals, z, y, x, 5, 5);
                    assert_eq!(
                        actual, 0.0,
                        "foreground voxel ({z},{y},{x}) must have EDT²=0 (seed), got {actual}"
                    );
                }
            }
        }
    }
    // Background corner (0,0,0): nearest foreground at distance 1.
    let corner_sq = at(&vals, 0, 0, 0, 5, 5);
    assert_eq!(corner_sq, 1.0, "background corner EDT²=1, got {corner_sq}");

    // EDT(0,0,0) = √1 = 1.0.
    let edt = distance_transform(&image, 0.5);
    let edt_vals = get_values(&edt);
    let corner = at(&edt_vals, 0, 0, 0, 5, 5);
    assert!(
        (corner - 1.0).abs() < 1e-6,
        "EDT(0,0,0) = {corner}, expected 1.0"
    );
}

// ── Test 5: 2D-equivalent test (nz=1 plane) with known geometry ───────

#[test]
fn test_2d_plane_known_geometry() {
    // A 1×5×5 "plane" (flat in Z). Background is a vertical stripe at x=0.
    // All voxels at x=0 are background (0.0), rest are foreground (1.0).
    //
    // New convention: foreground voxels are seeds.
    // For background voxel at (0, y, 0): nearest foreground is at (0, y, 1) → EDT²=1.
    // For foreground voxel at (0, y, x) with x > 0: seed → EDT²=0.
    let ny = 5;
    let nx = 5;
    let dims = [1, ny, nx];
    let mut data = vec![1.0f32; ny * nx];
    for y in 0..ny {
        data[y * nx] = 0.0; // x=0 column is background
    }

    let image = make_image_3d(data, dims);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);

    for y in 0..ny {
        // x=0: background, nearest foreground at x=1 → EDT²=1
        let bg = at(&vals, 0, y, 0, ny, nx);
        assert_eq!(bg, 1.0, "EDT²(0,{y},0) background must be 1, got {bg}");
        // x>0: foreground seed → EDT²=0
        for x in 1..nx {
            let fg = at(&vals, 0, y, x, ny, nx);
            assert_eq!(
                fg, 0.0,
                "EDT²(0,{y},{x}) foreground seed must be 0, got {fg}"
            );
        }
    }
}

// ── Test 6: Boundary test — 1×1×1 image ──────────────────────────────

#[test]
fn test_1x1x1_background() {
    // 1×1×1 all-background: no foreground seeds.
    // Convention (Sprint 81): EDT over empty set → 0.0 (safe sentinel).
    let image = make_image_3d(vec![0.0], [1, 1, 1]);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);
    assert_eq!(vals.len(), 1);
    assert_eq!(
        vals[0], 0.0,
        "1x1x1 all-background EDT²=0 (empty foreground, no seeds)"
    );
}

#[test]
fn test_1x1x1_foreground() {
    // 1×1×1 foreground: voxel is its own seed → EDT²=0.
    let image = make_image_3d(vec![1.0], [1, 1, 1]);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);
    assert_eq!(vals.len(), 1);
    assert_eq!(vals[0], 0.0, "1x1x1 foreground EDT²=0 (it is the seed)");
}

// ── Test 7: Two background voxels — verify minimum is chosen ──────────

#[test]
fn test_two_background_voxels_minimum_distance() {
    // 1×1×7 row. Background at x=0 and x=6. Foreground at x=1..5.
    // New convention: foreground seeds at x=1..5.
    // x=0 (bg): nearest foreground at x=1 → EDT²=1.
    // x=1..5 (fg): seeds → EDT²=0.
    // x=6 (bg): nearest foreground at x=5 → EDT²=1.
    let nx = 7;
    let mut data = vec![1.0f32; nx];
    data[0] = 0.0;
    data[6] = 0.0;

    let image = make_image_3d(data, [1, 1, nx]);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);

    let expected = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0];
    for x in 0..nx {
        assert_eq!(
            vals[x], expected[x],
            "EDT²(0,0,{x}) = {}, expected {}",
            vals[x], expected[x]
        );
    }
}

// ── Test 8: DistanceTransform unit struct API consistency ──────────────

#[test]
fn test_unit_struct_api_matches_free_functions() {
    let dims = [3, 3, 3];
    let mut data = vec![1.0f32; 27];
    data[0] = 0.0;

    let image = make_image_3d(data, dims);
    let free_sq = distance_transform_squared(&image, 0.5);
    let struct_sq = DistanceTransform::squared(&image, 0.5);

    let free_vals = get_values(&free_sq);
    let struct_vals = get_values(&struct_sq);
    assert_eq!(
        free_vals, struct_vals,
        "unit struct must match free function"
    );

    let free_edt = distance_transform(&image, 0.5);
    let struct_edt = DistanceTransform::transform(&image, 0.5);

    let free_edt_vals = get_values(&free_edt);
    let struct_edt_vals = get_values(&struct_edt);
    assert_eq!(
        free_edt_vals, struct_edt_vals,
        "unit struct transform must match free function"
    );
}

// ── Test 9: Spatial metadata preserved ────────────────────────────────

#[test]
fn test_preserves_spatial_metadata() {
    let device = Default::default();
    let tensor = Tensor::<TestBackend, 3>::from_data(
        TensorData::new(vec![0.0f32; 8], Shape::new([2, 2, 2])),
        &device,
    );
    let origin = Point::new([1.0, 2.0, 3.0]);
    let spacing = Spacing::new([0.5, 0.5, 0.5]);
    let direction = Direction::identity();
    let image: Image<TestBackend, 3> = Image::new(tensor, origin, spacing, direction);

    let result = distance_transform_squared(&image, 0.5);
    assert_eq!(result.origin(), &origin);
    assert_eq!(result.spacing(), &spacing);
    assert_eq!(result.direction(), &direction);
    assert_eq!(result.shape(), [2, 2, 2]);
}

// ── Test 10: Checkerboard pattern ─────────────────────────────────────

#[test]
fn test_checkerboard_volumetric() {
    // 1×2×2 checkerboard:
    //   (0,0,0)=bg, (0,0,1)=fg, (0,1,0)=fg, (0,1,1)=bg
    // New convention: foreground voxels are seeds.
    // bg(0,0,0): nearest fg at (0,0,1) or (0,1,0), distance 1 → EDT²=1.
    // fg(0,0,1): seed → EDT²=0.
    // fg(0,1,0): seed → EDT²=0.
    // bg(0,1,1): nearest fg at (0,0,1) or (0,1,0), distance 1 → EDT²=1.
    let data = vec![0.0, 1.0, 1.0, 0.0];
    let image = make_image_3d(data, [1, 2, 2]);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);

    assert_eq!(vals[0], 1.0, "(0,0,0) bg, nearest fg at distance 1");
    assert_eq!(vals[1], 0.0, "(0,0,1) fg seed");
    assert_eq!(vals[2], 0.0, "(0,1,0) fg seed");
    assert_eq!(vals[3], 1.0, "(0,1,1) bg, nearest fg at distance 1");
}

// ── Test 11: Asymmetric shape ─────────────────────────────────────────

#[test]
fn test_asymmetric_shape_2x3x4() {
    // 2×3×4, all foreground except (0,0,0) = background.
    // New convention: 23 foreground voxels are seeds → EDT²=0.
    // Background (0,0,0): nearest foreground at (1,0,0)/(0,1,0)/(0,0,1) → EDT²=1.
    let dims = [2, 3, 4];
    let total = 24;
    let mut data = vec![1.0f32; total];
    data[0] = 0.0;

    let image = make_image_3d(data, dims);
    let result = distance_transform_squared(&image, 0.5);
    let vals = get_values(&result);

    // All foreground voxels are seeds.
    for z in 0..2usize {
        for y in 0..3usize {
            for x in 0..4usize {
                if (z, y, x) != (0, 0, 0) {
                    let actual = at(&vals, z, y, x, 3, 4);
                    assert_eq!(
                        actual, 0.0,
                        "foreground voxel EDT²({z},{y},{x}) = {actual}, expected 0"
                    );
                }
            }
        }
    }
    // Background (0,0,0): nearest foreground at distance 1.
    let bg = at(&vals, 0, 0, 0, 3, 4);
    assert_eq!(bg, 1.0, "background (0,0,0) EDT²=1, got {bg}");
}

// ── Internal: lower_envelope_transform correctness ────────────────────

#[test]
fn test_lower_envelope_single_element() {
    let f = [7i64];
    let mut dt = [0i64];
    let mut v = [0usize];
    let mut z = [0i64; 2];
    lower_envelope_transform(&f[..], 1, &mut dt[..], &mut v[..], &mut z[..]);
    assert_eq!(dt[0], 7, "single element passthrough");
}

#[test]
fn test_lower_envelope_uniform() {
    // f = [5, 5, 5, 5]. dt[i] = min_j { (i-j)² + 5 } = 5 (at j=i).
    let f = [5i64; 4];
    let mut dt = [0i64; 4];
    let mut v = [0usize; 4];
    let mut z = [0i64; 5];
    lower_envelope_transform(&f[..], 4, &mut dt[..], &mut v[..], &mut z[..]);
    for i in 0..4 {
        assert_eq!(dt[i], 5, "uniform f: dt[{i}] = 5, got {}", dt[i]);
    }
}

#[test]
fn test_lower_envelope_known_case() {
    // f = [0, INF, INF, INF, 0] where INF = 100.
    // dt[i] = min(i², (i-4)²) since f[0]=0, f[4]=0.
    // dt[0]=0, dt[1]=1, dt[2]=4, dt[3]=1, dt[4]=0.
    let inf = 100i64;
    let f = [0, inf * inf, inf * inf, inf * inf, 0];
    let mut dt = [0i64; 5];
    let mut v = [0usize; 5];
    let mut z = [0i64; 6];
    lower_envelope_transform(&f[..], 5, &mut dt[..], &mut v[..], &mut z[..]);
    assert_eq!(dt[0], 0);
    assert_eq!(dt[1], 1);
    assert_eq!(dt[2], 4);
    assert_eq!(dt[3], 1);
    assert_eq!(dt[4], 0);
}

// ── Test: phase1_row ──────────────────────────────────────────────────

#[test]
fn test_phase1_row_all_background() {
    // All background, no foreground seeds → inf sentinel for all positions.
    let row = [false, false, false, false];
    let mut out = [0i64; 4];
    phase1_row(&row[..], 4, 100, &mut out[..]);
    assert_eq!(out, [100, 100, 100, 100]);
}

#[test]
fn test_phase1_row_single_bg_at_start() {
    // Foreground seeds at x=1..4. Background x=0: nearest foreground at x=1 → dist=1.
    let row = [false, true, true, true, true];
    let mut out = [0i64; 5];
    phase1_row(&row[..], 5, 100, &mut out[..]);
    assert_eq!(out, [1, 0, 0, 0, 0]);
}

#[test]
fn test_phase1_row_bg_at_both_ends() {
    // Foreground seeds at x=1,2,3. Background x=0 → dist=1; background x=4 → dist=1.
    let row = [false, true, true, true, false];
    let mut out = [0i64; 5];
    phase1_row(&row[..], 5, 100, &mut out[..]);
    assert_eq!(out, [1, 0, 0, 0, 1]);
}

#[test]
fn test_phase1_row_all_foreground() {
    // All foreground: every position is a seed → distance = 0 for all.
    let row = [true, true, true];
    let mut out = [0i64; 3];
    phase1_row(&row[..], 3, 100, &mut out[..]);
    assert_eq!(out, [0, 0, 0]);
}
