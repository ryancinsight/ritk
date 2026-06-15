use super::*;
use burn_ndarray::NdArray;

type B = NdArray<f32>;

// ── generate_grid ────────────────────────────────────────────────────────

/// Shape of a 3D grid is [D*H*W, 3].
#[test]
fn grid_3d_shape_is_n_by_3() {
    let device = Default::default();
    let g = generate_grid::<B, 3>([2, 3, 4], &device);
    let [rows, cols] = g.dims();
    assert_eq!(rows, 2 * 3 * 4, "row count must equal D*H*W");
    assert_eq!(cols, 3, "column count must be 3 (innermost-first [x,y,z])");
}

/// First voxel (z=0, y=0, x=0) must be [0, 0, 0].
#[test]
fn grid_3d_first_voxel_is_origin() {
    let device = Default::default();
    let g = generate_grid::<B, 3>([3, 3, 3], &device);
    let first = g.clone().slice([0..1]).into_data();
    let vals = first.as_slice::<f32>().unwrap();
    // column order: innermost-first [x, y, z]
    assert_eq!(vals[0], 0.0, "x of first voxel");
    assert_eq!(vals[1], 0.0, "y of first voxel");
    assert_eq!(vals[2], 0.0, "z of first voxel");
}

/// Last voxel of [D, H, W] must be [W-1, H-1, D-1] in innermost-first [x,y,z].
///
/// # Derivation
/// The innermost loop is over x (0..W), middle y (0..H), outer z (0..D), so the
/// last row is z=D-1, y=H-1, x=W-1. Columns are innermost-first (col0=x), so the
/// row is [x, y, z] = [W-1, H-1, D-1] — matching the interpolation kernels and
/// the (reversed-mapping) `index_to_world_tensor`.
#[test]
fn grid_3d_last_voxel_matches_shape_minus_one() {
    let device = Default::default();
    let d = 2usize;
    let h = 3usize;
    let w = 4usize;
    let g = generate_grid::<B, 3>([d, h, w], &device);
    let n = d * h * w;
    let last = g.clone().slice([n - 1..n]).into_data();
    let vals = last.as_slice::<f32>().unwrap();
    assert_eq!(vals[0], (w - 1) as f32, "x of last voxel");
    assert_eq!(vals[1], (h - 1) as f32, "y of last voxel");
    assert_eq!(vals[2], (d - 1) as f32, "z of last voxel");
}

/// Shape of a 2D grid is [H*W, 2].
#[test]
fn grid_2d_shape_is_n_by_2() {
    let device = Default::default();
    let g = generate_grid::<B, 2>([5, 7], &device);
    let [rows, cols] = g.dims();
    assert_eq!(rows, 5 * 7, "row count must equal H*W");
    assert_eq!(cols, 2, "column count must be 2 (x, y)");
}

/// First pixel (y=0, x=0) must be [0, 0].
#[test]
fn grid_2d_first_pixel_is_origin() {
    let device = Default::default();
    let g = generate_grid::<B, 2>([4, 6], &device);
    let first = g.clone().slice([0..1]).into_data();
    let vals = first.as_slice::<f32>().unwrap();
    assert_eq!(vals[0], 0.0, "x of first pixel");
    assert_eq!(vals[1], 0.0, "y of first pixel");
}

// ── generate_random_points ───────────────────────────────────────────────

/// Random points tensor must have shape [N, D].
#[test]
fn random_points_shape_is_n_by_d() {
    let device = Default::default();
    let pts = generate_random_points::<B, 3>([10, 20, 30], 50, &device);
    assert_eq!(pts.dims(), [50, 3]);
}

/// All random index values must lie within the per-axis bound, using the
/// innermost-first column convention shared with [`generate_grid`] and the
/// interpolation kernels: column 0 = x bounded by `shape[D-1]-1`, …,
/// column D-1 = z bounded by `shape[0]-1`.
///
/// # Derivation
/// For shape=[4, 6, 8] (= [z, y, x]), column j ranges over shape[D-1-j], so
/// the per-column maxima are [x_max, y_max, z_max] = [7, 5, 3].  Asserting
/// column 0 against shape[0] (z=3) would be the wrong axis — that mismatch is
/// the transposition bug that collapsed MI sampling on anisotropic images.
#[test]
fn random_points_within_bounds() {
    let device = Default::default();
    let shape = [4usize, 6, 8];
    let pts = generate_random_points::<B, 3>(shape, 200, &device);
    let data = pts.into_data();
    let vals = data.as_slice::<f32>().unwrap();
    // vals layout: [row0_x, row0_y, row0_z, row1_x, ...]
    for (i, &v) in vals.iter().enumerate() {
        let col = i % 3; // column 0 = x = shape[D-1], column D-1 = z = shape[0]
        let max_idx = (shape[3 - 1 - col] - 1) as f32;
        assert!(
            v >= 0.0 && v <= max_idx + 1e-4, // small epsilon for float rounding
            "random index col={col} out of [0, {max_idx}]: got {v}"
        );
    }
}

/// Regression for the axis-transposition bug: on a strongly anisotropic
/// shape the x-column (col 0) must be allowed to exceed the z-extent, and the
/// z-column (col 2) must stay within the (small) z-extent.  Under the old
/// `shape[col]` scaling the z-column would be drawn over [0, nx-1], sending
/// MI samples massively out of bounds and collapsing the joint histogram.
#[test]
fn random_points_anisotropic_axis_order() {
    let device = Default::default();
    let shape = [3usize, 256, 256]; // [z=3, y=256, x=256] — thin slab
    let pts = generate_random_points::<B, 3>(shape, 4000, &device);
    let data = pts.into_data();
    let vals = data.as_slice::<f32>().unwrap();
    let mut max_x = 0.0f32;
    let mut max_z = 0.0f32;
    for (i, &v) in vals.iter().enumerate() {
        match i % 3 {
            0 => max_x = max_x.max(v),
            2 => {
                max_z = max_z.max(v);
                assert!(
                    v <= (shape[0] - 1) as f32 + 1e-4,
                    "z-column (col 2) must stay within z-extent {}: got {v}",
                    shape[0] - 1
                );
            }
            _ => {}
        }
    }
    // With 4000 samples the x-column must reach well beyond the z-extent (2),
    // confirming column 0 spans the x-axis (255), not the z-axis.
    assert!(
        max_x > (shape[0] - 1) as f32,
        "x-column max {max_x} must exceed z-extent {}; axes are transposed",
        shape[0] - 1
    );
}
