use super::*;
use coeus_core::SequentialBackend;

type B = SequentialBackend;

// ── generate_grid ────────────────────────────────────────────────────────

/// Shape of a 3D grid is [D*H*W, 3].
#[test]
fn grid_3d_shape_is_n_by_3() {
    let backend = B::default();
    let g = generate_grid::<f32, B, 3>([2, 3, 4], &backend);
    let shape = g.shape();
    assert_eq!(shape[0], 2 * 3 * 4, "row count must equal D*H*W");
    assert_eq!(
        shape[1], 3,
        "column count must be 3 (innermost-first [x,y,z])"
    );
}

/// First voxel (z=0, y=0, x=0) must be [0, 0, 0].
#[test]
fn grid_3d_first_voxel_is_origin() {
    let backend = B::default();
    let g = generate_grid::<f32, B, 3>([3, 3, 3], &backend);
    let first = g.slice(&[(0, 1), (0, 3)]);
    let vals = first.as_slice();
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
    let backend = B::default();
    let d = 2usize;
    let h = 3usize;
    let w = 4usize;
    let g = generate_grid::<f32, B, 3>([d, h, w], &backend);
    let n = d * h * w;
    let last = g.slice(&[(n - 1, n), (0, 3)]);
    let vals = last.as_slice();
    assert_eq!(vals[0], (w - 1) as f32, "x of last voxel");
    assert_eq!(vals[1], (h - 1) as f32, "y of last voxel");
    assert_eq!(vals[2], (d - 1) as f32, "z of last voxel");
}

/// Shape of a 2D grid is [H*W, 2].
#[test]
fn grid_2d_shape_is_n_by_2() {
    let backend = B::default();
    let g = generate_grid::<f32, B, 2>([5, 7], &backend);
    let shape = g.shape();
    assert_eq!(shape[0], 5 * 7, "row count must equal H*W");
    assert_eq!(shape[1], 2, "column count must be 2 (x, y)");
}

/// First pixel (y=0, x=0) must be [0, 0].
#[test]
fn grid_2d_first_pixel_is_origin() {
    let backend = B::default();
    let g = generate_grid::<f32, B, 2>([4, 6], &backend);
    let first = g.slice(&[(0, 1), (0, 2)]);
    let vals = first.as_slice();
    assert_eq!(vals[0], 0.0, "x of first pixel");
    assert_eq!(vals[1], 0.0, "y of first pixel");
}
