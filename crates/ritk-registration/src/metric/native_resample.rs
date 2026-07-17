//! Shared Coeus-native resample substrate for register metrics.
//!
//! Every intensity-based native metric (NGF, MSE, …) resamples the moving image
//! onto the fixed grid through the identical path — the fixed index grid, the
//! native index→world map, the native affine transform, the native world→index
//! map, and the native trilinear kernel. That path lives here ONCE (SSOT); each
//! metric module keeps only its own arithmetic on the resampled host values.
//!
//! Conventions (bit-faithful to the Burn `grid::generate_grid` /
//! `index_to_world_tensor` / `world_to_index_tensor` path the native batch
//! transforms reproduce): index-space columns are innermost-first
//! (`col 0 = x = axis D-1`), world-space columns are axis-major.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut};
use coeus_tensor::Tensor;
use ritk_image::native::Image;
use ritk_interpolation::trilinear_interpolation;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;

/// World coordinates (axis-major, flat `[N·3]`) of every fixed-grid voxel, in
/// fixed C-order — computed once per fixed image via the native batch transform.
pub fn fixed_world_points<B>(fixed: &Image<f32, B, 3>) -> Vec<f32>
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + coeus_core::CpuAddressableStorageMut<f32>,
{
    let shape = fixed.shape();
    let idx = ritk_image::grid::generate_grid::<f32, B, 3>(shape, &B::default());
    fixed.index_to_world_native(&idx).as_slice().to_vec()
}

/// Resample `moving` through `transform` at the fixed-grid world points
/// (`[N·3]` axis-major, from [`fixed_world_points`]), returning the `N`
/// interpolated host values in fixed C-order.
///
/// Fixed world points (axis-major) are affine-transformed into moving-space
/// world points, mapped to moving continuous indices (innermost-first), and
/// sampled by the native trilinear kernel — whose grid layout is `[1, 3, N, 1, 1]`
/// with channels (z, y, x) = axes (0, 1, 2), so channel `k` reads the
/// innermost-first index column `2 - k`.
pub fn resample_moving_at_world<B>(
    fixed_world: &[f32],
    moving: &Image<f32, B, 3>,
    transform: &AtlasAffineTransform<B, 3>,
) -> Vec<f32>
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let n = fixed_world.len() / 3;

    let world_img = Image::<f32, B, 2>::from_flat(
        fixed_world.to_vec(),
        [n, 3],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )
    .expect("fixed world points carry a valid [N, 3] rank-2 layout");
    let moving_world = transform
        .transform_points(&world_img)
        .expect("affine transform of [N, 3] world points")
        .data_vec();

    let moving_world_t = Tensor::<f32, B>::from_slice([n, 3], &moving_world);
    let moving_idx = moving.world_to_index_native(&moving_world_t);
    let mi = moving_idx.as_slice();

    let mut grid = vec![0.0f32; 3 * n];
    for p in 0..n {
        grid[p] = mi[p * 3 + 2]; // channel 0 = z (axis 0) ← col 2
        grid[n + p] = mi[p * 3 + 1]; // channel 1 = y (axis 1) ← col 1
        grid[2 * n + p] = mi[p * 3]; // channel 2 = x (axis 2) ← col 0
    }

    let [d, h, w] = moving.shape();
    let moving_rank5 = Image::<f32, B, 5>::from_flat_on(
        moving.data_vec(),
        [1, 1, d, h, w],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &B::default(),
    )
    .expect("moving image data matches rank-5 interpolation shape");
    let grid_rank5 = Image::<f32, B, 5>::from_flat_on(
        grid,
        [1, 3, n, 1, 1],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
        &B::default(),
    )
    .expect("sampling grid data matches rank-5 interpolation shape");
    let sampled = trilinear_interpolation(&moving_rank5, &grid_rank5)
        .expect("native trilinear interpolation accepts generated grid");
    sampled.data_vec()
}
