//! Coeus-native 3-D resampling substrate.
//!
//! The native registration metrics and dense-field warp share one physical
//! resampling pipeline: fixed-grid indices become axis-major world points,
//! optional affine mapping produces moving-space points, and trilinear sampling
//! evaluates the moving volume. All samples outside ITK's half-voxel buffer are
//! zero-filled rather than clamped.

use coeus_core::{ComputeBackend, CpuAddressableStorageMut};
use coeus_tensor::Tensor;
use ritk_image::Image;
use ritk_interpolation::native::trilinear_interpolation;
use ritk_spatial::{Direction, Point, Spacing};
use ritk_transform::transform::affine::AtlasAffineTransform;

/// Return axis-major world coordinates for every voxel of `fixed` in C order.
#[must_use]
pub fn fixed_world_points<B>(fixed: &Image<f32, B, 3>) -> Vec<f32>
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let indices = ritk_image::grid::generate_grid::<f32, B, 3>(fixed.shape(), &B::default());
    fixed.index_to_world_native(&indices).as_slice().to_vec()
}

/// Sample `moving` at axis-major world points, returning zero outside its buffer.
///
/// `world_points` is a flat `[point_count, 3]` coordinate array in the same
/// axis-major order as `Image` metadata. Its length must be divisible by three.
///
/// # Errors
///
/// Returns an error when `world_points` is not a flat sequence of 3-D
/// coordinates.
pub fn sample_moving_at_world<B>(
    moving: &Image<f32, B, 3>,
    world_points: &[f32],
) -> anyhow::Result<Vec<f32>>
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    anyhow::ensure!(
        world_points.len().is_multiple_of(3),
        "world points must be a flat [point_count, 3] array, got length {}",
        world_points.len()
    );
    let point_count = world_points.len() / 3;
    let world = Tensor::<f32, B>::from_slice([point_count, 3], world_points);
    let indices = moving.world_to_index_native(&world);
    let indices = indices.as_slice();
    let mut grid = vec![0.0f32; 3 * point_count];
    for point in 0..point_count {
        grid[point] = indices[point * 3 + 2];
        grid[point_count + point] = indices[point * 3 + 1];
        grid[2 * point_count + point] = indices[point * 3];
    }

    let [depth, height, width] = moving.shape();
    anyhow::ensure!(
        depth > 0 && height > 0 && width > 0,
        "cannot sample an empty moving image with shape {:?}",
        moving.shape()
    );
    let moving_values = moving.data_cow();
    let sampled = trilinear_interpolation(
        &moving_values,
        1,
        1,
        depth,
        height,
        width,
        &grid,
        point_count,
        1,
        1,
    );

    Ok(sampled
        .into_iter()
        .enumerate()
        .map(|(point, value)| {
            let inside = (0..3).all(|column| {
                let axis = 2 - column;
                let index = indices[point * 3 + column];
                index >= -0.5 && index < moving.shape()[axis] as f32 - 0.5
            });
            if inside {
                value
            } else {
                0.0
            }
        })
        .collect())
}

/// Transform fixed-grid world points into moving space and sample `moving`.
///
/// `fixed_world` is a flat axis-major `[point_count, 3]` coordinate array.
/// The affine maps fixed physical coordinates to moving physical coordinates.
///
/// # Errors
///
/// Returns an error when `fixed_world` is not a flat sequence of 3-D
/// coordinates, a native point image cannot be built, or the affine rejects the
/// point batch.
pub fn resample_moving_at_world<B>(
    fixed_world: &[f32],
    moving: &Image<f32, B, 3>,
    transform: &AtlasAffineTransform<B, 3>,
) -> anyhow::Result<Vec<f32>>
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    anyhow::ensure!(
        fixed_world.len().is_multiple_of(3),
        "fixed world points must be a flat [point_count, 3] array, got length {}",
        fixed_world.len()
    );
    let point_count = fixed_world.len() / 3;
    let fixed_world = Image::<f32, B, 2>::from_flat(
        fixed_world.to_vec(),
        [point_count, 3],
        Point::origin(),
        Spacing::uniform(1.0),
        Direction::identity(),
    )?;
    let moving_world = transform
        .transform_points(&fixed_world)
        .map_err(|error| anyhow::anyhow!("affine resampling transform failed: {error}"))?
        .data_vec();
    sample_moving_at_world(moving, &moving_world)
}

/// Resample `moving` onto `reference` through a fixed-to-moving affine transform.
///
/// The output adopts `reference`'s geometry and has one value for every voxel
/// on its grid.
pub fn resample_image_native<B>(
    reference: &Image<f32, B, 3>,
    moving: &Image<f32, B, 3>,
    transform: &AtlasAffineTransform<B, 3>,
) -> anyhow::Result<Image<f32, B, 3>>
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: coeus_core::CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    Image::from_flat(
        resample_moving_at_world(&fixed_world_points(reference), moving, transform)?,
        reference.shape(),
        *reference.origin(),
        *reference.spacing(),
        *reference.direction(),
    )
}
