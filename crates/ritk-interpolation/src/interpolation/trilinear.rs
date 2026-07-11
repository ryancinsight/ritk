//! Native coordinate-grid trilinear interpolation.

use coeus_core::{ComputeBackend, CpuAddressableStorage, CpuAddressableStorageMut};
use ritk_image::native::Image;

/// Sample a rank-5 image at a `(z, y, x)` voxel-coordinate grid.
///
/// The image shape is `[batch, channel, depth, height, width]`; the grid shape
/// is `[batch, 3, output_depth, output_height, output_width]`. Out-of-bounds
/// coordinates use border replication. Output physical metadata is inherited
/// from `image`.
///
/// # Errors
///
/// Returns an error when the Coeus interpolation contract is violated or the
/// rank-5 output image cannot be constructed.
pub fn trilinear_interpolation<B>(
    image: &Image<f32, B, 5>,
    grid: &Image<f32, B, 5>,
) -> anyhow::Result<Image<f32, B, 5>>
where
    B: coeus_core::Backend + ComputeBackend + Default,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32> + CpuAddressableStorageMut<f32>,
{
    let output = coeus_ops::linear_interpolation::<3, _, _>(
        image.data(),
        grid.data(),
        coeus_ops::Replicate,
    )?;
    Image::new(
        output,
        *image.origin(),
        *image.spacing(),
        *image.direction(),
    )
}

#[cfg(test)]
#[path = "tests_trilinear.rs"]
mod tests;
