//! Coeus-backed image statistics boundary.
//!
//! The statistical algorithm remains owned by the parent module.  This module
//! only adapts `ritk_image::Image` into borrowed host slices through the
//! Coeus tensor-ops migration seam.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use ritk_tensor_ops::native as tensor_ops;

use super::{compute_statistics_from_slice, masked_statistics_from_slices, ImageStatistics};

/// Compute statistics over all voxels in a Coeus-backed image.
///
/// # Errors
/// Returns an error when the image tensor is not host-addressable, rank-checked,
/// or contiguous according to the Coeus tensor-ops boundary.
pub fn compute_statistics<B, const D: usize>(
    image: &Image<f32, B, D>,
) -> anyhow::Result<ImageStatistics>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (values, _) = tensor_ops::extract_image_slice(image)?;
    Ok(compute_statistics_from_slice(values, 0)?)
}

/// Compute statistics over voxels where `mask > 0.5`.
///
/// # Errors
/// Returns an error when image extraction fails, the image and mask element
/// counts differ, or the mask contains no foreground voxels.
pub fn masked_statistics<B, const D: usize>(
    image: &Image<f32, B, D>,
    mask: &Image<f32, B, D>,
) -> anyhow::Result<ImageStatistics>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (image_values, _) = tensor_ops::extract_image_slice(image)?;
    let (mask_values, _) = tensor_ops::extract_image_slice(mask)?;

    Ok(masked_statistics_from_slices(image_values, mask_values, 0)?)
}
