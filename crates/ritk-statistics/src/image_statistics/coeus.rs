//! Coeus-backed image statistics boundary.
//!
//! The statistical algorithm remains owned by the parent module.  This module
//! only adapts `ritk_image::coeus::Image` into borrowed host slices through the
//! Coeus tensor-ops migration seam.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::coeus::Image;
use ritk_tensor_ops::coeus as tensor_ops;

use super::{compute_from_values, compute_statistics_from_slice, ImageStatistics};

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
    Ok(compute_statistics_from_slice(values, 0))
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

    if image_values.len() != mask_values.len() {
        anyhow::bail!(
            "coeus image statistics: image element count {} does not match mask element count {}",
            image_values.len(),
            mask_values.len()
        );
    }

    let values: Vec<f32> = image_values
        .iter()
        .zip(mask_values.iter())
        .filter(|(_, &mask)| mask > crate::FOREGROUND_THRESHOLD)
        .map(|(&value, _)| value)
        .collect();

    if values.is_empty() {
        anyhow::bail!("coeus image statistics: mask contains no foreground voxels");
    }

    Ok(compute_from_values(&values, 0))
}
