//! Coeus-native image comparison metrics.
//!
//! These are the Coeus-backend counterparts to the Burn-generic free functions
//! in the parent module. Each adapter borrows contiguous host storage from a
//! [`ritk_image::Image`] via the [`ritk_tensor_ops::native`] seam and
//! delegates to the same shared host core the Coeus path uses, so the metric math
//! has exactly one home per metric (no cloned algorithm).
//!
//! Metrics whose Coeus path stays on-device (`dice_coefficient`, `psnr`) share
//! their *host* core with the native path here; metrics whose Coeus path already
//! computes on host slices (`similarity_index`, `ssim`, `hausdorff_distance`,
//! `mean_surface_distance`) share the identical core.

use coeus_core::{ComputeBackend, CpuAddressableStorage};
use ritk_image::Image;
use ritk_tensor_ops::native as tensor_ops;

use super::overlap::{dice_from_slices, similarity_index_from_slices};
use super::quality::{psnr_from_slices, ssim_from_slices};
use super::surface::{hausdorff_from_flat, msd_from_flat};

/// Dice similarity coefficient between two Coeus-backed binary masks.
///
/// See [`super::dice_coefficient`] for the formula. Returns `1.0` when both
/// masks are empty.
///
/// # Errors
/// Returns an error when either tensor is not host-addressable/contiguous, or
/// when the two images have different element counts.
pub fn dice_coefficient<B, const D: usize>(
    prediction: &Image<f32, B, D>,
    ground_truth: &Image<f32, B, D>,
) -> anyhow::Result<f32>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (pred, _) = tensor_ops::extract_image_slice(prediction)?;
    let (gt, _) = tensor_ops::extract_image_slice(ground_truth)?;
    ensure_same_len(pred.len(), gt.len(), "dice_coefficient")?;
    Ok(dice_from_slices(pred, gt))
}

/// ITK `SimilarityIndex` between two Coeus-backed images (binarized: any nonzero
/// voxel is foreground). Returns `0.0` when both foreground sets are empty.
///
/// # Errors
/// Returns an error when either tensor is not host-addressable/contiguous, or
/// when the two images have different element counts.
pub fn similarity_index<B, const D: usize>(
    a: &Image<f32, B, D>,
    b: &Image<f32, B, D>,
) -> anyhow::Result<f32>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (sa, _) = tensor_ops::extract_image_slice(a)?;
    let (sb, _) = tensor_ops::extract_image_slice(b)?;
    ensure_same_len(sa.len(), sb.len(), "similarity_index")?;
    Ok(similarity_index_from_slices(sa, sb))
}

/// Peak signal-to-noise ratio between two Coeus-backed images.
///
/// Returns `f32::INFINITY` when the images are identical. See [`super::psnr`].
///
/// # Errors
/// Returns an error when either tensor is not host-addressable/contiguous, or
/// when the two images have different element counts.
pub fn psnr<B, const D: usize>(
    image: &Image<f32, B, D>,
    reference: &Image<f32, B, D>,
    max_val: f32,
) -> anyhow::Result<f32>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (img, _) = tensor_ops::extract_image_slice(image)?;
    let (rf, _) = tensor_ops::extract_image_slice(reference)?;
    ensure_same_len(img.len(), rf.len(), "psnr")?;
    Ok(psnr_from_slices(img, rf, max_val))
}

/// Global structural similarity index between two Coeus-backed images.
///
/// See [`super::ssim`] (Wang et al. 2004, single global window).
///
/// # Errors
/// Returns an error when either tensor is not host-addressable/contiguous, or
/// when the two images have different element counts.
pub fn ssim<B, const D: usize>(
    image: &Image<f32, B, D>,
    reference: &Image<f32, B, D>,
    max_val: f32,
) -> anyhow::Result<f32>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (img, _) = tensor_ops::extract_image_slice(image)?;
    let (rf, _) = tensor_ops::extract_image_slice(reference)?;
    ensure_same_len(img.len(), rf.len(), "ssim")?;
    ssim_from_slices(img, rf, max_val)
}

/// Symmetric Hausdorff distance between two Coeus-backed binary masks.
///
/// See [`super::hausdorff_distance`]. `spacing` is the physical voxel size per
/// axis.
///
/// # Errors
/// Returns an error when either tensor is not host-addressable or contiguous.
pub fn hausdorff_distance<B, const D: usize>(
    prediction: &Image<f32, B, D>,
    ground_truth: &Image<f32, B, D>,
    spacing: &[f64; D],
) -> anyhow::Result<f32>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (pred, pred_shape) = tensor_ops::extract_image_slice(prediction)?;
    let (gt, gt_shape) = tensor_ops::extract_image_slice(ground_truth)?;
    Ok(hausdorff_from_flat(pred, pred_shape, gt, gt_shape, spacing))
}

/// Symmetric mean surface distance between two Coeus-backed binary masks.
///
/// See [`super::mean_surface_distance`].
///
/// # Errors
/// Returns an error when either tensor is not host-addressable or contiguous.
pub fn mean_surface_distance<B, const D: usize>(
    prediction: &Image<f32, B, D>,
    ground_truth: &Image<f32, B, D>,
    spacing: &[f64; D],
) -> anyhow::Result<f32>
where
    B: ComputeBackend,
    B::DeviceBuffer<f32>: CpuAddressableStorage<f32>,
{
    let (pred, pred_shape) = tensor_ops::extract_image_slice(prediction)?;
    let (gt, gt_shape) = tensor_ops::extract_image_slice(ground_truth)?;
    Ok(msd_from_flat(pred, pred_shape, gt, gt_shape, spacing))
}

/// Reject element-count mismatches at the metric boundary.
fn ensure_same_len(a: usize, b: usize, metric: &str) -> anyhow::Result<()> {
    if a != b {
        anyhow::bail!(
            "coeus {metric}: image element count {a} does not match reference element count {b}"
        );
    }
    Ok(())
}
