//! Descriptive statistics, image comparison, noise estimation, and label statistics.

use crate::errors::RitkResult;
use crate::image::{with_tensor_slice, PyImage};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_core::statistics::compute_label_intensity_statistics_from_slices as core_label_intensity_stats_from_slices;
use ritk_core::statistics::image_statistics::compute_statistics_from_slice;
use ritk_core::statistics::noise_estimation::{
    estimate_noise_mad_from_slice, estimate_noise_mad_masked_from_slices,
};
use ritk_core::statistics::{
    dice_coefficient as core_dice_coefficient, hausdorff_distance as core_hausdorff_distance,
    mean_surface_distance as core_mean_surface_distance, psnr as core_psnr, ssim as core_ssim,
    ImageStatistics,
};
use std::sync::Arc;

/// Convert [`ImageStatistics`] to a Python dict with keys:
/// `min`, `max`, `mean`, `std`, `p25`, `p50`, `p75`.
pub(super) fn stats_to_dict(py: Python<'_>, stats: &ImageStatistics) -> RitkResult<Py<PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("min", stats.min)?;
    dict.set_item("max", stats.max)?;
    dict.set_item("mean", stats.mean)?;
    dict.set_item("std", stats.std)?;
    dict.set_item("p25", stats.percentiles[0])?;
    dict.set_item("p50", stats.percentiles[1])?;
    dict.set_item("p75", stats.percentiles[2])?;
    Ok(dict.unbind())
}

/// Compute descriptive statistics over all voxels in an image.
///
/// Delegates to `ritk_core::statistics::compute_statistics`.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     dict with keys: min, max, mean, std, p25, p50, p75 (all float).
///     Percentiles correspond to the 25th, 50th (median), and 75th percentiles.
#[pyfunction]
pub fn compute_statistics(py: Python<'_>, image: &PyImage) -> RitkResult<Py<PyDict>> {
    let stats = with_tensor_slice(image.inner.data(), compute_statistics_from_slice);
    stats_to_dict(py, &stats)
}

/// Compute descriptive statistics over foreground voxels (mask > 0.5).
///
/// Delegates to `ritk_core::statistics::masked_statistics`.
///
/// Args:
///     image: Input PyImage.
///     mask:  Binary mask PyImage (same shape as image; values > 0.5 = foreground).
///
/// Returns:
///     dict with keys: min, max, mean, std, p25, p50, p75 (all float).
///     Statistics are computed only over voxels where mask > 0.5.
///
/// Raises:
///     RuntimeError: if image and mask shapes differ or mask has no foreground voxels.
#[pyfunction]
pub fn masked_statistics(py: Python<'_>, image: &PyImage, mask: &PyImage) -> RitkResult<Py<PyDict>> {
    let stats = with_tensor_slice(image.inner.data(), |img_slice| {
        with_tensor_slice(mask.inner.data(), |mask_slice| {
            assert_eq!(
                img_slice.len(),
                mask_slice.len(),
                "image and mask must have identical element count"
            );
            let values: Vec<f32> = img_slice
                .iter()
                .zip(mask_slice.iter())
                .filter(|(_, &m)| m > 0.5)
                .map(|(&v, _)| v)
                .collect();
            assert!(!values.is_empty(), "mask contains no foreground voxels");
            compute_statistics_from_slice(&values)
        })
    });
    stats_to_dict(py, &stats)
}

/// Compute the Sørensen–Dice coefficient between two binary masks.
///
/// Delegates to `ritk_core::statistics::dice_coefficient`.
///
/// Args:
///     image1: First binary segmentation PyImage.
///     image2: Second binary segmentation PyImage (same shape as image1).
///
/// Returns:
///     Dice coefficient in [0, 1]. Returns 1.0 if both images have zero volume.
#[pyfunction]
pub fn dice_coefficient(image1: &PyImage, image2: &PyImage) -> f32 {
    core_dice_coefficient(
        image1.inner.as_ref(),
        image2.inner.as_ref(),
    )
}

/// Compute the symmetric Hausdorff distance between two binary masks.
///
/// Delegates to `ritk_core::statistics::hausdorff_distance`.
/// HD(A, B) = max( hd(∂A→∂B), hd(∂B→∂A) ). Distance in mm.
///
/// Args:
///     image1: First binary mask PyImage.
///     image2: Second binary mask PyImage (same shape as image1).
///
/// Returns:
///     Hausdorff distance (float, mm). Returns 0.0 if both boundaries are empty.
#[pyfunction]
pub fn hausdorff_distance(py: Python<'_>, image1: &PyImage, image2: &PyImage) -> f32 {
    let sp = image1.inner.spacing();
    let spacing: [f64; 3] = [sp[0], sp[1], sp[2]];
    let arc1 = Arc::clone(&image1.inner);
    let arc2 = Arc::clone(&image2.inner);
    py.allow_threads(|| core_hausdorff_distance(arc1.as_ref(), arc2.as_ref(), &spacing))
}

/// Compute the symmetric mean surface distance between two binary masks.
///
/// Delegates to `ritk_core::statistics::mean_surface_distance`.
/// MSD = ( MSD(∂A→∂B) + MSD(∂B→∂A) ) / 2. Distance in mm.
///
/// Args:
///     image1: First binary mask PyImage.
///     image2: Second binary mask PyImage (same shape as image1).
///
/// Returns:
///     Mean surface distance (float, mm). Returns 0.0 if both boundaries are empty.
#[pyfunction]
pub fn mean_surface_distance(py: Python<'_>, image1: &PyImage, image2: &PyImage) -> f32 {
    let sp = image1.inner.spacing();
    let spacing: [f64; 3] = [sp[0], sp[1], sp[2]];
    let arc1 = Arc::clone(&image1.inner);
    let arc2 = Arc::clone(&image2.inner);
    py.allow_threads(|| core_mean_surface_distance(arc1.as_ref(), arc2.as_ref(), &spacing))
}

/// Compute the Peak Signal-to-Noise Ratio between two images.
///
/// Delegates to `ritk_core::statistics::psnr`.
/// Formula: PSNR = 10 · log₁₀(MAX² / MSE).
///
/// Args:
///     image1:  Test image (PyImage).
///     image2:  Reference image (PyImage, same shape as image1).
///     max_val: Dynamic range of pixel values (default 1.0).
///
/// Returns:
///     PSNR in decibels (dB). Returns infinity when images are identical (MSE = 0).
#[pyfunction]
#[pyo3(signature = (image1, image2, max_val=1.0))]
pub fn psnr(image1: &PyImage, image2: &PyImage, max_val: f32) -> f32 {
    core_psnr(
        image1.inner.as_ref(),
        image2.inner.as_ref(),
        max_val,
    )
}

/// Compute the Structural Similarity Index (SSIM) between two images.
///
/// Delegates to `ritk_core::statistics::ssim`.
/// Wang et al. (2004), C₁ = (0.01·MAX)², C₂ = (0.03·MAX)².
///
/// Args:
///     image1:  Test image (PyImage).
///     image2:  Reference image (PyImage, same shape as image1).
///     max_val: Dynamic range of pixel values (default 1.0).
///
/// Returns:
///     SSIM in [-1, 1]. Returns 1.0 for identical images.
#[pyfunction]
#[pyo3(signature = (image1, image2, max_val=1.0))]
pub fn ssim(image1: &PyImage, image2: &PyImage, max_val: f32) -> f32 {
    core_ssim(
        image1.inner.as_ref(),
        image2.inner.as_ref(),
        max_val,
    )
}

/// Estimate additive Gaussian noise σ̂ via the Median Absolute Deviation (MAD).
///
/// Formula: σ̂ = 1.4826 · median(|Xᵢ − median(X)|).
/// The 1.4826 constant is 1 / Φ⁻¹(3/4) (Hampel 1974).
///
/// Args:
///     image: Input PyImage.
///     mask:  Optional binary mask PyImage. If provided, only foreground
///            voxels (mask > 0.5) contribute to the estimate.
///
/// Returns:
///     Estimated noise standard deviation (float). Returns 0.0 for constant
///     images or empty masks.
#[pyfunction]
#[pyo3(signature = (image, mask=None))]
pub fn estimate_noise(image: &PyImage, mask: Option<&PyImage>) -> f32 {
    match mask {
        Some(m) => with_tensor_slice(image.inner.data(), |img_slice| {
            with_tensor_slice(m.inner.data(), |mask_slice| {
                estimate_noise_mad_masked_from_slices(img_slice, mask_slice)
            })
        }),
        None => with_tensor_slice(image.inner.data(), estimate_noise_mad_from_slice),
    }
}

/// Compute per-label intensity statistics over a co-registered intensity image.
///
/// Delegates to `ritk_core::statistics::compute_label_intensity_statistics_from_slices`.
/// Background (label 0) is excluded. Results are sorted by label ascending.
///
/// Args:
///     label_image:     Label image (integer labels stored as f32; 0 = background).
///     intensity_image: Intensity image with identical shape to `label_image`.
///
/// Returns:
///     list of dicts, one per label, sorted ascending by label, each with keys:
///     `label` (int), `count` (int), `min` (float), `max` (float),
///     `mean` (float), `std` (float).
///
/// Raises:
///     RuntimeError: if images have different element counts or shapes.
#[pyfunction]
pub fn compute_label_intensity_statistics(
    py: Python<'_>,
    label_image: &PyImage,
    intensity_image: &PyImage,
) -> RitkResult<Py<PyList>> {
    let stats = with_tensor_slice(label_image.inner.data(), |label_slice| {
        with_tensor_slice(intensity_image.inner.data(), |intensity_slice| {
            core_label_intensity_stats_from_slices(label_slice, intensity_slice)
        })
    });
    let list = PyList::empty_bound(py);
    for s in &stats {
        let dict = PyDict::new_bound(py);
        dict.set_item("label", s.label)?;
        dict.set_item("count", s.count)?;
        dict.set_item("min", s.min)?;
        dict.set_item("max", s.max)?;
        dict.set_item("mean", s.mean)?;
        dict.set_item("std", s.std)?;
        list.append(dict)?;
    }
    Ok(list.into())
}
