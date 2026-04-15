//! Python-exposed image statistics, comparison, and normalization functions.
//!
//! All functions delegate to `ritk_core::statistics` implementations (SSOT).
//!
//! # Module structure
//!
//! ## Descriptive statistics
//! - [`compute_statistics`]: Min, max, mean, std, p25/p50/p75 over all voxels.
//! - [`masked_statistics`]:  Same, restricted to foreground voxels of a binary mask.
//!
//! ## Image comparison
//! - [`dice_coefficient`]:      Sørensen–Dice overlap for a given label.
//! - [`hausdorff_distance`]:    Symmetric Hausdorff distance between binary masks.
//! - [`mean_surface_distance`]: Symmetric mean surface distance between binary masks.
//! - [`psnr`]:                  Peak signal-to-noise ratio.
//! - [`ssim`]:                  Structural similarity index.
//!
//! ## Noise estimation
//! - [`estimate_noise`]: MAD-based Gaussian noise σ̂ (optionally masked).
//!
//! ## Normalization
//! - [`minmax_normalize`]:       Min-max rescale to \[0, 1\].
//! - [`minmax_normalize_range`]: Min-max rescale to \[target\_min, target\_max\].
//! - [`zscore_normalize`]:       Z-score standardization (zero mean, unit variance).
//! - [`histogram_match`]:        CDF-based histogram matching to a reference image.
//! - [`white_stripe_normalize`]: Shinohara et al. (2014) white stripe MRI normalization.

use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use ritk_core::statistics::normalization::white_stripe::{
    MriContrast, WhiteStripeConfig, WhiteStripeNormalizer,
};
use ritk_core::statistics::normalization::{HistogramMatcher, MinMaxNormalizer, ZScoreNormalizer};
use ritk_core::statistics::{
    compute_statistics as core_compute_statistics, dice_coefficient as core_dice_coefficient,
    estimate_noise_mad, estimate_noise_mad_masked, hausdorff_distance as core_hausdorff_distance,
    masked_statistics as core_masked_statistics,
    mean_surface_distance as core_mean_surface_distance, psnr as core_psnr, ssim as core_ssim,
    ImageStatistics,
};
use std::sync::Arc;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Convert [`ImageStatistics`] to a Python dict with keys:
/// `min`, `max`, `mean`, `std`, `p25`, `p50`, `p75`.
fn stats_to_dict(py: Python<'_>, stats: &ImageStatistics) -> PyResult<Py<PyDict>> {
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

// ── compute_statistics ────────────────────────────────────────────────────────

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
pub fn compute_statistics(py: Python<'_>, image: &PyImage) -> PyResult<Py<PyDict>> {
    let stats = core_compute_statistics(image.inner.as_ref());
    stats_to_dict(py, &stats)
}

// ── masked_statistics ─────────────────────────────────────────────────────────

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
pub fn masked_statistics(py: Python<'_>, image: &PyImage, mask: &PyImage) -> PyResult<Py<PyDict>> {
    let stats = core_masked_statistics(image.inner.as_ref(), mask.inner.as_ref());
    stats_to_dict(py, &stats)
}

// ── dice_coefficient ──────────────────────────────────────────────────────────

/// Compute the Sørensen–Dice coefficient between two binary masks.
///
/// Delegates to `ritk_core::statistics::dice_coefficient`.
///
/// Measures the overlap between two binary segmentation maps.
/// Both images should contain binary values (0/1).
///
/// Args:
///     image1: First binary segmentation PyImage.
///     image2: Second binary segmentation PyImage (same shape as image1).
///
/// Returns:
///     Dice coefficient in [0, 1]. Returns 1.0 if both images have zero
///     volume.
#[pyfunction]
pub fn dice_coefficient(image1: &PyImage, image2: &PyImage) -> PyResult<f32> {
    Ok(core_dice_coefficient(
        image1.inner.as_ref(),
        image2.inner.as_ref(),
    ))
}

// ── hausdorff_distance ────────────────────────────────────────────────────────

/// Compute the symmetric Hausdorff distance between two binary masks.
///
/// Delegates to `ritk_core::statistics::hausdorff_distance`.
/// The distance is in physical units (mm) derived from the images' voxel spacing.
///
/// The Hausdorff distance is the maximum of the two directed Hausdorff
/// distances: HD(A, B) = max( hd(∂A→∂B), hd(∂B→∂A) ).
///
/// Args:
///     image1: First binary mask PyImage.
///     image2: Second binary mask PyImage (same shape as image1).
///
/// Returns:
///     Hausdorff distance (float, mm). Returns 0.0 if both boundaries are empty.
#[pyfunction]
pub fn hausdorff_distance(image1: &PyImage, image2: &PyImage) -> PyResult<f32> {
    let sp = image1.inner.spacing();
    let spacing: [f64; 3] = [sp[0], sp[1], sp[2]];
    Ok(core_hausdorff_distance(
        image1.inner.as_ref(),
        image2.inner.as_ref(),
        &spacing,
    ))
}

// ── mean_surface_distance ─────────────────────────────────────────────────────

/// Compute the symmetric mean surface distance between two binary masks.
///
/// Delegates to `ritk_core::statistics::mean_surface_distance`.
/// The distance is in physical units (mm) derived from the images' voxel spacing.
///
/// MSD = ( MSD(∂A→∂B) + MSD(∂B→∂A) ) / 2.
///
/// Args:
///     image1: First binary mask PyImage.
///     image2: Second binary mask PyImage (same shape as image1).
///
/// Returns:
///     Mean surface distance (float, mm). Returns 0.0 if both boundaries are empty.
#[pyfunction]
pub fn mean_surface_distance(image1: &PyImage, image2: &PyImage) -> PyResult<f32> {
    let sp = image1.inner.spacing();
    let spacing: [f64; 3] = [sp[0], sp[1], sp[2]];
    Ok(core_mean_surface_distance(
        image1.inner.as_ref(),
        image2.inner.as_ref(),
        &spacing,
    ))
}

// ── psnr ──────────────────────────────────────────────────────────────────────

/// Compute the Peak Signal-to-Noise Ratio between two images.
///
/// Delegates to `ritk_core::statistics::psnr`.
///
/// Formula: PSNR = 10 · log₁₀(MAX² / MSE), where MSE is the mean squared
/// error between the two images.
///
/// Args:
///     image1: Test image (PyImage).
///     image2: Reference image (PyImage, same shape as image1).
///
/// Returns:
///     PSNR in decibels (dB). Returns infinity when images are identical (MSE = 0).
///
/// Args:
///     image1:  Test image (PyImage).
///     image2:  Reference image (PyImage, same shape as image1).
///     max_val: Dynamic range of pixel values (default 1.0, suitable for
///              normalized images; use 255.0 for 8-bit data).
///
/// Returns:
///     PSNR in decibels (dB).
#[pyfunction]
#[pyo3(signature = (image1, image2, max_val=1.0))]
pub fn psnr(image1: &PyImage, image2: &PyImage, max_val: f32) -> PyResult<f32> {
    Ok(core_psnr(
        image1.inner.as_ref(),
        image2.inner.as_ref(),
        max_val,
    ))
}

// ── ssim ──────────────────────────────────────────────────────────────────────

/// Compute the Structural Similarity Index (SSIM) between two images.
///
/// Delegates to `ritk_core::statistics::ssim`.
/// Uses the Wang et al. (2004) formulation with stability constants
/// C₁ = (0.01·MAX)² and C₂ = (0.03·MAX)².
///
/// Args:
///     image1: Test image (PyImage).
///     image2: Reference image (PyImage, same shape as image1).
///
/// Returns:
///     SSIM in [-1, 1]. Returns 1.0 for identical images.
///
/// Args:
///     image1:  Test image (PyImage).
///     image2:  Reference image (PyImage, same shape as image1).
///     max_val: Dynamic range of pixel values (default 1.0, suitable for
///              normalized images; use 255.0 for 8-bit data).
///
/// Returns:
///     SSIM in [-1, 1].
#[pyfunction]
#[pyo3(signature = (image1, image2, max_val=1.0))]
pub fn ssim(image1: &PyImage, image2: &PyImage, max_val: f32) -> PyResult<f32> {
    Ok(core_ssim(
        image1.inner.as_ref(),
        image2.inner.as_ref(),
        max_val,
    ))
}

// ── estimate_noise ────────────────────────────────────────────────────────────

/// Estimate additive Gaussian noise σ̂ via the Median Absolute Deviation (MAD).
///
/// Delegates to `ritk_core::statistics::estimate_noise_mad` when no mask is
/// provided, or `ritk_core::statistics::estimate_noise_mad_masked` when a mask
/// is given.
///
/// Formula: σ̂ = 1.4826 · median(|Xᵢ − median(X)|).
///
/// The 1.4826 constant is 1 / Φ⁻¹(3/4), making the MAD a consistent
/// estimator of σ under Gaussian noise (Hampel 1974).
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
pub fn estimate_noise(image: &PyImage, mask: Option<&PyImage>) -> PyResult<f32> {
    let sigma = match mask {
        Some(m) => estimate_noise_mad_masked(image.inner.as_ref(), m.inner.as_ref()),
        None => estimate_noise_mad(image.inner.as_ref()),
    };
    Ok(sigma)
}

// ── minmax_normalize ──────────────────────────────────────────────────────────

/// Normalize image intensities to [0, 1] via min-max rescaling.
///
/// Delegates to `ritk_core::statistics::normalization::MinMaxNormalizer`.
///
/// Formula: output = (input − min) / (max − min + ε), ε = 1e-8.
/// Spatial metadata (origin, spacing, direction) is preserved.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     Normalized PyImage with intensities in [0, 1].
#[pyfunction]
pub fn minmax_normalize(image: &PyImage) -> PyResult<PyImage> {
    let result = MinMaxNormalizer::new().normalize(image.inner.as_ref());
    Ok(into_py_image(result))
}

// ── minmax_normalize_range ────────────────────────────────────────────────────

/// Normalize image intensities to [target_min, target_max] via min-max rescaling.
///
/// Delegates to `ritk_core::statistics::normalization::MinMaxNormalizer`.
///
/// Formula:
///   N(x) = (x − min) / (max − min + ε)
///   output = N(x) · (target_max − target_min) + target_min
///
/// Spatial metadata (origin, spacing, direction) is preserved.
///
/// Args:
///     image:      Input PyImage.
///     target_min: Lower bound of the output intensity range.
///     target_max: Upper bound of the output intensity range.
///
/// Returns:
///     Normalized PyImage with intensities in [target_min, target_max].
#[pyfunction]
pub fn minmax_normalize_range(
    image: &PyImage,
    target_min: f32,
    target_max: f32,
) -> PyResult<PyImage> {
    let result =
        MinMaxNormalizer::with_range(target_min, target_max).normalize(image.inner.as_ref());
    Ok(into_py_image(result))
}

// ── zscore_normalize ──────────────────────────────────────────────────────────

/// Normalize image intensities to zero mean and unit variance (Z-score).
///
/// Delegates to `ritk_core::statistics::normalization::ZScoreNormalizer`.
///
/// Formula: output = (input − μ) / (σ + ε), ε = 1e-8.
/// μ and σ are population statistics computed from all voxels.
/// Spatial metadata (origin, spacing, direction) is preserved.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     Normalized PyImage with E[output] ≈ 0, Var[output] ≈ 1.
#[pyfunction]
pub fn zscore_normalize(image: &PyImage) -> PyResult<PyImage> {
    let result = ZScoreNormalizer::new().normalize(image.inner.as_ref());
    Ok(into_py_image(result))
}

// ── histogram_match ───────────────────────────────────────────────────────────

/// Match the intensity histogram of a source image to a reference image.
///
/// Delegates to `ritk_core::statistics::normalization::HistogramMatcher`.
///
/// Applies the transform T(v) = F_ref⁻¹(F_src(v)) via a piecewise-linear
/// lookup table built from the empirical CDFs of both images.
/// Spatial metadata of the source is preserved.
///
/// Args:
///     source:    Image whose histogram is to be transformed (PyImage).
///     reference: Image whose histogram serves as the target distribution (PyImage).
///
/// Returns:
///     PyImage with matched histogram, same shape and spatial metadata as source.
#[pyfunction]
pub fn histogram_match(py: Python<'_>, source: &PyImage, reference: &PyImage) -> PyResult<PyImage> {
    let source_arc = Arc::clone(&source.inner);
    let reference_arc = Arc::clone(&reference.inner);
    let result = py.allow_threads(|| {
        HistogramMatcher::new(256).match_histograms(source_arc.as_ref(), reference_arc.as_ref())
    });
    Ok(into_py_image(result))
}

// ── white_stripe_normalize ────────────────────────────────────────────────────

/// Normalize a brain MRI using the Shinohara et al. (2014) white stripe method.
///
/// Delegates to `ritk_core::statistics::normalization::white_stripe::WhiteStripeNormalizer`.
///
/// Detects the white matter (WM) peak in the intensity histogram via kernel
/// density estimation (KDE), selects voxels within a quantile stripe around
/// the peak, and normalizes the full image to zero mean and unit variance
/// within that stripe: I_norm = (I − μ_ws) / (σ_ws + ε).
///
/// Args:
///     image:    Input brain MRI PyImage.
///     mask:     Optional brain mask PyImage (foreground > 0.5).
///               If None, all voxels with intensity > 0 are used as foreground.
///     contrast: MRI contrast type, "t1" or "t2" (case-insensitive, default "t1").
///               T1: WM peak searched in the upper half of the histogram.
///               T2: WM peak searched in the lower half of the histogram.
///     width:    White stripe half-width in quantile units (default 0.05,
///               i.e. ±5 percentile points around the WM peak quantile).
///
/// Returns:
///     Tuple of (normalized_image, mu, sigma, wm_peak, stripe_size):
///       - normalized_image: PyImage normalized by white stripe statistics.
///       - mu:          White stripe mean intensity (float).
///       - sigma:       White stripe population standard deviation (float).
///       - wm_peak:     Detected white matter peak intensity (float).
///       - stripe_size: Number of voxels in the white stripe (int).
///
/// Raises:
///     ValueError:   if contrast is not "t1" or "t2".
///     RuntimeError: if no foreground voxels exist or white stripe is empty.
#[pyfunction]
#[pyo3(signature = (image, mask=None, contrast=None, width=None))]
pub fn white_stripe_normalize(
    py: Python<'_>,
    image: &PyImage,
    mask: Option<&PyImage>,
    contrast: Option<&str>,
    width: Option<f64>,
) -> PyResult<(PyImage, f64, f64, f64, usize)> {
    let mri_contrast = match contrast.unwrap_or("t1") {
        "t1" | "T1" => MriContrast::T1,
        "t2" | "T2" => MriContrast::T2,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "white_stripe_normalize: contrast must be \"t1\" or \"t2\", got \"{other}\""
            )));
        }
    };

    let config = WhiteStripeConfig {
        contrast: mri_contrast,
        width: width.unwrap_or(0.05),
        ..Default::default()
    };

    // Clone Arc handles before releasing the GIL — zero-cost, no voxel copying.
    let image_arc = Arc::clone(&image.inner);
    let mask_arc = mask.map(|m| Arc::clone(&m.inner));

    let result = py.allow_threads(|| {
        let mask_ref = mask_arc.as_ref().map(|arc| arc.as_ref());
        WhiteStripeNormalizer::normalize(image_arc.as_ref(), mask_ref, &config)
    });

    Ok((
        into_py_image(result.normalized),
        result.mu,
        result.sigma,
        result.wm_peak,
        result.stripe_size,
    ))
}

// ── Submodule registration ────────────────────────────────────────────────────

/// Register the `statistics` submodule and all 13 exposed functions into `parent`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "statistics")?;
    m.add_function(wrap_pyfunction!(compute_statistics, &m)?)?;
    m.add_function(wrap_pyfunction!(masked_statistics, &m)?)?;
    m.add_function(wrap_pyfunction!(dice_coefficient, &m)?)?;
    m.add_function(wrap_pyfunction!(hausdorff_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_surface_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(psnr, &m)?)?;
    m.add_function(wrap_pyfunction!(ssim, &m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise, &m)?)?;
    m.add_function(wrap_pyfunction!(minmax_normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(minmax_normalize_range, &m)?)?;
    m.add_function(wrap_pyfunction!(zscore_normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(histogram_match, &m)?)?;
    m.add_function(wrap_pyfunction!(white_stripe_normalize, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
