//! Image intensity normalization: min-max, z-score, histogram matching,
//! white stripe, and Nyul-Udupa piecewise-linear standardization.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_from_py, into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_statistics::normalization::white_stripe::{
    MriContrast, WhiteStripeConfig, WhiteStripeNormalizer,
};
use ritk_statistics::normalization::{
    HistogramMatcher, MinMaxNormalizer, NyulUdupaNormalizer, ZScoreNormalizer,
};

/// Validate that `target_min < target_max` for a min-max range normalization.
pub(super) fn validate_range(target_min: f32, target_max: f32) -> Result<(), String> {
    if target_min >= target_max {
        Err(format!(
            "minmax_normalize_range: target_min ({target_min}) must be strictly less than \
             target_max ({target_max})"
        ))
    } else {
        Ok(())
    }
}

/// Validate that `percentiles` is strictly ascending with at least 2 elements.
pub(super) fn validate_percentiles(p: &[f64]) -> Result<(), String> {
    if p.len() < 2 {
        return Err(format!(
            "nyul_udupa_normalize: percentiles must contain ≥ 2 values, got {}",
            p.len()
        ));
    }
    for i in 1..p.len() {
        if p[i] <= p[i - 1] {
            return Err(format!(
                "nyul_udupa_normalize: percentiles must be strictly ascending: \
                 p[{}]={} ≤ p[{}]={}",
                i,
                p[i],
                i - 1,
                p[i - 1]
            ));
        }
    }
    Ok(())
}

/// Normalize image intensities to [0, 1] via min-max rescaling.
///
/// Formula: output = (input − min) / (max − min + ε), ε = 1e-8.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     Normalized PyImage with intensities in [0, 1].
#[pyfunction]
pub fn minmax_normalize(image: &PyImage) -> PyImage {
    into_py_image(MinMaxNormalizer::new().normalize(&image_from_py(image)))
}

/// Normalize image intensities to [target_min, target_max] via min-max rescaling.
///
/// Formula: N(x) = (x − min) / (max − min + ε); output = N(x) · (target_max − target_min) + target_min.
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
) -> RitkResult<PyImage> {
    if let Err(e) = validate_range(target_min, target_max) {
        return Err(RitkPyError::value(e));
    }
    Ok(into_py_image(
        MinMaxNormalizer::with_range(target_min, target_max).normalize(&image_from_py(image)),
    ))
}

/// Normalize image intensities to zero mean and unit variance (Z-score).
///
/// Formula: output = (input − μ) / (σ + ε), ε = 1e-8.
///
/// Args:
///     image: Input PyImage.
///     mask:  Optional binary mask PyImage (foreground > 0.5). Must have the
///            same element count as `image`. Defaults to None (full-image stats).
///
/// Returns:
///     Normalized PyImage with E\[output\] ≈ 0, Var\[output\] ≈ 1.
#[pyfunction]
#[pyo3(signature = (image, mask = None))]
pub fn zscore_normalize(
    py: Python<'_>,
    image: &PyImage,
    mask: Option<&PyImage>,
) -> RitkResult<PyImage> {
    let image_arc = image_from_py(image);
    let result = match mask {
        Some(m) => {
            if image_arc.shape() != m.inner.shape() {
                return Err(RitkPyError::value(
                    "zscore_normalize: mask must have the same shape as image",
                ));
            }
            let mask_arc = image_from_py(m);
            py.allow_threads(|| ZScoreNormalizer::new().normalize_masked(&image_arc, &mask_arc))
        }
        None => py.allow_threads(|| ZScoreNormalizer::new().normalize(&image_arc)),
    };
    Ok(into_py_image(result))
}

/// Match the intensity histogram of a source image to a reference image.
///
/// Follows ITK's `HistogramMatchingImageFilter`: quantile match-point landmarks
/// with an optional mean threshold, mapped piecewise-linearly. Equivalent to
/// `SimpleITK.HistogramMatching(source, reference, num_bins, num_match_points,
/// threshold_at_mean)`.
///
/// Args:
///     source:           Image whose histogram is to be transformed (PyImage).
///     reference:        Image whose histogram is the target distribution (PyImage).
///     num_bins:         Number of histogram levels (default 256).
///     num_match_points: Number of interior quantile match points (default 7).
///     threshold_at_mean: Exclude sub-mean (background) intensities from the
///                        landmark estimation (default True).
///
/// Returns:
///     PyImage with matched histogram, same shape and spatial metadata as source.
#[pyfunction]
#[pyo3(signature = (source, reference, num_bins = 256, num_match_points = 7, threshold_at_mean = true))]
pub fn histogram_match(
    py: Python<'_>,
    source: &PyImage,
    reference: &PyImage,
    num_bins: usize,
    num_match_points: usize,
    threshold_at_mean: bool,
) -> RitkResult<PyImage> {
    if num_bins < 2 {
        return Err(RitkPyError::value(format!(
            "histogram_match: num_bins must be ≥ 2, got {num_bins}"
        )));
    }
    let source_arc = image_from_py(source);
    let reference_arc = image_from_py(reference);
    let result = py.allow_threads(|| {
        HistogramMatcher::new(num_bins)
            .with_match_points(num_match_points)
            .with_threshold_at_mean(threshold_at_mean)
            .match_histograms(&source_arc, &reference_arc)
    });
    Ok(into_py_image(result))
}

/// Normalize a brain MRI using the Shinohara et al. (2014) white stripe method.
///
/// Detects the WM peak via KDE, selects voxels within a quantile stripe,
/// and normalizes: I_norm = (I − μ_ws) / (σ_ws + ε).
///
/// Args:
///     image:    Input brain MRI PyImage.
///     mask:     Optional brain mask PyImage (foreground > 0.5).
///     contrast: MRI contrast type, "t1" or "t2" (case-insensitive, default "t1").
///     width:    White stripe half-width in quantile units (default 0.05).
///
/// Returns:
///     Tuple of (normalized_image, mu, sigma, wm_peak, stripe_size).
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
) -> RitkResult<(PyImage, f64, f64, f64, usize)> {
    let mri_contrast = match contrast.unwrap_or("t1") {
        "t1" | "T1" => MriContrast::T1,
        "t2" | "T2" => MriContrast::T2,
        other => {
            return Err(RitkPyError::value(format!(
                "white_stripe_normalize: contrast must be \"t1\" or \"t2\", got \"{other}\""
            )));
        }
    };

    let config = WhiteStripeConfig {
        contrast: mri_contrast,
        width: width.unwrap_or(0.05),
        ..Default::default()
    };

    let image_arc = image_from_py(image);
    let mask_arc = mask.map(image_from_py);

    let result = py.allow_threads(|| {
        let mask_ref = mask_arc.as_ref();
        WhiteStripeNormalizer::normalize(&image_arc, mask_ref, &config)
    });

    Ok((
        into_py_image(result.normalized),
        result.mu,
        result.sigma,
        result.wm_peak,
        result.stripe_size,
    ))
}

/// Normalize an image using Nyul-Udupa piecewise-linear histogram standardization.
///
/// Reference: Nyul & Udupa (1999), *IEEE Trans. Med. Imaging* 18(4):301-306.
///
/// Args:
///     image:           Image to normalize.
///     training_images: List of PyImage used to learn the standard landmarks.
///     percentiles:     Optional list of percentile ranks in [0, 100] (strictly
///                      ascending, ≥ 2 values). Default: [1,10,20,...,90,99].
///
/// Returns:
///     Normalized PyImage with the same shape and spatial metadata as `image`.
///
/// Raises:
///     ValueError:   if percentiles has fewer than 2 values or is not strictly ascending.
///     RuntimeError: if training_images is empty or normalization fails.
#[pyfunction]
#[pyo3(signature = (image, training_images, percentiles = None))]
pub fn nyul_udupa_normalize(
    py: Python<'_>,
    image: &PyImage,
    training_images: Vec<PyRef<'_, PyImage>>,
    percentiles: Option<Vec<f64>>,
) -> RitkResult<PyImage> {
    if training_images.is_empty() {
        return Err(RitkPyError::runtime(
            "training_images must contain at least one image",
        ));
    }

    if let Some(ref p) = percentiles {
        if let Err(e) = validate_percentiles(p) {
            return Err(RitkPyError::value(e));
        }
    }

    let training_arcs: Vec<_> = training_images
        .iter()
        .map(|py_img| image_from_py(py_img))
        .collect();
    let input_arc = image_from_py(image);

    py.allow_threads(|| {
        let refs: Vec<_> = training_arcs.iter().collect();
        let mut normalizer = match percentiles {
            Some(p) => NyulUdupaNormalizer::with_percentiles(p),
            None => NyulUdupaNormalizer::new(),
        };
        normalizer.learn_standard(&refs);
        normalizer.apply(&input_arc).map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(into_py_image)
}

#[cfg(test)]
mod tests {
    use super::{validate_percentiles, validate_range};

    #[test]
    fn test_validate_percentiles_empty_returns_error() {
        let result = validate_percentiles(&[]);
        assert!(result.is_err(), "empty slice must be rejected");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("≥ 2"),
            "error must mention ≥ 2 values, got: {msg}"
        );
    }

    #[test]
    fn test_validate_percentiles_single_element_returns_error() {
        let result = validate_percentiles(&[0.5]);
        assert!(result.is_err(), "single element must be rejected");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("≥ 2"),
            "error must mention ≥ 2 values, got: {msg}"
        );
    }

    #[test]
    fn test_validate_percentiles_equal_elements_returns_error() {
        let result = validate_percentiles(&[0.1, 0.1]);
        assert!(result.is_err(), "equal elements must be rejected");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("strictly ascending"),
            "error must mention strictly ascending, got: {msg}"
        );
    }

    #[test]
    fn test_validate_percentiles_descending_elements_returns_error() {
        let result = validate_percentiles(&[0.5, 0.1, 0.9]);
        assert!(result.is_err(), "descending pair must be rejected");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("strictly ascending"),
            "error must mention strictly ascending, got: {msg}"
        );
    }

    #[test]
    fn test_validate_percentiles_two_valid_ascending_returns_ok() {
        let result = validate_percentiles(&[0.1, 0.9]);
        assert_eq!(result, Ok(()), "two ascending values must be accepted");
    }

    #[test]
    fn test_validate_percentiles_multi_ascending_returns_ok() {
        let result = validate_percentiles(&[
            0.01, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.99,
        ]);
        assert_eq!(result, Ok(()), "standard Nyul percentiles must be accepted");
    }

    #[test]
    fn test_validate_range_equal_bounds_returns_error() {
        let result = validate_range(0.5, 0.5);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("strictly less than"),
            "error must mention 'strictly less than', got: {msg}"
        );
    }

    #[test]
    fn test_validate_range_inverted_bounds_returns_error() {
        let result = validate_range(1.0, 0.0);
        assert!(result.is_err());
        let msg = result.unwrap_err();
        assert!(
            msg.contains("strictly less than"),
            "error must mention 'strictly less than', got: {msg}"
        );
    }

    #[test]
    fn test_validate_range_zero_to_one_returns_ok() {
        assert!(validate_range(0.0, 1.0).is_ok());
    }

    #[test]
    fn test_validate_range_negative_to_positive_returns_ok() {
        assert!(validate_range(-1.0, 1.0).is_ok());
    }
}
