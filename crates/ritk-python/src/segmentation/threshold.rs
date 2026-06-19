//! Automatic thresholding methods: Otsu, Li, Yen, Kapur, Triangle, Multi-Otsu, binary threshold.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, vec_to_image_like, with_tensor_slice, PyImage};
use pyo3::prelude::*;
use ritk_segmentation::threshold::huang::compute_huang_threshold_from_slice;
use ritk_segmentation::threshold::intermodes::compute_intermodes_threshold_from_slice;
use ritk_segmentation::threshold::isodata::compute_isodata_threshold_from_slice;
use ritk_segmentation::threshold::kapur::compute_kapur_threshold_from_slice;
use ritk_segmentation::threshold::kittler::compute_kittler_illingworth_threshold_from_slice;
use ritk_segmentation::threshold::li::compute_li_threshold_from_slice;
use ritk_segmentation::threshold::moments::compute_moments_threshold_from_slice;
use ritk_segmentation::threshold::multi_otsu::compute_multi_otsu_thresholds_from_slice;
use ritk_segmentation::threshold::otsu::compute_otsu_threshold_from_slice;
use ritk_segmentation::threshold::renyi::compute_renyi_entropy_threshold_from_slice;
use ritk_segmentation::threshold::shanbhag::compute_shanbhag_threshold_from_slice;
use ritk_segmentation::threshold::triangle::compute_triangle_threshold_from_slice;
use ritk_segmentation::threshold::yen::compute_yen_threshold_from_slice;
use ritk_segmentation::BinaryThreshold;
use std::sync::Arc;

/// Compute the Otsu threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::OtsuThreshold` (256-bin histogram,
/// maximises between-class variance σ²_B).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn otsu_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_otsu_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Li minimum cross-entropy threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::LiThreshold` (256-bin histogram,
/// iterative cross-entropy minimisation, Li & Tam 1998).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn li_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_li_threshold_from_slice(slice, 256, 1000);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Yen maximum correlation threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::YenThreshold` (256-bin histogram,
/// Yen, Chang & Chang 1995).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn yen_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_yen_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Kapur maximum entropy threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::KapurThreshold` (256-bin histogram,
/// Kapur, Sahoo & Wong 1985).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn kapur_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_kapur_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the IsoData (Ridler–Calvard) threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::IsoDataThreshold` (256-bin histogram).
#[pyfunction]
pub fn isodata_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_isodata_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Renyi-entropy threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::RenyiEntropyThreshold` (256-bin histogram).
#[pyfunction]
pub fn renyi_entropy_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_renyi_entropy_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Kittler-Illingworth minimum-error threshold and produce a mask.
///
/// Delegates to `ritk_segmentation::KittlerIllingworthThreshold` (256-bin histogram).
#[pyfunction]
pub fn kittler_illingworth_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_kittler_illingworth_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Shanbhag threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::ShanbhagThreshold` (256-bin histogram).
#[pyfunction]
pub fn shanbhag_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_shanbhag_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Huang (fuzzy-entropy) threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::HuangThreshold` (256-bin histogram).
#[pyfunction]
pub fn huang_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_huang_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Intermodes threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::IntermodesThreshold` (256-bin histogram).
#[pyfunction]
pub fn intermodes_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_intermodes_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Moments (Tsai) threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::MomentsThreshold` (256-bin histogram).
#[pyfunction]
pub fn moments_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_moments_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute the Triangle (Zack) threshold and produce a binary mask.
///
/// Delegates to `ritk_segmentation::TriangleThreshold` (256-bin histogram,
/// Zack, Rogers & Latt 1977).
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     (threshold, mask): threshold value as f32 and binary mask as PyImage.
#[pyfunction]
pub fn triangle_threshold(py: Python<'_>, image: &PyImage) -> (f32, PyImage) {
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (threshold, mask_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let threshold = compute_triangle_threshold_from_slice(slice, 256);
            let mask_vals: Vec<f32> = slice
                .iter()
                .map(|&v| if v >= threshold { 1.0_f32 } else { 0.0_f32 })
                .collect();
            (threshold, mask_vals)
        })
    });
    let mask = vec_to_image_like(mask_vals, dims, arc.as_ref());
    (threshold, into_py_image(mask))
}

/// Compute multi-class Otsu thresholds and produce a labeled image.
///
/// Delegates to `ritk_segmentation::MultiOtsuThreshold`. Returns K−1
/// thresholds and a label image with class indices {0, 1, …, K−1} as f32.
///
/// Args:
///     image:       Input PyImage.
///     num_classes: Number of intensity classes (≥ 2). Default 3.
///
/// Returns:
///     (thresholds, labeled_image): list of K−1 threshold values and labeled PyImage.
#[pyfunction]
#[pyo3(signature = (image, num_classes=3))]
pub fn multi_otsu_threshold(
    py: Python<'_>,
    image: &PyImage,
    num_classes: usize,
) -> RitkResult<(Vec<f32>, PyImage)> {
    if num_classes < 2 {
        return Err(RitkPyError::value("num_classes must be ≥ 2"));
    }
    let arc = Arc::clone(&image.inner);
    let dims = arc.shape();
    let (thresholds, label_vals) = with_tensor_slice(arc.data(), |slice| {
        py.allow_threads(|| {
            let thresholds = compute_multi_otsu_thresholds_from_slice(slice, num_classes, 256);
            let label_vals: Vec<f32> = slice
                .iter()
                .map(|&v| thresholds.iter().filter(|&&t| v >= t).count() as f32)
                .collect();
            (thresholds, label_vals)
        })
    });
    let labeled = vec_to_image_like(label_vals, dims, arc.as_ref());
    Ok((thresholds, into_py_image(labeled)))
}

/// Apply user-specified binary threshold segmentation.
///
/// Classifies voxels in `[lower, upper]` as `inside_value` (default 1.0)
/// and all others as `outside_value` (default 0.0).
///
/// Delegates to `ritk_segmentation::BinaryThreshold`.
///
/// Args:
///     image:         Input PyImage.
///     lower:         Lower intensity bound (inclusive). Defaults to f32::NEG_INFINITY.
///     upper:         Upper intensity bound (inclusive). Defaults to f32::INFINITY.
///     inside_value:  Output value for voxels inside the band (default 1.0).
///     outside_value: Output value for voxels outside the band (default 0.0).
///
/// Returns:
///     Binary-labeled PyImage with the same shape as `image`.
///
/// Raises:
///     ValueError: if lower > upper, or if inside_value / outside_value is not finite.
#[pyfunction]
#[pyo3(signature = (image, lower=None, upper=None, inside_value=1.0, outside_value=0.0))]
pub fn binary_threshold_segment(
    py: Python<'_>,
    image: &PyImage,
    lower: Option<f32>,
    upper: Option<f32>,
    inside_value: f32,
    outside_value: f32,
) -> RitkResult<PyImage> {
    let lower = lower.unwrap_or(f32::NEG_INFINITY);
    let upper = upper.unwrap_or(f32::INFINITY);
    if lower > upper {
        return Err(RitkPyError::value(format!(
            "lower bound ({lower}) must be ≤ upper bound ({upper})"
        )));
    }
    if !inside_value.is_finite() {
        return Err(RitkPyError::value(format!(
            "inside_value must be finite, got {inside_value}"
        )));
    }
    if !outside_value.is_finite() {
        return Err(RitkPyError::value(format!(
            "outside_value must be finite, got {outside_value}"
        )));
    }
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        BinaryThreshold::new(lower, upper)
            .with_values(inside_value, outside_value)
            .apply(image.as_ref())
    });
    Ok(into_py_image(result))
}
