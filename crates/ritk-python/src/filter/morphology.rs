//! Morphological filters: grayscale erosion/dilation, label morphology, top-hat, hit-or-miss.

use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::{
    BlackTopHatFilter, GrayscaleDilation, GrayscaleErosion, HitOrMissTransform, LabelClosing,
    LabelDilation, LabelErosion, LabelOpening, MorphologicalReconstruction, ReconstructionMode,
    WhiteTopHatFilter,
};

/// Apply grayscale morphological erosion with a flat cubic structuring element.
///
/// Each output voxel is the minimum of its (2r+1)³ cubic neighbourhood
/// (replicate padding at boundaries).
///
/// Args:
///     image:  Input PyImage.
///     radius: Structuring element half-width in voxels (default 1 → 3×3×3).
///
/// Returns:
///     Eroded PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn grayscale_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = GrayscaleErosion::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Apply grayscale morphological dilation with a flat cubic structuring element.
///
/// Each output voxel is the maximum of its (2r+1)³ cubic neighbourhood
/// (replicate padding at boundaries).
///
/// Args:
///     image:  Input PyImage.
///     radius: Structuring element half-width in voxels (default 1 → 3×3×3).
///
/// Returns:
///     Dilated PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, radius=1))]
pub fn grayscale_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = GrayscaleDilation::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Erode labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelErosion::new(radius)
            .apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Opening on a 3-D label volume (erosion followed by dilation).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_opening(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelOpening::new(radius)
            .apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Closing on a 3-D label volume (dilation followed by erosion).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_closing(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let img = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelClosing::new(radius)
            .apply(img.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Dilate labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        LabelDilation::new(radius)
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    });
    Ok(into_py_image(result?))
}

/// Apply white top-hat transform (image minus morphological opening).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn white_top_hat(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        WhiteTopHatFilter::new(radius)
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    });
    Ok(into_py_image(result?))
}

/// Apply black top-hat transform (morphological closing minus image).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn black_top_hat(py: Python<'_>, image: &PyImage, radius: usize) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        BlackTopHatFilter::new(radius)
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    });
    Ok(into_py_image(result?))
}

/// Apply hit-or-miss transform for shape detection in binary images.
///
/// Args:
///     image:     Binary input PyImage.
///     fg_radius: Foreground structuring element radius.
///     bg_radius: Background structuring element radius.
#[pyfunction]
#[pyo3(signature = (image, fg_radius=1_usize, bg_radius=2_usize))]
pub fn hit_or_miss(
    py: Python<'_>,
    image: &PyImage,
    fg_radius: usize,
    bg_radius: usize,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        HitOrMissTransform::new(fg_radius, bg_radius)
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    });
    Ok(into_py_image(result?))
}

/// Geodesic morphological reconstruction.
#[pyfunction]
#[pyo3(signature = (marker, mask, mode = "dilation"))]
pub fn morphological_reconstruction(
    py: Python<'_>,
    marker: &PyImage,
    mask: &PyImage,
    mode: &str,
) -> PyResult<PyImage> {
    let recon_mode = match mode {
        "dilation" => ReconstructionMode::Dilation,
        "erosion" => ReconstructionMode::Erosion,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown reconstruction mode '{}'. Use 'dilation' or 'erosion'.",
                other
            )))
        }
    };
    let marker_arc = std::sync::Arc::clone(&marker.inner);
    let mask_arc = std::sync::Arc::clone(&mask.inner);
    let result = py.allow_threads(|| {
        MorphologicalReconstruction::new(recon_mode)
            .apply(marker_arc.as_ref(), mask_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}
