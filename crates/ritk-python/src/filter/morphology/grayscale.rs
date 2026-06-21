use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    BlackTopHatFilter, GrayscaleClosingFilter, GrayscaleDilation, GrayscaleErosion,
    GrayscaleOpeningFilter, WhiteTopHatFilter,
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
pub fn grayscale_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = GrayscaleErosion::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
pub fn grayscale_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = GrayscaleDilation::new(radius);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply white top-hat transform (image minus morphological opening).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn white_top_hat(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        WhiteTopHatFilter::new(radius)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply black top-hat transform (morphological closing minus image).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn black_top_hat(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        BlackTopHatFilter::new(radius)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Grayscale morphological closing with a flat cubic (box) SE of half-width
/// `radius`. ITK Parity: GrayscaleMorphologicalClosingImageFilter
/// (`sitk.GrayscaleMorphologicalClosing`, box SE).
#[pyfunction]
pub fn grayscale_closing(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        GrayscaleClosingFilter::new(radius)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Grayscale morphological opening with a flat cubic (box) SE of half-width
/// `radius`. ITK Parity: GrayscaleMorphologicalOpeningImageFilter
/// (`sitk.GrayscaleMorphologicalOpening`, box SE).
#[pyfunction]
pub fn grayscale_opening(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        GrayscaleOpeningFilter::new(radius)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
