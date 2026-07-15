use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use coeus_core::MoiraiBackend;
use pyo3::prelude::*;
use ritk_filter::{LabelClosing, LabelDilation, LabelErosion, LabelOpening};
use std::sync::Arc;

/// Erode labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        LabelErosion::new(radius)
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Opening on a 3-D label volume (erosion followed by dilation).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_opening(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        LabelOpening::new(radius)
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Closing on a 3-D label volume (dilation followed by erosion).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_closing(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        LabelClosing::new(radius)
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Dilate labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        LabelDilation::new(radius)
            .apply_native(native.as_ref(), &backend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
