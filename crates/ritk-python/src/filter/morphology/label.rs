use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, py_image_to_burn, PyImage};
use pyo3::prelude::*;
use ritk_filter::{LabelClosing, LabelDilation, LabelErosion, LabelOpening};

/// Erode labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_erosion(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let img = py_image_to_burn(image);
    py.allow_threads(|| {
        LabelErosion::new(radius)
            .apply(&img)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Opening on a 3-D label volume (erosion followed by dilation).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_opening(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let img = py_image_to_burn(image);
    py.allow_threads(|| {
        LabelOpening::new(radius)
            .apply(&img)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Closing on a 3-D label volume (dilation followed by erosion).
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_closing(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let img = py_image_to_burn(image);
    py.allow_threads(|| {
        LabelClosing::new(radius)
            .apply(&img)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Dilate labeled regions in a 3-D label volume.
#[pyfunction]
#[pyo3(signature = (image, radius=1_usize))]
pub fn label_dilation(py: Python<'_>, image: &PyImage, radius: usize) -> RitkResult<PyImage> {
    let image = py_image_to_burn(image);
    py.allow_threads(|| {
        LabelDilation::new(radius)
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}
