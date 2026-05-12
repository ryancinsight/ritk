//! Pixelwise binary image arithmetic filters.
//!
//! Each function combines two co-registered images of identical shape via a
//! pointwise binary operation applied independently to every voxel.
//! Spatial metadata (origin, spacing, direction) is taken from the first input image.
//!
//! ITK Parity: AddImageFilter, SubtractImageFilter, MultiplyImageFilter,
//!             DivideImageFilter, MinimumImageFilter, MaximumImageFilter

use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::{
    AddImageFilter, DivideImageFilter, ImageMaxFilter, ImageMinFilter, MultiplyImageFilter,
    SubtractImageFilter,
};

/// Pixelwise addition: out(x) = a(x) + b(x).
///
/// ITK Parity: AddImageFilter
#[pyfunction]
pub fn add_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> PyResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    let result = py.allow_threads(|| {
        AddImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Pixelwise subtraction: out(x) = a(x) - b(x).
///
/// ITK Parity: SubtractImageFilter
#[pyfunction]
pub fn subtract_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> PyResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    let result = py.allow_threads(|| {
        SubtractImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Pixelwise multiplication: out(x) = a(x) * b(x).
///
/// ITK Parity: MultiplyImageFilter
#[pyfunction]
pub fn multiply_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> PyResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    let result = py.allow_threads(|| {
        MultiplyImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Pixelwise division: out(x) = a(x) / b(x). Division by zero yields 0.
///
/// ITK Parity: DivideImageFilter
#[pyfunction]
pub fn divide_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> PyResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    let result = py.allow_threads(|| {
        DivideImageFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Pixelwise minimum: out(x) = min(a(x), b(x)).
///
/// ITK Parity: MinimumImageFilter
#[pyfunction]
pub fn minimum_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> PyResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    let result = py.allow_threads(|| {
        ImageMinFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Pixelwise maximum: out(x) = max(a(x), b(x)).
///
/// ITK Parity: MaximumImageFilter
#[pyfunction]
pub fn maximum_images(py: Python<'_>, a: &PyImage, b: &PyImage) -> PyResult<PyImage> {
    let a_arc = std::sync::Arc::clone(&a.inner);
    let b_arc = std::sync::Arc::clone(&b.inner);
    let result = py.allow_threads(|| {
        ImageMaxFilter::new()
            .apply(a_arc.as_ref(), b_arc.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}
