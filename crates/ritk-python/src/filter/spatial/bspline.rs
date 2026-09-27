use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use std::sync::Arc;

/// Cubic B-spline decomposition: recover the interpolation coefficients of an
/// image (mirror boundary), matching `SimpleITK.BSplineDecomposition` at the
/// default spline order 3.
#[pyfunction]
pub fn bspline_decomposition(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let arc = Arc::clone(&image.inner);
    py.detach(|| {
        ritk_filter::bspline_decomposition(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
