use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_from_py, into_py_image, PyImage};
use coeus_core::MoiraiBackend;
use pyo3::prelude::*;
use std::sync::Arc;

/// Per-voxel stochastic fractal dimension, matching
/// `SimpleITK.StochasticFractalDimension`.
///
/// Estimates the local fractal dimension `D = 3 − slope` from the log-log
/// scaling of mean absolute intensity differences against physical distance in a
/// `(2·radius+1)^3` neighborhood (ITK default `radius = 2`). `radius` is applied
/// to every axis. The mask input of the ITK filter is not exposed (unmasked
/// path). Physical distances honour the image spacing.
#[pyfunction]
#[pyo3(signature = (image, radius=2))]
pub fn stochastic_fractal_dimension(
    py: Python<'_>,
    image: &PyImage,
    radius: usize,
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    let backend = MoiraiBackend;
    py.allow_threads(|| {
        ritk_filter::StochasticFractalDimensionFilter::new([radius; 3])
            .apply(native.as_ref(), &backend)
            .map_err(|error| RitkPyError::runtime(error.to_string()))
    })
    .map(into_py_image)
}

/// Cubic B-spline decomposition: recover the interpolation coefficients of an
/// image (mirror boundary), matching `SimpleITK.BSplineDecomposition` at the
/// default spline order 3.
#[pyfunction]
pub fn bspline_decomposition(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let arc = image_from_py(image);
    py.allow_threads(|| {
        ritk_filter::bspline_decomposition(&arc).map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
