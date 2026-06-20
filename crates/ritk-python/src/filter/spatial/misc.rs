use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;

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
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        ritk_filter::StochasticFractalDimensionFilter::new([radius, radius, radius])
            .apply(arc.as_ref())
    });
    Ok(into_py_image(out))
}

/// Cubic B-spline decomposition: recover the interpolation coefficients of an
/// image (mirror boundary), matching `SimpleITK.BSplineDecomposition` at the
/// default spline order 3.
#[pyfunction]
pub fn bspline_decomposition(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        ritk_filter::bspline_decomposition(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
