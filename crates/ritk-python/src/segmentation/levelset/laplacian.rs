//! Laplacian level set segmentation.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_from_py, into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::edge::GaussianSigma;
use ritk_segmentation::LaplacianLevelSet;

/// Configuration options for [`laplacian_level_set_segment`].
#[pyclass(name = "LaplacianLevelSetOptions")]
#[derive(Clone)]
pub struct PyLaplacianLevelSetOptions {
    /// Weight of Laplacian propagation term.
    #[pyo3(get, set)]
    pub propagation_weight: f64,
    /// Weight of curvature regularisation term.
    #[pyo3(get, set)]
    pub curvature_weight: f64,
    /// Gaussian pre-smoothing standard deviation.
    #[pyo3(get, set)]
    pub sigma: f64,
    /// Euler time step.
    #[pyo3(get, set)]
    pub dt: f64,
    /// Maximum PDE iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Convergence tolerance on max|delta phi|/dt.
    #[pyo3(get, set)]
    pub tolerance: f64,
}

#[pymethods]
impl PyLaplacianLevelSetOptions {
    #[new]
    #[pyo3(signature = (propagation_weight=1.0, curvature_weight=0.2, sigma=1.0, dt=0.05, max_iterations=200, tolerance=1e-3))]
    pub fn new(
        propagation_weight: f64,
        curvature_weight: f64,
        sigma: f64,
        dt: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            propagation_weight,
            curvature_weight,
            sigma,
            dt,
            max_iterations,
            tolerance,
        }
    }
}

/// Laplacian level set segmentation.
///
/// Evolves a level set using a speed function derived from the image Laplacian.
/// Positive propagation speed is applied where L(I) < 0 (local bright maxima).
///
/// Args:
///     image: Input PyImage.
///     initial_phi: Initial level set function (signed distance).
///     opts: `LaplacianLevelSetOptions` controlling PDE parameters.
///
/// Returns:
///     Binary mask PyImage (1.0=foreground, 0.0=background).
///
/// Raises:
///     RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, opts = None))]
pub fn laplacian_level_set_segment(
    py: Python<'_>,
    image: &PyImage,
    initial_phi: &PyImage,
    opts: Option<PyLaplacianLevelSetOptions>,
) -> RitkResult<PyImage> {
    let opts =
        opts.unwrap_or_else(|| PyLaplacianLevelSetOptions::new(1.0, 0.2, 1.0, 0.05, 200, 1e-3));
    let image_arc = image_from_py(image);
    let phi_arc = image_from_py(initial_phi);
    py.allow_threads(|| {
        let mut seg = LaplacianLevelSet::new();
        seg.propagation_weight = opts.propagation_weight;
        seg.curvature_weight = opts.curvature_weight;
        seg.sigma = GaussianSigma::new_unchecked(opts.sigma);
        seg.dt = opts.dt;
        seg.max_iterations = opts.max_iterations;
        seg.tolerance = opts.tolerance;
        seg.apply(&image_arc, &phi_arc).map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(into_py_image)
}
