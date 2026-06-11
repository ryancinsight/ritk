//! Chan-Vese level set segmentation.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_segmentation::ChanVeseSegmentation;
use std::sync::Arc;

/// Configuration options for [`chan_vese_segment`].
#[pyclass(name = "ChanVeseOptions")]
#[derive(Clone)]
pub struct PyChanVeseOptions {
    /// Curvature (length) penalty weight.
    #[pyo3(get, set)]
    pub mu: f64,
    /// Area penalty weight.
    #[pyo3(get, set)]
    pub nu: f64,
    /// Data fidelity weight for inside region.
    #[pyo3(get, set)]
    pub lambda1: f64,
    /// Data fidelity weight for outside region.
    #[pyo3(get, set)]
    pub lambda2: f64,
    /// Maximum PDE evolution iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Euler forward time step.
    #[pyo3(get, set)]
    pub dt: f64,
    /// Convergence tolerance on max|Δφ|/dt.
    #[pyo3(get, set)]
    pub tolerance: f64,
}

#[pymethods]
impl PyChanVeseOptions {
    #[new]
    #[pyo3(signature = (mu=0.25, nu=0.0, lambda1=1.0, lambda2=1.0, max_iterations=200, dt=0.1, tolerance=1e-3))]
    pub fn new(
        mu: f64,
        nu: f64,
        lambda1: f64,
        lambda2: f64,
        max_iterations: usize,
        dt: f64,
        tolerance: f64,
    ) -> Self {
        Self {
            mu,
            nu,
            lambda1,
            lambda2,
            max_iterations,
            dt,
            tolerance,
        }
    }
}

/// Segment a 3D image via Chan-Vese level set evolution.
///
/// Delegates to `ritk_segmentation::ChanVeseSegmentation` (Active
/// Contours Without Edges, Chan & Vese 2001). Evolves a level set function
/// under an energy functional driven by region statistics (no edges required).
///
/// Args:
///     image: Input PyImage.
///     opts: [`ChanVeseOptions`] controlling PDE parameters and stopping criteria.
///
/// Returns:
///     Binary mask PyImage (1.0 = inside, 0.0 = outside).
#[pyfunction]
#[pyo3(signature = (image, opts = None))]
pub fn chan_vese_segment(
    py: Python<'_>,
    image: &PyImage,
    opts: Option<PyChanVeseOptions>,
) -> RitkResult<PyImage> {
    let opts = opts.unwrap_or_else(|| PyChanVeseOptions::new(0.25, 0.0, 1.0, 1.0, 200, 0.1, 1e-3));
    let image_arc = Arc::clone(&image.inner);
    py.allow_threads(|| {
        let mut seg = ChanVeseSegmentation::new();
        seg.mu = opts.mu;
        seg.nu = opts.nu;
        seg.lambda1 = opts.lambda1;
        seg.lambda2 = opts.lambda2;
        seg.max_iterations = opts.max_iterations;
        seg.dt = opts.dt;
        seg.tolerance = opts.tolerance;
        seg.apply(image_arc.as_ref()).map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(into_py_image)
}
