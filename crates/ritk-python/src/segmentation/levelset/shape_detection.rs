//! Shape Detection level set segmentation.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, py_image_to_burn, PyImage};
use pyo3::prelude::*;
use ritk_filter::edge::GaussianSigma;
use ritk_segmentation::ShapeDetectionSegmentation;

/// Configuration options for [`shape_detection_segment`].
#[pyclass(name = "ShapeDetectionOptions")]
#[derive(Clone)]
pub struct PyShapeDetectionOptions {
    /// Weight of curvature term.
    #[pyo3(get, set)]
    pub curvature_weight: f64,
    /// Weight of propagation term.
    #[pyo3(get, set)]
    pub propagation_weight: f64,
    /// Weight of advection term.
    #[pyo3(get, set)]
    pub advection_weight: f64,
    /// K parameter for edge potential.
    #[pyo3(get, set)]
    pub edge_k: f64,
    /// Smoothing sigma for gradient filter.
    #[pyo3(get, set)]
    pub sigma: f64,
    /// Time step.
    #[pyo3(get, set)]
    pub dt: f64,
    /// Maximum iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
    /// Convergence tolerance.
    #[pyo3(get, set)]
    pub tolerance: f64,
}

impl Default for PyShapeDetectionOptions {
    fn default() -> Self {
        Self {
            curvature_weight: 1.0,
            propagation_weight: 1.0,
            advection_weight: 1.0,
            edge_k: 1.0,
            sigma: 1.0,
            dt: 0.05,
            max_iterations: 200,
            tolerance: 1e-3,
        }
    }
}

#[pymethods]
impl PyShapeDetectionOptions {
    #[new]
    #[pyo3(signature = (
        curvature_weight = 1.0,
        propagation_weight = 1.0,
        advection_weight = 1.0,
        edge_k = 1.0,
        sigma = 1.0,
        dt = 0.05,
        max_iterations = 200,
        tolerance = 1e-3,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        curvature_weight: f64,
        propagation_weight: f64,
        advection_weight: f64,
        edge_k: f64,
        sigma: f64,
        dt: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            curvature_weight,
            propagation_weight,
            advection_weight,
            edge_k,
            sigma,
            dt,
            max_iterations,
            tolerance,
        }
    }
}

/// Shape-detection level set segmentation.
///
/// Uses a speed function that slows evolution at edges detected by a
/// gradient-magnitude filter, enabling detection of topological changes.
///
/// Args:
///     image: Input PyImage.
///     initial_phi: Initial level set function (signed distance).
///     opts: `ShapeDetectionOptions` controlling PDE parameters.
///
/// Returns:
///     Evolved level set function as PyImage.
///
/// Raises:
///     RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, opts = None))]
pub fn shape_detection_segment(
    py: Python<'_>,
    image: &PyImage,
    initial_phi: &PyImage,
    opts: Option<PyShapeDetectionOptions>,
) -> RitkResult<PyImage> {
    let opts = opts.unwrap_or_default();
    let image_arc = py_image_to_burn(image);
    let phi_arc = py_image_to_burn(initial_phi);
    py.allow_threads(|| {
        let mut seg = ShapeDetectionSegmentation::new();
        seg.curvature_weight = opts.curvature_weight;
        seg.propagation_weight = opts.propagation_weight;
        seg.advection_weight = opts.advection_weight;
        seg.edge_k = opts.edge_k;
        seg.sigma = GaussianSigma::new_unchecked(opts.sigma);
        seg.dt = opts.dt;
        seg.max_iterations = opts.max_iterations;
        seg.tolerance = opts.tolerance;
        seg.apply(&image_arc, &phi_arc).map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(burn_into_py_image)
}
