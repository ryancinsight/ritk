//! Threshold level set segmentation.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_segmentation::ThresholdLevelSet;
use std::sync::Arc;

/// Configuration options for [`threshold_level_set_segment`].
#[pyclass(name = "ThresholdLevelSetOptions")]
#[derive(Clone)]
pub struct PyThresholdLevelSetOptions {
    /// Lower intensity threshold.
    #[pyo3(get, set)]
    pub lower_threshold: f64,
    /// Upper intensity threshold.
    #[pyo3(get, set)]
    pub upper_threshold: f64,
    /// Weight of propagation term.
    #[pyo3(get, set)]
    pub propagation_weight: f64,
    /// Weight of curvature term.
    #[pyo3(get, set)]
    pub curvature_weight: f64,
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

#[pymethods]
impl PyThresholdLevelSetOptions {
    #[new]
    #[pyo3(signature = (lower_threshold, upper_threshold, propagation_weight=1.0, curvature_weight=0.2, dt=0.05, max_iterations=200, tolerance=1e-3))]
    pub fn new(
        lower_threshold: f64,
        upper_threshold: f64,
        propagation_weight: f64,
        curvature_weight: f64,
        dt: f64,
        max_iterations: usize,
        tolerance: f64,
    ) -> Self {
        Self {
            lower_threshold,
            upper_threshold,
            propagation_weight,
            curvature_weight,
            dt,
            max_iterations,
            tolerance,
        }
    }
}

/// Threshold-based level set segmentation.
///
/// Evolves a level set using a speed function derived from intensity
/// thresholds. The region between lower_threshold and upper_threshold has
/// zero speed; outside this band propagation occurs.
///
/// Args:
///     image: Input PyImage.
///     initial_phi: Initial level set function (signed distance).
///     opts: `ThresholdLevelSetOptions` controlling thresholds and PDE parameters.
///
/// Returns:
///     Evolved level set function as PyImage.
///
/// Raises:
///     RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, opts))]
pub fn threshold_level_set_segment(
    py: Python<'_>,
    image: &PyImage,
    initial_phi: &PyImage,
    opts: PyThresholdLevelSetOptions,
) -> RitkResult<PyImage> {
    let image_arc = Arc::clone(&image.inner);
    let phi_arc = Arc::clone(&initial_phi.inner);
    py.allow_threads(|| {
        let mut seg = ThresholdLevelSet::new(opts.lower_threshold, opts.upper_threshold);
        seg.propagation_weight = opts.propagation_weight;
        seg.curvature_weight = opts.curvature_weight;
        seg.dt = opts.dt;
        seg.max_iterations = opts.max_iterations;
        seg.tolerance = opts.tolerance;
        seg.apply(image_arc.as_ref(), phi_arc.as_ref())
            .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(into_py_image)
}
