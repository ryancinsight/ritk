//! Geodesic Active Contour level set segmentation.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::edge::GaussianSigma;
use ritk_segmentation::GeodesicActiveContourSegmentation;
use std::sync::Arc;

/// Configuration options for [`geodesic_active_contour_segment`].
#[pyclass(name = "GeodesicActiveContourOptions")]
#[derive(Clone)]
pub struct PyGacOptions {
    /// Balloon force ν (expansion if > 0).
    #[pyo3(get, set)]
    pub propagation_weight: f64,
    /// Weight on curvature regularisation.
    #[pyo3(get, set)]
    pub curvature_weight: f64,
    /// Weight on ∇g·∇φ edge attraction.
    #[pyo3(get, set)]
    pub advection_weight: f64,
    /// Edge stopping sensitivity parameter k.
    #[pyo3(get, set)]
    pub edge_k: f64,
    /// Gaussian pre-smoothing σ for gradient.
    #[pyo3(get, set)]
    pub sigma: f64,
    /// Euler forward time step Δt.
    #[pyo3(get, set)]
    pub dt: f64,
    /// Maximum PDE iterations.
    #[pyo3(get, set)]
    pub max_iterations: usize,
}

#[pymethods]
impl PyGacOptions {
    #[new]
    #[pyo3(signature = (propagation_weight=1.0, curvature_weight=1.0, advection_weight=1.0, edge_k=1.0, sigma=1.0, dt=0.05, max_iterations=200))]
    pub fn new(
        propagation_weight: f64,
        curvature_weight: f64,
        advection_weight: f64,
        edge_k: f64,
        sigma: f64,
        dt: f64,
        max_iterations: usize,
    ) -> Self {
        Self {
            propagation_weight,
            curvature_weight,
            advection_weight,
            edge_k,
            sigma,
            dt,
            max_iterations,
        }
    }
}

/// Segment a 3D image via Geodesic Active Contour level set evolution.
///
/// Delegates to `ritk_segmentation::GeodesicActiveContourSegmentation`
/// (Caselles, Kimmel & Sapiro 1997). Evolves an initial level set function
/// toward image edges using the GAC PDE.
///
/// Args:
///     image: Input PyImage.
///     initial_phi: Initial level set function PyImage (same shape as image).
///         φ < 0 inside the initial contour, φ > 0 outside.
///     opts: `GeodesicActiveContourOptions` controlling PDE parameters.
///
/// Returns:
///     Binary mask PyImage (1.0 where φ < 0, 0.0 elsewhere).
///
/// Raises:
///     RuntimeError: if image and initial_phi shapes do not match.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, opts = None))]
pub fn geodesic_active_contour_segment(
    py: Python<'_>,
    image: &PyImage,
    initial_phi: &PyImage,
    opts: Option<PyGacOptions>,
) -> RitkResult<PyImage> {
    let opts = opts.unwrap_or_else(|| PyGacOptions::new(1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 200));
    let image_arc = Arc::clone(&image.inner);
    let phi_arc = Arc::clone(&initial_phi.inner);
    py.allow_threads(|| {
        let mut seg = GeodesicActiveContourSegmentation::new();
        seg.propagation_weight = opts.propagation_weight;
        seg.curvature_weight = opts.curvature_weight;
        seg.advection_weight = opts.advection_weight;
        seg.edge_k = opts.edge_k;
        seg.sigma = GaussianSigma::new_unchecked(opts.sigma);
        seg.dt = opts.dt;
        seg.max_iterations = opts.max_iterations;
        seg.apply(image_arc.as_ref(), phi_arc.as_ref())
            .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(into_py_image)
}
