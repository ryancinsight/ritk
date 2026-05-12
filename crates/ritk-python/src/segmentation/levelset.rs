//! Level set segmentation methods: Chan-Vese, Geodesic Active Contour, Shape Detection,
//! Threshold level set, and Laplacian level set.

use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::segmentation::{
    ChanVeseSegmentation, GeodesicActiveContourSegmentation, LaplacianLevelSet,
    ShapeDetectionSegmentation, ThresholdLevelSet,
};
use std::sync::Arc;

/// Segment a 3D image via Chan-Vese level set evolution.
///
/// Delegates to `ritk_core::segmentation::ChanVeseSegmentation` (Active
/// Contours Without Edges, Chan & Vese 2001). Evolves a level set function
/// under an energy functional driven by region statistics (no edges required).
///
/// Args:
///     image:          Input PyImage.
///     mu:             Curvature (length) penalty weight. Default 0.25.
///     nu:             Area penalty weight. Default 0.0.
///     lambda1:        Data fidelity weight for inside region. Default 1.0.
///     lambda2:        Data fidelity weight for outside region. Default 1.0.
///     max_iterations: Maximum PDE evolution iterations. Default 200.
///     dt:             Euler forward time step. Default 0.1.
///     tolerance:      Convergence tolerance on max|Δφ|/dt. Default 1e-3.
///
/// Returns:
///     Binary mask PyImage (1.0 = inside, 0.0 = outside).
#[pyfunction]
#[pyo3(signature = (image, mu=0.25, nu=0.0, lambda1=1.0, lambda2=1.0, max_iterations=200, dt=0.1, tolerance=1e-3))]
pub fn chan_vese_segment(
    py: Python<'_>,
    image: &PyImage,
    mu: f64,
    nu: f64,
    lambda1: f64,
    lambda2: f64,
    max_iterations: usize,
    dt: f64,
    tolerance: f64,
) -> PyResult<PyImage> {
    let image_arc = Arc::clone(&image.inner);
    let result = py
        .allow_threads(|| {
            let mut seg = ChanVeseSegmentation::new();
            seg.mu = mu;
            seg.nu = nu;
            seg.lambda1 = lambda1;
            seg.lambda2 = lambda2;
            seg.max_iterations = max_iterations;
            seg.dt = dt;
            seg.tolerance = tolerance;
            seg.apply(image_arc.as_ref()).map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(into_py_image(result))
}

/// Segment a 3D image via Geodesic Active Contour level set evolution.
///
/// Delegates to `ritk_core::segmentation::GeodesicActiveContourSegmentation`
/// (Caselles, Kimmel & Sapiro 1997). Evolves an initial level set function
/// toward image edges using the GAC PDE.
///
/// Args:
///     image:              Input PyImage.
///     initial_phi:        Initial level set function PyImage (same shape as image).
///                         φ < 0 inside the initial contour, φ > 0 outside.
///     propagation_weight: Balloon force ν (expansion if > 0). Default 1.0.
///     curvature_weight:   Weight on curvature regularisation. Default 1.0.
///     advection_weight:   Weight on ∇g·∇φ edge attraction. Default 1.0.
///     edge_k:             Edge stopping sensitivity parameter k. Default 1.0.
///     sigma:              Gaussian pre-smoothing σ for gradient. Default 1.0.
///     dt:                 Euler forward time step Δt. Default 0.05.
///     max_iterations:     Maximum PDE iterations. Default 200.
///
/// Returns:
///     Binary mask PyImage (1.0 where φ < 0, 0.0 elsewhere).
///
/// Raises:
///     RuntimeError: if image and initial_phi shapes do not match.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, propagation_weight=1.0, curvature_weight=1.0, advection_weight=1.0, edge_k=1.0, sigma=1.0, dt=0.05, max_iterations=200))]
pub fn geodesic_active_contour_segment(
    py: Python<'_>,
    image: &PyImage,
    initial_phi: &PyImage,
    propagation_weight: f64,
    curvature_weight: f64,
    advection_weight: f64,
    edge_k: f64,
    sigma: f64,
    dt: f64,
    max_iterations: usize,
) -> PyResult<PyImage> {
    let image_arc = Arc::clone(&image.inner);
    let phi_arc = Arc::clone(&initial_phi.inner);
    let result = py
        .allow_threads(|| {
            let mut seg = GeodesicActiveContourSegmentation::new();
            seg.propagation_weight = propagation_weight;
            seg.curvature_weight = curvature_weight;
            seg.advection_weight = advection_weight;
            seg.edge_k = edge_k;
            seg.sigma = sigma;
            seg.dt = dt;
            seg.max_iterations = max_iterations;
            seg.apply(image_arc.as_ref(), phi_arc.as_ref())
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(into_py_image(result))
}

/// Shape-detection level set segmentation.
///
/// Uses a speed function that slows evolution at edges detected by a
/// gradient-magnitude filter, enabling detection of topological changes.
///
/// Args:
///     image:              Input PyImage.
///     initial_phi:        Initial level set function (signed distance).
///     curvature_weight:   Weight of curvature term (default 1.0).
///     propagation_weight: Weight of propagation term (default 1.0).
///     advection_weight:   Weight of advection term (default 1.0).
///     edge_k:             K parameter for edge potential (default 1.0).
///     sigma:              Smoothing sigma for gradient filter (default 1.0).
///     dt:                 Time step (default 0.05).
///     max_iterations:     Maximum iterations (default 200).
///     tolerance:          Convergence tolerance (default 1e-3).
///
/// Returns:
///     Evolved level set function as PyImage.
///
/// Raises:
///     RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, curvature_weight=1.0, propagation_weight=1.0, advection_weight=1.0, edge_k=1.0, sigma=1.0, dt=0.05, max_iterations=200, tolerance=1e-3))]
pub fn shape_detection_segment(
    py: Python<'_>,
    image: &PyImage,
    initial_phi: &PyImage,
    curvature_weight: f64,
    propagation_weight: f64,
    advection_weight: f64,
    edge_k: f64,
    sigma: f64,
    dt: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PyImage> {
    let image_arc = Arc::clone(&image.inner);
    let phi_arc = Arc::clone(&initial_phi.inner);
    let result = py
        .allow_threads(|| {
            let mut seg = ShapeDetectionSegmentation::new();
            seg.curvature_weight = curvature_weight;
            seg.propagation_weight = propagation_weight;
            seg.advection_weight = advection_weight;
            seg.edge_k = edge_k;
            seg.sigma = sigma;
            seg.dt = dt;
            seg.max_iterations = max_iterations;
            seg.tolerance = tolerance;
            seg.apply(image_arc.as_ref(), phi_arc.as_ref())
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(into_py_image(result))
}

/// Threshold-based level set segmentation.
///
/// Evolves a level set using a speed function derived from intensity
/// thresholds. The region between lower_threshold and upper_threshold has
/// zero speed; outside this band propagation occurs.
///
/// Args:
///     image:              Input PyImage.
///     initial_phi:        Initial level set function (signed distance).
///     lower_threshold:    Lower intensity threshold.
///     upper_threshold:    Upper intensity threshold.
///     propagation_weight: Weight of propagation term (default 1.0).
///     curvature_weight:   Weight of curvature term (default 0.2).
///     dt:                 Time step (default 0.05).
///     max_iterations:     Maximum iterations (default 200).
///     tolerance:          Convergence tolerance (default 1e-3).
///
/// Returns:
///     Evolved level set function as PyImage.
///
/// Raises:
///     RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, lower_threshold, upper_threshold, propagation_weight=1.0, curvature_weight=0.2, dt=0.05, max_iterations=200, tolerance=1e-3))]
pub fn threshold_level_set_segment(
    py: Python<'_>,
    image: &PyImage,
    initial_phi: &PyImage,
    lower_threshold: f64,
    upper_threshold: f64,
    propagation_weight: f64,
    curvature_weight: f64,
    dt: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PyImage> {
    let image_arc = Arc::clone(&image.inner);
    let phi_arc = Arc::clone(&initial_phi.inner);
    let result = py
        .allow_threads(|| {
            let mut seg = ThresholdLevelSet::new(lower_threshold, upper_threshold);
            seg.propagation_weight = propagation_weight;
            seg.curvature_weight = curvature_weight;
            seg.dt = dt;
            seg.max_iterations = max_iterations;
            seg.tolerance = tolerance;
            seg.apply(image_arc.as_ref(), phi_arc.as_ref())
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(into_py_image(result))
}

/// Laplacian level set segmentation.
///
/// Evolves a level set using a speed function derived from the image Laplacian.
/// Positive propagation speed is applied where L(I) < 0 (local bright maxima).
///
/// Args:
///     image:              Input PyImage.
///     initial_phi:        Initial level set function (signed distance).
///     propagation_weight: Weight of Laplacian propagation term (default 1.0).
///     curvature_weight:   Weight of curvature regularisation term (default 0.2).
///     sigma:              Gaussian pre-smoothing standard deviation (default 1.0).
///     dt:                 Euler time step (default 0.05).
///     max_iterations:     Maximum PDE iterations (default 200).
///     tolerance:          Convergence tolerance on max|delta phi|/dt (default 1e-3).
///
/// Returns:
///     Binary mask PyImage (1.0=foreground, 0.0=background).
///
/// Raises:
///     RuntimeError: if computation fails.
#[pyfunction]
#[pyo3(signature = (image, initial_phi, propagation_weight=1.0, curvature_weight=0.2, sigma=1.0, dt=0.05, max_iterations=200, tolerance=1e-3))]
pub fn laplacian_level_set_segment(
    py: Python<'_>,
    image: &PyImage,
    initial_phi: &PyImage,
    propagation_weight: f64,
    curvature_weight: f64,
    sigma: f64,
    dt: f64,
    max_iterations: usize,
    tolerance: f64,
) -> PyResult<PyImage> {
    let image_arc = Arc::clone(&image.inner);
    let phi_arc = Arc::clone(&initial_phi.inner);
    let result = py
        .allow_threads(|| {
            let mut seg = LaplacianLevelSet::new();
            seg.propagation_weight = propagation_weight;
            seg.curvature_weight = curvature_weight;
            seg.sigma = sigma;
            seg.dt = dt;
            seg.max_iterations = max_iterations;
            seg.tolerance = tolerance;
            seg.apply(image_arc.as_ref(), phi_arc.as_ref())
                .map_err(|e| e.to_string())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    Ok(into_py_image(result))
}
