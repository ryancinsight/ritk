//! Level set segmentation methods: Chan-Vese, Geodesic Active Contour, Shape Detection,
//! Threshold level set, and Laplacian level set.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::segmentation::{
    ChanVeseSegmentation, GeodesicActiveContourSegmentation, LaplacianLevelSet,
    ShapeDetectionSegmentation, ThresholdLevelSet,
};
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
/// Delegates to `ritk_core::segmentation::ChanVeseSegmentation` (Active
/// Contours Without Edges, Chan & Vese 2001). Evolves a level set function
/// under an energy functional driven by region statistics (no edges required).
///
/// Args:
///     image: Input PyImage.
///     opts:  [`ChanVeseOptions`] controlling PDE parameters and stopping criteria.
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
/// Delegates to `ritk_core::segmentation::GeodesicActiveContourSegmentation`
/// (Caselles, Kimmel & Sapiro 1997). Evolves an initial level set function
/// toward image edges using the GAC PDE.
///
/// Args:
///     image:       Input PyImage.
///     initial_phi: Initial level set function PyImage (same shape as image).
///                  φ < 0 inside the initial contour, φ > 0 outside.
///     opts:        [`GeodesicActiveContourOptions`] controlling PDE parameters.
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
        seg.sigma = opts.sigma;
        seg.dt = opts.dt;
        seg.max_iterations = opts.max_iterations;
        seg.apply(image_arc.as_ref(), phi_arc.as_ref())
            .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(into_py_image)
}

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
    pub fn new() -> Self {
        Self::default()
    }
}

/// Shape-detection level set segmentation.
///
/// Uses a speed function that slows evolution at edges detected by a
/// gradient-magnitude filter, enabling detection of topological changes.
///
/// Args:
///     image:       Input PyImage.
///     initial_phi: Initial level set function (signed distance).
///     opts:        [`ShapeDetectionOptions`] controlling PDE parameters.
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
    let image_arc = Arc::clone(&image.inner);
    let phi_arc = Arc::clone(&initial_phi.inner);
    py.allow_threads(|| {
        let mut seg = ShapeDetectionSegmentation::new();
        seg.curvature_weight = opts.curvature_weight;
        seg.propagation_weight = opts.propagation_weight;
        seg.advection_weight = opts.advection_weight;
        seg.edge_k = opts.edge_k;
        seg.sigma = opts.sigma;
        seg.dt = opts.dt;
        seg.max_iterations = opts.max_iterations;
        seg.tolerance = opts.tolerance;
        seg.apply(image_arc.as_ref(), phi_arc.as_ref())
            .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(into_py_image)
}

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
///     image:       Input PyImage.
///     initial_phi: Initial level set function (signed distance).
///     opts:        [`ThresholdLevelSetOptions`] controlling thresholds and PDE parameters.
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
///     image:       Input PyImage.
///     initial_phi: Initial level set function (signed distance).
///     opts:        [`LaplacianLevelSetOptions`] controlling PDE parameters.
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
    let image_arc = Arc::clone(&image.inner);
    let phi_arc = Arc::clone(&initial_phi.inner);
    py.allow_threads(|| {
        let mut seg = LaplacianLevelSet::new();
        seg.propagation_weight = opts.propagation_weight;
        seg.curvature_weight = opts.curvature_weight;
        seg.sigma = opts.sigma;
        seg.dt = opts.dt;
        seg.max_iterations = opts.max_iterations;
        seg.tolerance = opts.tolerance;
        seg.apply(image_arc.as_ref(), phi_arc.as_ref())
            .map_err(|e| e.to_string())
    })
    .map_err(RitkPyError::runtime)
    .map(into_py_image)
}
