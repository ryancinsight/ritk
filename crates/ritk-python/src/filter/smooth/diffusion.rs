//! Diffusion-family filters: Perona–Malik anisotropic, curvature anisotropic, and coherence-enhancing diffusion.
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::diffusion::{
    CoherenceConfig, ConductanceFunction, CurvatureConfig, DiffusionConfig,
    GradientAnisotropicDiffusionFilter, GradientDiffusionConfig,
};
use ritk_filter::edge::GaussianSigma;
use ritk_filter::{CoherenceEnhancingDiffusionFilter, CurvatureAnisotropicDiffusionFilter};

/// Conductance function kind for anisotropic diffusion, replacing `exponential: bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyConductanceKind {
    /// Quadratic conductance: c(s) = 1/(1+(s/K)²)
    Quadratic,
    /// Exponential conductance: c(s) = exp(-(s/K)²)
    Exponential,
}

impl From<PyConductanceKind> for ConductanceFunction {
    fn from(kind: PyConductanceKind) -> Self {
        match kind {
            PyConductanceKind::Quadratic => ConductanceFunction::Quadratic,
            PyConductanceKind::Exponential => ConductanceFunction::Exponential,
        }
    }
}

impl<'a> From<Option<&'a str>> for PyConductanceKind {
    fn from(s: Option<&'a str>) -> Self {
        match s.map(|v| v.to_lowercase()).as_deref() {
            Some("quadratic") | Some("q") => PyConductanceKind::Quadratic,
            _ => PyConductanceKind::Exponential,
        }
    }
}

/// Apply Perona-Malik anisotropic diffusion for edge-preserving smoothing.
///
/// Reduces noise while preserving edges via the PDE:
/// ∂I/∂t = div(c(|∇I|) · ∇I)
///
/// The default `"exponential"` kind dispatches to the ITK-exact
/// `GradientAnisotropicDiffusionImageFilter` (face-gradient conductance with the
/// per-iteration average-gradient-magnitude `K` rescaling), so it matches
/// SimpleITK's `GradientAnisotropicDiffusion`. The `"quadratic"` kind uses the
/// crate's simpler Perona-Malik conductance `1/(1+(s/K)²)` (no SimpleITK
/// equivalent).
///
/// Args:
/// image: Input PyImage.
/// iterations: Number of explicit Euler time steps (default 20).
/// conductance: Edge-stopping parameter K (default 3.0; larger = more smoothing).
/// time_step: Euler step size Δt (default 0.0625; must be ≤ 1/6 for 3-D stability).
/// conductance_kind: Conductance function kind — "exponential" (default) or "quadratic".
///
/// Returns:
/// Smoothed PyImage with identical shape and spatial metadata.
///
/// Raises:
/// RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, iterations=20, conductance=3.0, time_step=0.0625, conductance_kind="exponential"))]
pub fn anisotropic_diffusion(
    py: Python<'_>,
    image: &PyImage,
    iterations: usize,
    conductance: f64,
    time_step: f64,
    conductance_kind: Option<&str>,
) -> RitkResult<PyImage> {
    let kind = PyConductanceKind::from(conductance_kind);
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| match kind {
        // ITK-exact gradient anisotropic diffusion (matches SimpleITK).
        PyConductanceKind::Exponential => {
            GradientAnisotropicDiffusionFilter::new(GradientDiffusionConfig {
                num_iterations: iterations,
                time_step: time_step as f32,
                conductance: conductance as f32,
            })
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
        }
        // Crate-specific Perona-Malik with quadratic conductance.
        PyConductanceKind::Quadratic => DiffusionConfig {
            num_iterations: iterations,
            conductance: conductance as f32,
            time_step: time_step as f32,
            function: ConductanceFunction::Quadratic,
        }
        .apply(image.as_ref())
        .map_err(|e| RitkPyError::runtime(e.to_string())),
    })
    .map(into_py_image)
}

/// Apply curvature anisotropic diffusion (ITK `CurvatureAnisotropicDiffusion`).
///
/// Modified Curvature Diffusion Equation (Whitaker & Xue 2001):
/// ∂I/∂t = |∇I| · ∇·( c(|∇I|) · ∇I/|∇I| )
///
/// Args:
///     image: Input PyImage.
///     iterations: Number of explicit Euler time steps (default 20).
///     time_step: Euler Δt (default 0.0625; ITK's stable step is smaller for
///         fine spacing — keep ≲ 0.044 for ~0.35 mm CT to avoid instability).
///     conductance: Conductance parameter K (default 3.0). Larger K → more
///         isotropic smoothing.
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, iterations=20, time_step=0.0625, conductance=3.0))]
pub fn curvature_anisotropic_diffusion(
    py: Python<'_>,
    image: &PyImage,
    iterations: usize,
    time_step: f64,
    conductance: f64,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = CurvatureAnisotropicDiffusionFilter::new(CurvatureConfig {
            num_iterations: iterations,
            time_step: time_step as f32,
            conductance: conductance as f32,
        });
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply coherence-enhancing diffusion (Weickert 1999).
///
/// Anisotropic diffusion that smooths along coherent structures (edges,
/// ridges) while preserving them across the structure orientation. Uses the
/// structure tensor to drive diffusion.
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian sigma for structure tensor smoothing (integration scale, default 3.0).
///     contrast: Contrast parameter C (default 1e-10).
///     alpha: Smoothing parameter in flat regions (default 0.001).
///     time_step: Euler step Δt (default 0.0625).
///     iterations: Number of iterations (default 10).
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=3.0, contrast=1e-10, alpha=0.001, time_step=0.0625, iterations=10))]
pub fn coherence_enhancing_diffusion(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    contrast: f64,
    alpha: f64,
    time_step: f64,
    iterations: usize,
) -> PyImage {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let config = CoherenceConfig {
            sigma: GaussianSigma::new_unchecked(sigma),
            contrast,
            alpha,
            time_step,
            n_iterations: iterations,
        };
        let filter = CoherenceEnhancingDiffusionFilter::new(config);
        filter.apply(image.as_ref())
    });
    into_py_image(result)
}
