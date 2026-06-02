//! Vesselness filters: Frangi multiscale vesselness, Sato line filter.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::vesselness::{FrangiConfig, SatoConfig};
use ritk_core::filter::{FrangiVesselnessFilter, SatoLineFilter};

/// Apply the Frangi multiscale vesselness filter.
///
/// Detects tubular structures (blood vessels, airways) by analysing Hessian
/// eigenvalues at multiple spatial scales.  Reference: Frangi et al. (1998),
/// *MICCAI* LNCS 1496:130–137.
///
/// Args:
///     image:          Input PyImage (should be pre-smoothed for noisy data).
///     scales:         List of σ values in mm (default [0.5, 1.0, 2.0]).
///     alpha:          Plate-vs-line anisotropy parameter (default 0.5).
///     beta:           Blobness parameter (default 0.5).
///     gamma:          Noise-suppression structureness threshold (default 15.0).
///     bright_vessels: If True, detect bright tubes on dark background (default True).
///
/// Returns:
///     PyImage of vesselness values in [0, 1], same shape and metadata as input.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, scales=None, alpha=0.5, beta=0.5, gamma=15.0, bright_vessels=true))]
pub fn frangi_vesselness(
    py: Python<'_>,
    image: &PyImage,
    scales: Option<Vec<f64>>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    bright_vessels: bool,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let config = FrangiConfig {
            scales: scales.unwrap_or_else(|| vec![0.5, 1.0, 2.0]),
            alpha,
            beta,
            gamma,
            bright_vessels,
        };
        let filter = FrangiVesselnessFilter::new(config);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply the Sato multi-scale line filter for curvilinear structure detection.
///
/// Detects tubular structures using multi-scale Hessian eigenvalue analysis
/// (Sato et al. 1998). The output is the per-voxel maximum response over all scales.
///
/// Args:
///     image:       Input PyImage.
///     scales:      List of Gaussian σ values (physical units, mm). Default [1.0, 2.0, 3.0].
///     alpha:       Cross-section anisotropy exponent [0.5, 2.0]. Default 0.5.
///     bright_tubes: If True detect bright tubes on dark background (default True).
///
/// Returns:
///     Line-enhanced PyImage with identical shape and metadata.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, scales=None, alpha=0.5, bright_tubes=true))]
pub fn sato_line_filter(
    py: Python<'_>,
    image: &PyImage,
    scales: Option<Vec<f64>>,
    alpha: f64,
    bright_tubes: bool,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = SatoLineFilter::new(SatoConfig {
            scales: scales.unwrap_or_else(|| vec![1.0, 2.0, 3.0]),
            alpha,
            bright_tubes,
        });
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
