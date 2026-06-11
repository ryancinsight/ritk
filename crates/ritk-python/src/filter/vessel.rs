//! Vesselness filters: Frangi multiscale vesselness, Sato line filter.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::vesselness::{FrangiConfig, SatoConfig, VesselPolarity};
use ritk_core::filter::{FrangiVesselnessFilter, SatoLineFilter};

/// Vessel/brightness polarity for vesselness filters, replacing `bright_vessels: bool`.
///
/// Eliminates boolean blindness: `polarity="bright"` vs `polarity="dark"` is
/// self-documenting compared to `bright_vessels=True/False`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyVesselPolarity {
    /// Detect bright structures on dark background.
    Bright,
    /// Detect dark structures on bright background.
    Dark,
}

impl<'py> FromPyObject<'py> for PyVesselPolarity {
    fn extract_bound(ob: &pyo3::Bound<'py, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        match s.to_lowercase().as_str() {
            "bright" => Ok(Self::Bright),
            "dark" => Ok(Self::Dark),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown vessel polarity '{}'. Choices: bright, dark",
                other
            ))),
        }
    }
}

impl From<PyVesselPolarity> for VesselPolarity {
    fn from(val: PyVesselPolarity) -> Self {
        match val {
            PyVesselPolarity::Bright => VesselPolarity::Bright,
            PyVesselPolarity::Dark => VesselPolarity::Dark,
        }
    }
}

/// Apply the Frangi multiscale vesselness filter.
///
/// Detects tubular structures (blood vessels, airways) by analysing Hessian
/// eigenvalues at multiple spatial scales. Reference: Frangi et al. (1998),
/// *MICCAI* LNCS 1496:130–137.
///
/// Args:
/// image: Input PyImage (should be pre-smoothed for noisy data).
/// scales: List of σ values in mm (default [0.5, 1.0, 2.0]).
/// alpha: Plate-vs-line anisotropy parameter (default 0.5).
/// beta: Blobness parameter (default 0.5).
/// gamma: Noise-suppression structureness threshold (default 15.0).
/// polarity: Vessel polarity: "bright" (default) or "dark".
///     "bright" detects bright tubes on dark background.
///     "dark" detects dark tubes on bright background.
///
/// Returns:
/// PyImage of vesselness values in [0, 1], same shape and metadata as input.
///
/// Raises:
/// RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, scales=None, alpha=0.5, beta=0.5, gamma=15.0, polarity=PyVesselPolarity::Bright))]
pub fn frangi_vesselness(
    py: Python<'_>,
    image: &PyImage,
    scales: Option<Vec<f64>>,
    alpha: f64,
    beta: f64,
    gamma: f64,
    polarity: PyVesselPolarity,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let config = FrangiConfig {
            scales: scales.unwrap_or_else(|| vec![0.5, 1.0, 2.0]),
            alpha,
            beta,
            gamma,
            polarity: VesselPolarity::from(polarity),
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
/// image: Input PyImage.
/// scales: List of Gaussian σ values (physical units, mm). Default [1.0, 2.0, 3.0].
/// alpha: Cross-section anisotropy exponent [0.5, 2.0]. Default 0.5.
/// polarity: Vessel polarity: "bright" (default) or "dark".
///     "bright" detects bright tubes on dark background.
///     "dark" detects dark tubes on bright background.
///
/// Returns:
/// Line-enhanced PyImage with identical shape and metadata.
///
/// Raises:
/// RuntimeError: on tensor extraction failure.
#[pyfunction]
#[pyo3(signature = (image, scales=None, alpha=0.5, polarity=PyVesselPolarity::Bright))]
pub fn sato_line_filter(
    py: Python<'_>,
    image: &PyImage,
    scales: Option<Vec<f64>>,
    alpha: f64,
    polarity: PyVesselPolarity,
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = SatoLineFilter::new(SatoConfig {
            scales: scales.unwrap_or_else(|| vec![1.0, 2.0, 3.0]),
            alpha,
            polarity: VesselPolarity::from(polarity),
        });
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
