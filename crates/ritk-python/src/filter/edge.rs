//! Edge detection and gradient filters: gradient magnitude, Laplacian, Canny, LoG, Sobel.

use crate::image::{into_py_image, with_tensor_slice, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::{
    CannyEdgeDetector, GradientMagnitudeFilter, LaplacianFilter, LaplacianOfGaussianFilter,
    SobelFilter,
};

/// Compute the gradient magnitude |∇I| via central finite differences.
///
/// Each gradient component is divided by the corresponding physical spacing so
/// the result is in (intensity / mm) units.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     PyImage of gradient magnitudes, same shape and metadata as input.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
pub fn gradient_magnitude(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let dims = arc.shape();
    let spacing = arc.spacing().clone();
    let result = with_tensor_slice(arc.data(), |vals| {
        let filter = GradientMagnitudeFilter::new([spacing[0], spacing[1], spacing[2]]);
        py.allow_threads(|| {
            filter
                .apply_from_slice(vals, dims, arc.as_ref())
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
        })
    })?;
    Ok(into_py_image(result))
}

/// Compute the discrete Laplacian ∇²I = ∂²I/∂z² + ∂²I/∂y² + ∂²I/∂x².
///
/// Uses second-order central finite differences with the image's physical
/// spacing, so the result is in (intensity / mm²) units.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     PyImage of Laplacian values, same shape and metadata as input.
///
/// Raises:
///     RuntimeError: on tensor extraction failure.
#[pyfunction]
pub fn laplacian(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let spacing = image.spacing();
        let filter = LaplacianFilter::new([spacing[0], spacing[1], spacing[2]]);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Apply the Canny edge detector to an image.
///
/// Pipeline: Gaussian smoothing (σ) → gradient magnitude → non-maximum
/// suppression → double-threshold hysteresis.  Reference: Canny, J. (1986),
/// *IEEE Trans. PAMI* 8(6):679–698.
///
/// Args:
///     image:          Input PyImage.
///     sigma:          Gaussian pre-smoothing σ in physical units (mm, default 1.0).
///     low_threshold:  Lower hysteresis threshold on gradient magnitude (default 0.1).
///     high_threshold: Upper hysteresis threshold on gradient magnitude (default 0.2).
///
/// Returns:
///     Binary edge PyImage (1.0 = edge, 0.0 = non-edge), same shape and metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0, low_threshold=0.1, high_threshold=0.2))]
pub fn canny_edge_detect(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    low_threshold: f64,
    high_threshold: f64,
) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = CannyEdgeDetector::new(sigma, low_threshold, high_threshold);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Apply the Laplacian of Gaussian (LoG) filter.
///
/// Computes ∇²(G_σ * I) by first applying separable Gaussian smoothing with
/// standard deviation σ, then computing the discrete Laplacian via
/// second-order finite differences.  Useful for blob detection and
/// zero-crossing edge detection (Marr & Hildreth 1980).
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian σ in physical units (mm, default 1.0).
///
/// Returns:
///     PyImage of LoG values, same shape and metadata as input.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0))]
pub fn laplacian_of_gaussian(py: Python<'_>, image: &PyImage, sigma: f64) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = LaplacianOfGaussianFilter::new(sigma);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}

/// Compute the Sobel gradient magnitude of an image.
///
/// Applies the 3×3×3 Sobel operator along each axis, scaled by the image's
/// physical spacing, then returns the Euclidean magnitude of the gradient
/// vector.
///
/// Args:
///     image: Input PyImage.
///
/// Returns:
///     PyImage of gradient magnitudes in (intensity / mm) units, same shape
///     and metadata.
///
/// Raises:
///     RuntimeError: on internal computation failure.
#[pyfunction]
pub fn sobel_gradient(py: Python<'_>, image: &PyImage) -> PyResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let spacing = image.spacing();
        let filter = SobelFilter::new([spacing[0], spacing[1], spacing[2]]);
        filter
            .apply(image.as_ref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    })?;
    Ok(into_py_image(result))
}
