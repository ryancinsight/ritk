//! Derivatives, Laplacian, and Sobel gradient filters.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, into_py_image, py_image_to_burn, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    DerivativeImageFilter, GradientMagnitudeFilter, LaplacianFilter, LaplacianSharpeningFilter,
    SobelFilter };
use std::sync::Arc;

/// Directional derivative (central differences) along `direction` (sitk axis:
/// 0 = x, 1 = y, 2 = z), of the given `order` (1 or 2). With `use_image_spacing`
/// the result is divided by `spacing^order`. ITK Parity: DerivativeImageFilter
/// (`sitk.Derivative`).
#[pyfunction]
#[pyo3(signature = (image, direction=0, order=1, use_image_spacing=true))]
pub fn derivative(
    py: Python<'_>,
    image: &PyImage,
    direction: usize,
    order: usize,
    use_image_spacing: bool,
) -> RitkResult<PyImage> {
    if direction > 2 {
        return Err(RitkPyError::value(format!(
            "derivative: direction must be 0 (x), 1 (y), or 2 (z); got {direction}"
        )));
    }
    // sitk direction [x,y,z] â†’ ritk tensor axis [z,y,x].
    let axis = 2 - direction;
    // TODO: DerivativeImageFilter still lacks apply_native; keep Burn roundtrip for now.
    let arc = py_image_to_burn(image);
    py.allow_threads(|| {
        DerivativeImageFilter::new(axis, order, use_image_spacing)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Compute the gradient magnitude |âˆ‡I| via central finite differences.
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
pub fn gradient_magnitude(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = GradientMagnitudeFilter::new(*native.spacing());
        filter
            .apply_native(native.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Compute the discrete Laplacian âˆ‡Â²I = âˆ‚Â²I/âˆ‚zÂ² + âˆ‚Â²I/âˆ‚yÂ² + âˆ‚Â²I/âˆ‚xÂ².
///
/// Uses second-order central finite differences with the image's physical
/// spacing, so the result is in (intensity / mmÂ²) units.
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
pub fn laplacian(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = LaplacianFilter::new(*native.spacing());
        filter
            .apply_native(native.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Sharpen an image by subtracting its (range-rescaled) Laplacian, matching
/// `SimpleITK.LaplacianSharpening`.
///
/// The Laplacian is rescaled into the input intensity range, subtracted from the
/// input, mean-restored, and clamped to the input range. With
/// `use_image_spacing` (default, ITK default) each axis second-derivative is
/// divided by `spacingÂ²`. All intermediate computation is in f64.
///
/// Args:
///     image: Input PyImage.
///     use_image_spacing: Scale the Laplacian by `1/spacingÂ²` per axis (default True).
///
/// Returns:
///     Sharpened PyImage, same shape and metadata as input.
#[pyfunction]
#[pyo3(signature = (image, use_image_spacing=true))]
pub fn laplacian_sharpening(
    py: Python<'_>,
    image: &PyImage,
    use_image_spacing: bool,
) -> RitkResult<PyImage> {
    // TODO: LaplacianSharpeningFilter still lacks apply_native; keep Burn roundtrip for now.
    let arc = py_image_to_burn(image);
    let out = py.allow_threads(|| LaplacianSharpeningFilter::new(use_image_spacing).apply(&arc));
    Ok(burn_into_py_image(out))
}

/// Compute the Sobel gradient magnitude of an image.
///
/// Applies the 3Ã—3Ã—3 Sobel operator along each axis, scaled by the image's
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
pub fn sobel_gradient(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    // TODO: SobelFilter still lacks apply_native; keep Burn roundtrip for now.
    let image = py_image_to_burn(image);
    py.allow_threads(|| {
        let spacing = image.spacing();
        let filter = SobelFilter::new(*spacing);
        filter
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}
