//! Gaussian-family smoothing filters: FIR Gaussian, discrete Gaussian, and recursive Gaussian (IIR).
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, Backend, PyImage};
use pyo3::prelude::*;
use ritk_core::filter::edge::GaussianSigma;
use ritk_core::filter::recursive_gaussian::DerivativeOrder;
use ritk_core::filter::{DiscreteGaussianFilter, GaussianFilter, RecursiveGaussianFilter};

/// Whether spatial filtering uses physical image spacing or voxel spacing.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PySpacingMode {
    Physical,
    Voxel,
}

/// Apply Gaussian smoothing to an image.
///
/// Uses `ritk_core::filter::GaussianFilter` (separable 1-D convolutions,
/// NdArray backend, CPU-only). The same sigma is applied along all three
/// axes (isotropic smoothing).
///
/// Args:
///     image: Input PyImage.
///     sigma: Standard deviation in physical units (mm).
///         The pixel-space sigma is computed as `sigma / spacing[d]`.
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, sigma))]
pub fn gaussian_filter(py: Python<'_>, image: &PyImage, sigma: f64) -> PyImage {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let filter = GaussianFilter::<Backend>::new(vec![GaussianSigma::new_unchecked(sigma); 3]);
        filter.apply(image.as_ref())
    });
    into_py_image(result)
}

/// Apply ITK-style discrete Gaussian smoothing parameterized by variance.
///
/// Uses `ritk_core::filter::DiscreteGaussianFilter` with separable 1-D
/// convolution, replicate padding, analytic kernel truncation from
/// `maximum_error`, and optional spacing-aware sigma conversion.
///
/// Args:
///     image: Input PyImage.
///     variance: Gaussian variance σ² in physical units².
///     maximum_error: Kernel truncation tolerance in (0, 1) (default 0.01).
///     spacing_mode: Whether to convert physical σ to pixel σ using image
///         spacing. [`PySpacingMode::Physical`] (default) applies the
///         per-axis spacing divisor;
///         [`PySpacingMode::Voxel`] treats σ as already in voxel units.
///
/// Returns:
///     Smoothed PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, variance, maximum_error=0.01, spacing_mode=PySpacingMode::Physical))]
pub fn discrete_gaussian(
    py: Python<'_>,
    image: &PyImage,
    variance: f64,
    maximum_error: f64,
    spacing_mode: PySpacingMode,
) -> PyImage {
    let image = std::sync::Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        let spacing_mode = match spacing_mode {
            PySpacingMode::Physical => ritk_core::filter::discrete_gaussian::SpacingMode::Physical,
            PySpacingMode::Voxel => ritk_core::filter::discrete_gaussian::SpacingMode::Voxel,
        };
        let filter = DiscreteGaussianFilter::<Backend>::new(vec![variance])
            .with_maximum_error(maximum_error)
            .with_spacing_mode(spacing_mode);
        filter.apply(image.as_ref())
    });
    into_py_image(result)
}

/// Apply a recursive Gaussian (Young–van Vliet 3rd-order IIR) filter.
///
/// Separable IIR approximation of the Gaussian and its first/second
/// derivatives. Constant-time per voxel regardless of σ (no explicit kernel
/// construction).
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian σ in physical units (mm, default 1.0).
///     order: Derivative order — 0 = smoothing, 1 = first derivative,
///         2 = second derivative (default 0).
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     ValueError: if `order` is not in {0, 1, 2}.
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0, order=0))]
pub fn recursive_gaussian(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    order: usize,
) -> RitkResult<PyImage> {
    let derivative_order = match order {
        0 => DerivativeOrder::Zero,
        1 => DerivativeOrder::First,
        2 => DerivativeOrder::Second,
        _ => {
            return Err(RitkPyError::value(format!(
                "recursive_gaussian: order must be 0, 1, or 2, got {order}"
            )));
        }
    };
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = RecursiveGaussianFilter::new(sigma).with_derivative_order(derivative_order);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
