//! Gaussian-family smoothing filters: FIR Gaussian, discrete Gaussian, and recursive Gaussian (IIR).
use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, into_py_image, py_image_to_burn, PyImage};
use coeus_core::MoiraiBackend;
use pyo3::prelude::*;
use ritk_filter::edge::GaussianSigma;
use ritk_filter::recursive_gaussian::DerivativeOrder;
use ritk_filter::{DiscreteGaussianFilter, GaussianFilter, RecursiveGaussianFilter};
use std::sync::Arc;

/// Whether spatial filtering uses physical image spacing or voxel spacing.
#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PySpacingMode {
    Physical,
    Voxel,
}

/// Apply Gaussian smoothing to an image.
///
/// Uses `ritk_filter::GaussianFilter` (separable 1-D convolutions,
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
pub fn gaussian_filter(py: Python<'_>, image: &PyImage, sigma: f64) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = GaussianFilter::<()>::new(vec![GaussianSigma::new_unchecked(sigma); 3]);
        filter
            .apply_native(native.as_ref(), &MoiraiBackend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply ITK-style discrete Gaussian smoothing parameterized by variance.
///
/// Uses `ritk_filter::DiscreteGaussianFilter` with separable 1-D
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
) -> RitkResult<PyImage> {
    let native = Arc::clone(&image.inner);
    py.allow_threads(|| {
        let spacing_mode = match spacing_mode {
            PySpacingMode::Physical => ritk_filter::discrete_gaussian::SpacingMode::Physical,
            PySpacingMode::Voxel => ritk_filter::discrete_gaussian::SpacingMode::Voxel,
        };
        let filter = DiscreteGaussianFilter::<crate::image::BurnBackend>::new(vec![
            ritk_filter::GaussianSigma::new(variance.sqrt())
                .expect("invariant: variance must be positive (validated by caller)"),
        ])
        .with_maximum_error(maximum_error)
        .with_spacing_mode(spacing_mode);
        filter
            .apply_native(native.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply the discrete Gaussian derivative filter.
///
/// Convolves with the 1-D `GaussianDerivativeOperator` of order `order_*` along
/// each axis (ITK `DiscreteGaussianDerivativeImageFilter`). `order` is given in
/// SimpleITK `(x, y, z)` order and reversed internally to ritk's axis-major
/// layout. Float-exact to sitk for `use_image_spacing=False` (voxel units) at all
/// derivative orders. `use_image_spacing=True` (physical) folds spacing into the
/// Gaussian width and is verified only for isotropic spacing.
///
/// Returns: derivative PyImage with identical shape and spatial metadata.
#[pyfunction]
#[pyo3(signature = (image, order_x, order_y, order_z, variance, maximum_error=0.01, use_image_spacing=false))]
#[allow(clippy::too_many_arguments)]
pub fn discrete_gaussian_derivative(
    py: Python<'_>,
    image: &PyImage,
    order_x: usize,
    order_y: usize,
    order_z: usize,
    variance: f64,
    maximum_error: f64,
    use_image_spacing: bool,
) -> PyImage {
    // TODO: migrate once DiscreteGaussianDerivativeFilter gains apply_native.
    let image = py_image_to_burn(image);
    // sitk (x, y, z) → ritk axis-major (z, y, x).
    let order = [order_z, order_y, order_x];
    let result = py.allow_threads(|| {
        ritk_filter::DiscreteGaussianDerivativeFilter::new(
            variance,
            order,
            maximum_error,
            use_image_spacing,
        )
        .apply(&image)
    });
    burn_into_py_image(result)
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
    let native = Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = RecursiveGaussianFilter::new(sigma).with_derivative_order(derivative_order);
        filter
            .apply_native(native.as_ref(), &MoiraiBackend)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Apply the single-axis recursive (Deriche) Gaussian or its directional
/// derivative along one axis only (the other axes are untouched).
///
/// Float-exact to SimpleITK
/// `RecursiveGaussian(image, sigma, normalizeAcrossScale=False, order, direction)`.
/// Unlike `recursive_gaussian` (which combines all axes into a gradient magnitude
/// or Laplacian), this returns the **signed** order-`order` derivative along a
/// single `direction`.
///
/// Args:
///     image: Input PyImage.
///     sigma: Gaussian σ in physical units (mm).
///     order: Derivative order — 0 = smoothing, 1 = first, 2 = second derivative.
///     direction: Axis index in ritk `(z, y, x)` order — 0 = z, 1 = y, 2 = x.
///         (SimpleITK's `direction` is in `(x, y, z)` order, so sitk direction 0
///         (x) corresponds to `direction=2` here.)
///
/// Returns:
///     Filtered PyImage with identical shape and spatial metadata.
///
/// Raises:
///     ValueError: if `order ∉ {0,1,2}` or `direction ∉ {0,1,2}`.
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, sigma=1.0, order=0, direction=2))]
pub fn recursive_gaussian_directional(
    py: Python<'_>,
    image: &PyImage,
    sigma: f64,
    order: usize,
    direction: usize,
) -> RitkResult<PyImage> {
    let derivative_order = match order {
        0 => DerivativeOrder::Zero,
        1 => DerivativeOrder::First,
        2 => DerivativeOrder::Second,
        _ => {
            return Err(RitkPyError::value(format!(
                "recursive_gaussian_directional: order must be 0, 1, or 2, got {order}"
            )));
        }
    };
    if direction > 2 {
        return Err(RitkPyError::value(format!(
            "recursive_gaussian_directional: direction must be 0, 1, or 2, got {direction}"
        )));
    }
    let native = Arc::clone(&image.inner);
    py.allow_threads(|| {
        ritk_filter::recursive_gaussian::recursive_gaussian_directional(
            native.as_ref(),
            sigma,
            derivative_order,
            direction,
            &MoiraiBackend,
        )
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
