//! Edge detection and gradient filters: gradient magnitude, Laplacian, Canny, LoG, Sobel.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, with_tensor_slice, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    edge::GaussianSigma, CannyEdgeDetector, DerivativeImageFilter, FastMarchingFilter,
    GradientMagnitudeFilter, IsoContourDistanceFilter, LaplacianFilter, LaplacianOfGaussianFilter,
    LaplacianSharpeningFilter, SobelFilter, ZeroCrossingBasedEdgeDetectionFilter,
};

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
    // sitk direction [x,y,z] → ritk tensor axis [z,y,x].
    let axis = 2 - direction;
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        DerivativeImageFilter::new(axis, order, use_image_spacing)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

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
pub fn gradient_magnitude(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let dims = arc.shape();
    let spacing = *arc.spacing();
    with_tensor_slice(arc.data(), |vals| {
        let filter = GradientMagnitudeFilter::new(spacing);
        py.allow_threads(|| {
            filter
                .apply_from_slice(vals, dims, arc.as_ref())
                .map_err(|e| RitkPyError::runtime(e.to_string()))
        })
    })
    .map(into_py_image)
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
pub fn laplacian(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let spacing = image.spacing();
        let filter = LaplacianFilter::new(*spacing);
        filter
            .apply(image.as_ref())
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
/// divided by `spacing²`. All intermediate computation is in f64.
///
/// Args:
///     image: Input PyImage.
///     use_image_spacing: Scale the Laplacian by `1/spacing²` per axis (default True).
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
    let arc = std::sync::Arc::clone(&image.inner);
    let out =
        py.allow_threads(|| LaplacianSharpeningFilter::new(use_image_spacing).apply(arc.as_ref()));
    Ok(into_py_image(out))
}

/// Zero-crossing-based edge detection, matching
/// `SimpleITK.ZeroCrossingBasedEdgeDetection`.
///
/// Pipeline: DiscreteGaussian (isotropic `variance`, `maximum_error`) → Laplacian
/// → zero-crossing detection. Edge voxels take `foreground_value`, the rest
/// `background_value`.
///
/// Args:
///     image: Input PyImage.
///     variance: Isotropic Gaussian variance, physical units (default 1.0).
///     maximum_error: Gaussian kernel truncation error (default 0.01).
///     foreground_value: Label for edge voxels (default 1.0).
///     background_value: Label for non-edge voxels (default 0.0).
///
/// Returns:
///     Binary edge PyImage, same shape and metadata as input.
#[pyfunction]
#[pyo3(signature = (image, variance=1.0_f64, maximum_error=0.01_f64, foreground_value=1.0_f32, background_value=0.0_f32))]
pub fn zero_crossing_based_edge_detection(
    py: Python<'_>,
    image: &PyImage,
    variance: f64,
    maximum_error: f64,
    foreground_value: f32,
    background_value: f32,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        ZeroCrossingBasedEdgeDetectionFilter::new(
            variance,
            maximum_error,
            foreground_value,
            background_value,
        )
        .apply(arc.as_ref())
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Solve the Eikonal arrival-time field by fast marching, matching
/// `SimpleITK.FastMarching`.
///
/// Propagates a front from `trial_points` (seeds) outward through the `image`
/// speed field, solving ‖∇T‖·F = 1. Voxels never reached keep a large sentinel
/// value.
///
/// Args:
///     image: Speed image (non-negative).
///     trial_points: Seed voxels, each `[z, y, x]`.
///     normalization_factor: Speed normalization (default 1.0).
///     stopping_value: Stop once the smallest arrival time exceeds this (default ∞).
///     initial_trial_values: Initial arrival time per seed; empty ⇒ all 0.
///
/// Returns:
///     Arrival-time PyImage, same shape and metadata as input.
#[pyfunction]
#[pyo3(signature = (image, trial_points, normalization_factor=1.0_f64,
                    stopping_value=None, initial_trial_values=Vec::new()))]
pub fn fast_marching(
    py: Python<'_>,
    image: &PyImage,
    trial_points: Vec<[usize; 3]>,
    normalization_factor: f64,
    stopping_value: Option<f64>,
    initial_trial_values: Vec<f64>,
) -> PyImage {
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        FastMarchingFilter {
            trial_points,
            initial_trial_values,
            normalization_factor,
            stopping_value: stopping_value.unwrap_or(f64::MAX / 2.0),
        }
        .apply(arc.as_ref())
    });
    into_py_image(out)
}

/// Narrow-band signed distance to the iso-contour, matching
/// `SimpleITK.IsoContourDistance`.
///
/// Voxels straddling the `level_set_value` iso-surface get a first-order signed
/// distance estimate (averaged-gradient interpolation, combined by minimum
/// magnitude); voxels away from it keep `±far_value`.
///
/// Args:
///     image: Input PyImage (a level-set / scalar field).
///     level_set_value: Iso-contour level (default 0.0).
///     far_value: Magnitude assigned away from the contour (default 10.0).
///
/// Returns:
///     PyImage of narrow-band signed distances, same shape and metadata.
#[pyfunction]
#[pyo3(signature = (image, level_set_value=0.0_f64, far_value=10.0_f64))]
pub fn iso_contour_distance(
    py: Python<'_>,
    image: &PyImage,
    level_set_value: f64,
    far_value: f64,
) -> PyImage {
    let arc = std::sync::Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        IsoContourDistanceFilter::new(level_set_value, far_value).apply(arc.as_ref())
    });
    into_py_image(out)
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
) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = CannyEdgeDetector::new(
            GaussianSigma::new_unchecked(sigma),
            low_threshold,
            high_threshold,
        );
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
pub fn laplacian_of_gaussian(py: Python<'_>, image: &PyImage, sigma: f64) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let filter = LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(sigma));
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
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
pub fn sobel_gradient(py: Python<'_>, image: &PyImage) -> RitkResult<PyImage> {
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        let spacing = image.spacing();
        let filter = SobelFilter::new(*spacing);
        filter
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
