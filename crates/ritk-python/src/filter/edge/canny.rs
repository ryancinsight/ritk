//! Canny edge detection, Laplacian of Gaussian, and level set filters.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, py_image_to_burn, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    edge::GaussianSigma, CannyEdgeDetectionImageFilter, CannyEdgeDetector,
    CannySegmentationLevelSet, LaplacianOfGaussianFilter,
};

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
    let image = py_image_to_burn(image);
    py.allow_threads(|| {
        let filter = CannyEdgeDetector::new(
            GaussianSigma::new_unchecked(sigma),
            low_threshold,
            high_threshold,
        );
        filter
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// ITK-exact Canny edge detection, matching `SimpleITK.CannyEdgeDetection`
/// (`itk::CannyEdgeDetectionImageFilter`).
///
/// Unlike `canny_edge_detect` (a generic gradient-non-maximum-suppression Canny),
/// this is ITK's zero-crossing-of-the-second-directional-derivative formulation:
/// DiscreteGaussian smoothing → 2nd directional derivative → gradient-maximum mask
/// × magnitude → zero crossing → multiply → hysteresis thresholding. Bit-exact to
/// sitk.
///
/// Args:
///     image:           Float32 PyImage (z==1 ⇒ 2-D).
///     lower_threshold: Lower hysteresis threshold on edge strength (default 0.0).
///     upper_threshold: Upper hysteresis threshold on edge strength (default 0.0).
///     variance:        Gaussian smoothing variance σ² (default 0.0).
///     maximum_error:   Discrete-Gaussian truncation error (default 0.01).
///
/// Returns:
///     Binary edge PyImage (1.0 = edge, 0.0 = non-edge).
#[pyfunction]
#[pyo3(signature = (image, lower_threshold=0.0, upper_threshold=0.0, variance=0.0, maximum_error=0.01))]
pub fn canny_edge_detection(
    py: Python<'_>,
    image: &PyImage,
    lower_threshold: f32,
    upper_threshold: f32,
    variance: f64,
    maximum_error: f64,
) -> PyImage {
    let image = py_image_to_burn(image);
    let result = py.allow_threads(|| {
        CannyEdgeDetectionImageFilter {
            variance,
            maximum_error,
            lower_threshold,
            upper_threshold,
        }
        .apply(&image)
    });
    burn_into_py_image(result)
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
    let image = py_image_to_burn(image);
    py.allow_threads(|| {
        let filter = LaplacianOfGaussianFilter::new(GaussianSigma::new_unchecked(sigma));
        filter
            .apply(&image)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Canny-edge-guided level set segmentation, matching
/// `SimpleITK.CannySegmentationLevelSetImageFilter`.
///
/// Evolves an initial level set `φ₀` guided by Canny edges of `feature_image`.
/// Evolves the initial level set toward Canny edges of the feature image via the
/// ITK SparseField solver, bit-exact to `sitk.CannySegmentationLevelSet`.
///
/// Args:
///     initial_level_set: Initial φ image (negative inside region of interest).
///     feature_image:     Image to detect edges in.
///     threshold:         Upper hysteresis threshold of the internal Canny detector.
///     variance:          Gaussian variance of the internal Canny detector.
///     propagation_scaling:  Weight on the propagation (balloon) term (default 1.0).
///     curvature_scaling:    Weight on the curvature regularisation (default 1.0).
///     advection_scaling:    Weight on the edge-attraction advection (default 1.0).
///     maximum_rms_error:    RMS convergence criterion (default 0.02).
///     number_of_iterations: Maximum PDE steps (default 1000).
///     iso_surface_value:    Iso value of the initial level set treated as φ=0.
///
/// Returns:
///     Evolved level-set PyImage (φ < 0 inside the segmented region).
#[pyfunction]
#[pyo3(signature = (initial_level_set, feature_image, threshold=0.0, variance=0.0,
                    propagation_scaling=1.0, curvature_scaling=1.0, advection_scaling=1.0,
                    maximum_rms_error=0.02, number_of_iterations=1000, iso_surface_value=0.0))]
#[allow(clippy::too_many_arguments)]
pub fn canny_segmentation_level_set(
    py: Python<'_>,
    initial_level_set: &PyImage,
    feature_image: &PyImage,
    threshold: f32,
    variance: f32,
    propagation_scaling: f32,
    curvature_scaling: f32,
    advection_scaling: f32,
    maximum_rms_error: f32,
    number_of_iterations: usize,
    iso_surface_value: f32,
) -> RitkResult<PyImage> {
    let arc_init = py_image_to_burn(initial_level_set);
    let arc_feat = py_image_to_burn(feature_image);
    let result = py.allow_threads(|| {
        CannySegmentationLevelSet {
            canny_threshold: threshold,
            canny_variance: variance,
            number_of_iterations,
            max_rms_error: maximum_rms_error,
            propagation_scaling,
            curvature_scaling,
            advection_scaling,
            iso_surface_value,
        }
        .apply(&arc_init, &arc_feat)
    });
    result
        .map(burn_into_py_image)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
}
