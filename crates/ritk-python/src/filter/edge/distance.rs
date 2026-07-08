//! Distance transform, level set reinitialisation, and zero-crossing filters.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{burn_into_py_image, py_image_to_burn, PyImage};
use pyo3::prelude::*;
use ritk_filter::{
    ApproximateSignedDistanceMapFilter, IsoContourDistanceFilter, ReinitializeLevelSetFilter,
    ZeroCrossingBasedEdgeDetectionFilter,
};

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
    let arc = py_image_to_burn(image);
    py.allow_threads(|| {
        ZeroCrossingBasedEdgeDetectionFilter::new(
            variance,
            maximum_error,
            foreground_value,
            background_value,
        )
        .apply(&arc)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
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
    let arc = py_image_to_burn(image);
    let out =
        py.allow_threads(|| IsoContourDistanceFilter::new(level_set_value, far_value).apply(&arc));
    burn_into_py_image(out)
}

/// Approximate signed distance map of a binary/label image, matching
/// `SimpleITK.ApproximateSignedDistanceMap` (inside negative, outside positive).
///
/// Composes an iso-contour distance at level `(inside+outside)/2` with a fast
/// chamfer propagation. `inside_value`/`outside_value` select the foreground.
///
/// Args:
///     image:         Input binary/label PyImage.
///     inside_value:  Pixel value of the inside region (default 1.0).
///     outside_value: Pixel value of the outside region (default 0.0).
///
/// Returns:
///     Signed distance PyImage, same shape and metadata as input.
#[pyfunction]
#[pyo3(signature = (image, inside_value=1.0_f64, outside_value=0.0_f64))]
pub fn approximate_signed_distance_map(
    py: Python<'_>,
    image: &PyImage,
    inside_value: f64,
    outside_value: f64,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    py.allow_threads(|| {
        ApproximateSignedDistanceMapFilter {
            inside_value,
            outside_value,
        }
        .apply(&arc)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}

/// Reinitialize a level-set image to a signed distance function, matching
/// `SimpleITK.ReinitializeLevelSet` (full-band).
///
/// Locates the zero level set and fast-marches unit speed outward/inward, giving
/// `+distance` outside and `-distance` inside.
///
/// Args:
///     image:           Input level-set PyImage.
///     level_set_value: Iso-value of the zero level set (default 0.0).
///
/// Returns:
///     Signed-distance PyImage, same shape and metadata as input.
#[pyfunction]
#[pyo3(signature = (image, level_set_value=0.0_f64))]
pub fn reinitialize_level_set(
    py: Python<'_>,
    image: &PyImage,
    level_set_value: f64,
) -> RitkResult<PyImage> {
    let arc = py_image_to_burn(image);
    py.allow_threads(|| {
        ReinitializeLevelSetFilter::new(level_set_value)
            .apply(&arc)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(burn_into_py_image)
}
