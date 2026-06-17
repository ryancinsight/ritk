//! Python-exposed intensity projection filters.
//!
//! All filters delegate to `ritk_filter::projection` (SSOT).
//!
//! # Output convention
//!
//! All projection functions return a 3-D PyImage where the projected axis has
//! size 1. For example, maximum intensity projection along Z of a [D, H, W]
//! image returns a [1, H, W] PyImage.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::projection::{
    BinaryProjectionFilter, BinaryThresholdProjectionFilter, MaxIntensityProjectionFilter,
    MeanIntensityProjectionFilter, MedianIntensityProjectionFilter, MinIntensityProjectionFilter,
    ProjectionAxis, StdDevIntensityProjectionFilter, SumIntensityProjectionFilter,
};

/// Parse an axis integer (0, 1, 2) into `ProjectionAxis`.
fn parse_axis(axis: usize) -> RitkResult<ProjectionAxis> {
    match axis {
        0 => Ok(ProjectionAxis::Z),
        1 => Ok(ProjectionAxis::Y),
        2 => Ok(ProjectionAxis::X),
        other => Err(RitkPyError::value(format!(
            "projection axis must be 0 (Z), 1 (Y), or 2 (X); got {other}"
        ))),
    }
}

/// Maximum intensity projection along a spatial axis.
///
/// Reduces the image along `axis` by taking the maximum voxel value. The
/// output has size 1 along the projected axis.
///
/// Args:
///     image: Input PyImage [D, H, W].
///     axis:  Axis to project along — 0 (Z), 1 (Y), or 2 (X).
///
/// Returns:
///     Projected PyImage with size 1 along `axis`.
///
/// Raises:
///     ValueError:   if `axis` is not 0, 1, or 2.
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, axis=0))]
pub fn max_intensity_projection(
    py: Python<'_>,
    image: &PyImage,
    axis: usize,
) -> RitkResult<PyImage> {
    let ax = parse_axis(axis)?;
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        MaxIntensityProjectionFilter::new(ax)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Minimum intensity projection along a spatial axis.
///
/// Args:
///     image: Input PyImage [D, H, W].
///     axis:  Axis to project along — 0 (Z), 1 (Y), or 2 (X).
///
/// Returns:
///     Projected PyImage with size 1 along `axis`.
///
/// Raises:
///     ValueError:   if `axis` is not 0, 1, or 2.
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, axis=0))]
pub fn min_intensity_projection(
    py: Python<'_>,
    image: &PyImage,
    axis: usize,
) -> RitkResult<PyImage> {
    let ax = parse_axis(axis)?;
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        MinIntensityProjectionFilter::new(ax)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Mean intensity projection along a spatial axis.
///
/// Args:
///     image: Input PyImage [D, H, W].
///     axis:  Axis to project along — 0 (Z), 1 (Y), or 2 (X).
///
/// Returns:
///     Projected PyImage with size 1 along `axis`.
///
/// Raises:
///     ValueError:   if `axis` is not 0, 1, or 2.
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, axis=0))]
pub fn mean_intensity_projection(
    py: Python<'_>,
    image: &PyImage,
    axis: usize,
) -> RitkResult<PyImage> {
    let ax = parse_axis(axis)?;
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        MeanIntensityProjectionFilter::new(ax)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Binary projection: foreground if any voxel along `axis` equals `foreground`.
///
/// ITK Parity: BinaryProjectionImageFilter (`sitk.BinaryProjection`).
#[pyfunction]
#[pyo3(signature = (image, axis=0, foreground=1.0, background=0.0))]
pub fn binary_projection(
    py: Python<'_>,
    image: &PyImage,
    axis: usize,
    foreground: f32,
    background: f32,
) -> RitkResult<PyImage> {
    let ax = parse_axis(axis)?;
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        BinaryProjectionFilter::new(ax, foreground, background)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Binary-threshold projection: foreground if any voxel along `axis` is
/// `>= threshold`.
///
/// ITK Parity: BinaryThresholdProjectionImageFilter (`sitk.BinaryThresholdProjection`).
#[pyfunction]
#[pyo3(signature = (image, axis=0, threshold=0.0, foreground=1.0, background=0.0))]
pub fn binary_threshold_projection(
    py: Python<'_>,
    image: &PyImage,
    axis: usize,
    threshold: f32,
    foreground: f32,
    background: f32,
) -> RitkResult<PyImage> {
    let ax = parse_axis(axis)?;
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        BinaryThresholdProjectionFilter::new(ax, threshold, foreground, background)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Median intensity projection along a spatial axis.
///
/// ITK Parity: MedianProjectionImageFilter (`sitk.MedianProjection`).
#[pyfunction]
#[pyo3(signature = (image, axis=0))]
pub fn median_intensity_projection(
    py: Python<'_>,
    image: &PyImage,
    axis: usize,
) -> RitkResult<PyImage> {
    let ax = parse_axis(axis)?;
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        MedianIntensityProjectionFilter::new(ax)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Sum intensity projection along a spatial axis.
///
/// Args:
///     image: Input PyImage [D, H, W].
///     axis:  Axis to project along — 0 (Z), 1 (Y), or 2 (X).
///
/// Returns:
///     Projected PyImage with size 1 along `axis`.
///
/// Raises:
///     ValueError:   if `axis` is not 0, 1, or 2.
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, axis=0))]
pub fn sum_intensity_projection(
    py: Python<'_>,
    image: &PyImage,
    axis: usize,
) -> RitkResult<PyImage> {
    let ax = parse_axis(axis)?;
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        SumIntensityProjectionFilter::new(ax)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Standard-deviation intensity projection along a spatial axis.
///
/// Computes the per-pixel population standard deviation across the projected
/// axis. Useful for temporal or motion variability analysis.
///
/// Args:
///     image: Input PyImage [D, H, W].
///     axis:  Axis to project along — 0 (Z), 1 (Y), or 2 (X).
///
/// Returns:
///     Projected PyImage with size 1 along `axis`.
///
/// Raises:
///     ValueError:   if `axis` is not 0, 1, or 2.
///     RuntimeError: on internal computation failure.
#[pyfunction]
#[pyo3(signature = (image, axis=0))]
pub fn stddev_intensity_projection(
    py: Python<'_>,
    image: &PyImage,
    axis: usize,
) -> RitkResult<PyImage> {
    let ax = parse_axis(axis)?;
    let image = std::sync::Arc::clone(&image.inner);
    py.allow_threads(|| {
        StdDevIntensityProjectionFilter::new(ax)
            .apply(image.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}
