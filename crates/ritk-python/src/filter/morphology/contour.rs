use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{BinaryContourImageFilter, Connectivity, LabelContourImageFilter};

/// Binary contour: keep only foreground voxels that touch the background
/// (the object boundary). ITK Parity: BinaryContourImageFilter
/// (`sitk.BinaryContour`).
#[pyfunction]
#[pyo3(signature = (image, fully_connected = false, foreground_value = 1.0))]
pub fn binary_contour(
    py: Python<'_>,
    image: &PyImage,
    fully_connected: bool,
    foreground_value: f32,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        BinaryContourImageFilter::new(conn, foreground_value)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Label contour: mark the boundary voxels of every labelled region (voxels
/// adjacent to a different label or the background). ITK Parity:
/// LabelContourImageFilter (`sitk.LabelContour`).
#[pyfunction]
#[pyo3(signature = (image, fully_connected = false, background_value = 0.0))]
pub fn label_contour(
    py: Python<'_>,
    image: &PyImage,
    fully_connected: bool,
    background_value: f32,
) -> RitkResult<PyImage> {
    let arc = std::sync::Arc::clone(&image.inner);
    let conn = connectivity_from(fully_connected);
    py.allow_threads(|| {
        LabelContourImageFilter::new(conn, background_value)
            .apply(arc.as_ref())
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
    .map(into_py_image)
}

/// Extract iso-contours at `contour_value` from a 2-D image (`z == 1`) by
/// marching squares, matching `sitk.ContourExtractor2DImageFilter`.
///
/// Returns a list of polyline contours; each contour is a list of `(y, x)`
/// pixel-coordinate vertices (closed loops repeat the first point at the end).
///
/// ITK Parity: ContourExtractor2DImageFilter (`sitk.ContourExtractor2DImageFilter`).
///
/// Args:
///     image:         scalar 2-D image (`z == 1`).
///     contour_value: iso-value to trace (default 0.0).
///
/// Returns:
///     list[list[tuple[float, float]]] — contours of `(y, x)` vertices.
#[pyfunction]
#[pyo3(signature = (image, contour_value = 0.0))]
pub fn contour_extractor_2d(
    py: Python<'_>,
    image: &PyImage,
    contour_value: f32,
) -> Vec<Vec<(f64, f64)>> {
    let arc = std::sync::Arc::clone(&image.inner);
    let contours = py.allow_threads(|| {
        ritk_filter::ContourExtractor2DImageFilter { contour_value }.apply(arc.as_ref())
    });
    contours
        .into_iter()
        .map(|c| c.into_iter().map(|p| (p.y as f64, p.x as f64)).collect())
        .collect()
}

/// Map the `fully_connected` flag to the structuring-element adjacency
/// (ITK's `FullyConnectedOff` → face, `On` → full).
pub(crate) fn connectivity_from(fully_connected: bool) -> Connectivity {
    if fully_connected {
        Connectivity::Vertex26
    } else {
        Connectivity::Face6
    }
}
