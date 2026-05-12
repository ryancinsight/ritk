//! Region growing segmentation: connected-threshold, confidence-connected,
//! and neighbourhood-connected.

use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_core::segmentation::{
    connected_threshold as core_connected_threshold, ConfidenceConnectedFilter,
    NeighborhoodConnectedFilter,
};
use std::sync::Arc;

/// Segment a region by connected-threshold flood-fill from a seed voxel.
///
/// Delegates to `ritk_core::segmentation::connected_threshold`. Grows a
/// 6-connected region from `seed` including all reachable voxels with
/// intensity in [lower, upper].
///
/// Args:
///     image: Input PyImage.
///     seed:  Seed voxel as [z, y, x] indices.
///     lower: Inclusive lower intensity bound.
///     upper: Inclusive upper intensity bound.
///
/// Returns:
///     Binary mask PyImage (1.0 = included, 0.0 = excluded).
///
/// Raises:
///     ValueError: if lower > upper or seed is out of bounds.
#[pyfunction]
#[pyo3(signature = (image, seed, lower, upper))]
pub fn connected_threshold_segment(
    py: Python<'_>,
    image: &PyImage,
    seed: [usize; 3],
    lower: f32,
    upper: f32,
) -> PyResult<PyImage> {
    if lower > upper {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "lower bound ({lower}) must be ≤ upper bound ({upper})"
        )));
    }
    let shape = image.inner.shape();
    if seed[0] >= shape[0] || seed[1] >= shape[1] || seed[2] >= shape[2] {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seed {:?} is out of bounds for image shape {:?}",
            seed, shape
        )));
    }
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| core_connected_threshold(image.as_ref(), seed, lower, upper));
    Ok(into_py_image(result))
}

/// Confidence-connected region growing (Yanowitz & Bruckstein 1989).
///
/// Iteratively grows a region from a seed voxel, adapting the intensity
/// window based on the running mean ± k·σ of currently-included voxels.
///
/// Args:
///     image:          Input PyImage.
///     seed:           Seed voxel as [z, y, x] integer list.
///     initial_lower:  Initial inclusive lower bound (first iteration, when σ=0).
///     initial_upper:  Initial inclusive upper bound (first iteration, when σ=0).
///     multiplier:     k for the adaptive k·σ window expansion (default 2.5).
///     max_iterations: Maximum region-growing iterations (default 15).
///
/// Returns:
///     Binary mask PyImage (1.0=foreground, 0.0=background).
///
/// Raises:
///     ValueError:   if seed does not have exactly 3 elements.
///     RuntimeError: on computation failure.
#[pyfunction]
#[pyo3(signature = (image, seed, initial_lower, initial_upper, multiplier=2.5, max_iterations=15))]
pub fn confidence_connected_segment(
    py: Python<'_>,
    image: &PyImage,
    seed: Vec<usize>,
    initial_lower: f32,
    initial_upper: f32,
    multiplier: f32,
    max_iterations: usize,
) -> PyResult<PyImage> {
    if seed.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seed must have exactly 3 elements, got {}",
            seed.len()
        )));
    }
    let inner = Arc::clone(&image.inner);
    let result = py.allow_threads(move || {
        ConfidenceConnectedFilter::new([seed[0], seed[1], seed[2]], initial_lower, initial_upper)
            .with_multiplier(multiplier)
            .with_max_iterations(max_iterations)
            .apply(inner.as_ref())
    });
    Ok(into_py_image(result))
}

/// Neighbourhood-connected region growing.
///
/// Grows a region from a seed: admits voxels whose rectangular neighbourhood
/// (±radius in each direction) all satisfy the intensity bounds.
///
/// Args:
///     image:  Input PyImage.
///     seed:   Seed voxel as [z, y, x] integer list.
///     lower:  Inclusive lower intensity bound.
///     upper:  Inclusive upper intensity bound.
///     radius: Neighbourhood half-radius (uniform in all 3 axes, default 1 → 3×3×3).
///
/// Returns:
///     Binary mask PyImage (1.0=foreground, 0.0=background).
///
/// Raises:
///     ValueError:   if seed does not have exactly 3 elements.
///     RuntimeError: on computation failure.
#[pyfunction]
#[pyo3(signature = (image, seed, lower, upper, radius=1))]
pub fn neighborhood_connected_segment(
    py: Python<'_>,
    image: &PyImage,
    seed: Vec<usize>,
    lower: f32,
    upper: f32,
    radius: usize,
) -> PyResult<PyImage> {
    if seed.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "seed must have exactly 3 elements, got {}",
            seed.len()
        )));
    }
    let inner = Arc::clone(&image.inner);
    let result = py.allow_threads(move || {
        NeighborhoodConnectedFilter::new([seed[0], seed[1], seed[2]], lower, upper)
            .with_radius([radius, radius, radius])
            .apply(inner.as_ref())
    });
    Ok(into_py_image(result))
}
