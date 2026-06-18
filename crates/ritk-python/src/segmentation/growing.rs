//! Region growing segmentation: connected-threshold, confidence-connected,
//! and neighbourhood-connected.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_segmentation::{
    connected_threshold as core_connected_threshold, ConfidenceConnectedFilter,
    IsolatedConnectedFilter, NeighborhoodConnectedFilter,
};
use std::sync::Arc;

/// Segment a region by connected-threshold flood-fill from a seed voxel.
///
/// Delegates to `ritk_segmentation::connected_threshold`. Grows a
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
) -> RitkResult<PyImage> {
    if lower > upper {
        return Err(RitkPyError::value(format!(
            "lower bound ({lower}) must be ≤ upper bound ({upper})"
        )));
    }
    let shape = image.inner.shape();
    if seed[0] >= shape[0] || seed[1] >= shape[1] || seed[2] >= shape[2] {
        return Err(RitkPyError::value(format!(
            "seed {:?} is out of bounds for image shape {:?}",
            seed, shape
        )));
    }
    let image = Arc::clone(&image.inner);
    let result = py.allow_threads(|| core_connected_threshold(image.as_ref(), seed, lower, upper));
    Ok(into_py_image(result))
}

/// Isolated-connected segmentation, matching `SimpleITK.IsolatedConnected`.
///
/// Binary-searches the threshold that just separates `seed1` from `seed2`: the
/// region grown from `seed1` at the resulting threshold excludes `seed2`. With
/// `find_upper_threshold` (default) the upper bound is searched over the band
/// `[lower, ·]`; otherwise the lower bound over `[·, upper]`.
///
/// Args:
///     image:  Input PyImage.
///     seed1:  Seed to keep, [z, y, x].
///     seed2:  Seed to isolate out, [z, y, x].
///     lower:  Band floor / search floor (default 0.0).
///     upper:  Band ceiling / search ceiling (default 1.0).
///     replace_value: Value written to the grown region (default 1.0).
///     isolated_value_tolerance: Bisection stop tolerance (default 1.0).
///     find_upper_threshold: Search the upper threshold (default True).
///
/// Returns:
///     Binary PyImage of the isolated region.
#[pyfunction]
#[pyo3(signature = (image, seed1, seed2, lower=0.0_f32, upper=1.0_f32,
                    replace_value=1.0_f32, isolated_value_tolerance=1.0_f64,
                    find_upper_threshold=true))]
#[allow(clippy::too_many_arguments)]
pub fn isolated_connected_segment(
    py: Python<'_>,
    image: &PyImage,
    seed1: [usize; 3],
    seed2: [usize; 3],
    lower: f32,
    upper: f32,
    replace_value: f32,
    isolated_value_tolerance: f64,
    find_upper_threshold: bool,
) -> PyImage {
    let arc = Arc::clone(&image.inner);
    let out = py.allow_threads(|| {
        IsolatedConnectedFilter {
            seed1,
            seed2,
            lower,
            upper,
            replace_value,
            isolated_value_tolerance,
            find_upper_threshold,
        }
        .apply(arc.as_ref())
    });
    into_py_image(out)
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
) -> RitkResult<PyImage> {
    if seed.len() != 3 {
        return Err(RitkPyError::value(format!(
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
) -> RitkResult<PyImage> {
    if seed.len() != 3 {
        return Err(RitkPyError::value(format!(
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
