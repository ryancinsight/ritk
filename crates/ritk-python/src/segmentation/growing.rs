//! Region growing segmentation: connected-threshold, confidence-connected,
//! and neighbourhood-connected.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::Backend;
use crate::image::{into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_image::Image;
use ritk_segmentation::{
    connected_threshold as core_connected_threshold,
    vector_confidence_connected_image as core_vector_confidence_connected,
    ConfidenceConnectedFilter, IsolatedConnectedFilter, IsolatedWatershed,
    NeighborhoodConnectedFilter,
};
use std::sync::Arc;

/// Vector confidence-connected region growing, matching
/// `sitk.VectorConfidenceConnected`.
///
/// Grows a region from `seeds` over a multi-channel image: a voxel joins when its
/// Mahalanobis distance to the region's mean/covariance is within `multiplier`
/// standard deviations, iterated `number_of_iterations` times.
///
/// ITK Parity: `VectorConfidenceConnectedImageFilter`. Region-exact to SimpleITK
/// for well-conditioned inputs; near-singular covariance (tiny regions at very
/// tight multipliers) is a documented cross-implementation numerical limit.
///
/// Args:
///     channels: list of scalar component images (one per vector component).
///     seeds: list of `[z, y, x]` seed indices.
///     multiplier: confidence-interval width (default 2.5).
///     number_of_iterations: statistic-recompute passes (default 4).
///     initial_neighborhood_radius: seed-statistics radius (default 1).
///     replace_value: in-region label value (default 1.0).
///
/// Returns:
///     binary label image (`replace_value` inside the region).
#[pyfunction]
#[pyo3(signature = (channels, seeds, multiplier=2.5, number_of_iterations=4,
                    initial_neighborhood_radius=1, replace_value=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn vector_confidence_connected_segment(
    py: Python<'_>,
    channels: Vec<PyRef<'_, PyImage>>,
    seeds: Vec<[usize; 3]>,
    multiplier: f64,
    number_of_iterations: u32,
    initial_neighborhood_radius: usize,
    replace_value: f32,
) -> RitkResult<PyImage> {
    if channels.is_empty() {
        return Err(RitkPyError::value(
            "vector_confidence_connected: at least one channel is required",
        ));
    }
    let arcs: Vec<Arc<Image<Backend, 3>>> = channels.iter().map(|p| Arc::clone(&p.inner)).collect();
    let out = py
        .allow_threads(|| {
            let refs: Vec<&Image<Backend, 3>> = arcs.iter().map(|a| a.as_ref()).collect();
            core_vector_confidence_connected(
                &refs,
                &seeds,
                multiplier,
                number_of_iterations,
                initial_neighborhood_radius,
                replace_value,
            )
        })
        .map_err(RitkPyError::value)?;
    Ok(into_py_image(out))
}

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

// ── IsolatedWatershed ────────────────────────────────────────────────────────

/// Isolated watershed segmentation, matching `SimpleITK.IsolatedWatershed`.
///
/// Finds the lowest threshold T* such that `seed1` and `seed2` are in separate
/// 6-connected components of {x : I(x) ≤ T*}, then labels those regions.
///
/// Label convention:
///     1 = region reachable from seed1 at T*
///     2 = region reachable from seed2 at T*
///     3 = remaining voxels
///
/// Args:
///     image:                    Input scalar 3-D PyImage (normalised to [0, 1]).
///     seed1:                    `[z, y, x]` index of the first seed voxel.
///     seed2:                    `[z, y, x]` index of the second seed voxel.
///     threshold:                Lower bound for the search (default 0.0).
///     isolated_value_tolerance: Binary-search convergence precision (default 0.001).
///     upper_value_limit:        Upper bound for the search (default 1.0).
///
/// Returns:
///     Label PyImage (f32 values 1.0, 2.0, or 3.0).
#[pyfunction]
#[pyo3(signature = (image, seed1, seed2, threshold=0.0,
                    isolated_value_tolerance=0.001, upper_value_limit=1.0))]
#[allow(clippy::too_many_arguments)]
pub fn isolated_watershed_segment(
    py: Python<'_>,
    image: &PyImage,
    seed1: [usize; 3],
    seed2: [usize; 3],
    threshold: f32,
    isolated_value_tolerance: f32,
    upper_value_limit: f32,
) -> RitkResult<PyImage> {
    let arc = Arc::clone(&image.inner);
    let result = py.allow_threads(|| {
        IsolatedWatershed {
            seed1,
            seed2,
            threshold,
            isolated_value_tolerance,
            upper_value_limit,
        }
        .apply(arc.as_ref())
    });
    result
        .map(into_py_image)
        .map_err(|e| crate::errors::RitkPyError::runtime(e.to_string()))
}
