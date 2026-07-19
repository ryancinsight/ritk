//! Region growing segmentation: connected-threshold, confidence-connected,
//! and neighbourhood-connected.

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{into_py_image, native_into_py_image, py_image_to_native, PyImage};
use coeus_core::{MoiraiBackend, SequentialBackend};
use pyo3::prelude::*;
use ritk_segmentation::{
    ConfidenceConnectedFilter, ConnectedThresholdFilter, IsolatedConnectedConfig,
    IsolatedConnectedFilter, IsolatedWatershed, IsolatedWatershedConfig, IsolationThreshold,
    NeighborhoodConnectedFilter, VectorConfidenceConnectedConfig, VectorConfidenceConnectedFilter,
};

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
    let config = VectorConfidenceConnectedConfig::new(
        multiplier,
        number_of_iterations,
        initial_neighborhood_radius,
        replace_value,
    )
    .map_err(|error| RitkPyError::value(error.to_string()))?;
    let filter = VectorConfidenceConnectedFilter::new(seeds, config);
    let images: Vec<_> = channels.iter().map(|image| image.inner.clone()).collect();
    py.allow_threads(move || {
        let references: Vec<_> = images.iter().map(AsRef::as_ref).collect();
        filter.apply_native(&references, &MoiraiBackend)
    })
    .map(into_py_image)
    .map_err(|error| RitkPyError::value(error.to_string()))
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
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(RitkPyError::value(format!(
            "bounds must be finite and ordered, got [{lower}, {upper}]"
        )));
    }
    let image = image.inner.clone();
    py.allow_threads(move || {
        ConnectedThresholdFilter::new(seed, lower, upper)
            .apply_native(image.as_ref(), &MoiraiBackend)
    })
    .map(into_py_image)
    .map_err(|error| RitkPyError::value(error.to_string()))
}

/// Isolated-connected segmentation, matching `SimpleITK.IsolatedConnected`.
///
/// Binary-searches the threshold that just separates `seed1` from `seed2`: the
/// region grown from `seed1` at the resulting threshold excludes `seed2`. With
/// `find_upper_threshold` (default) the upper bound is searched over the band
/// `[lower, Â·]`; otherwise the lower bound over `[Â·, upper]`.
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
///     `(image, thresholding_failed)`, where `image` is the binary isolated
///     region and `thresholding_failed` records whether the final ITK-compatible
///     band still connects both seeds.
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
) -> RitkResult<(PyImage, bool)> {
    let threshold = if find_upper_threshold {
        IsolationThreshold::Upper
    } else {
        IsolationThreshold::Lower
    };
    let config = IsolatedConnectedConfig::new(
        lower,
        upper,
        replace_value,
        isolated_value_tolerance,
        threshold,
    )
    .map_err(|error| RitkPyError::value(error.to_string()))?;
    let filter = IsolatedConnectedFilter::new(seed1, seed2, config);
    let image = image.inner.clone();
    py.allow_threads(move || filter.apply_native(image.as_ref(), &MoiraiBackend))
        .map(|output| {
            let thresholding_failed = output.thresholding_failed();
            (into_py_image(output.into_image()), thresholding_failed)
        })
        .map_err(|error| RitkPyError::value(error.to_string()))
}

/// Confidence-connected region growing (Yanowitz & Bruckstein 1989).
///
/// Iteratively grows a region from a seed voxel, adapting the intensity
/// window based on the running mean Â± kÂ·Ïƒ of currently-included voxels.
///
/// Args:
///     image:          Input PyImage.
///     seed:           Seed voxel as [z, y, x] integer list.
///     initial_lower:  Initial inclusive lower bound (first iteration, when Ïƒ=0).
///     initial_upper:  Initial inclusive upper bound (first iteration, when Ïƒ=0).
///     multiplier:     k for the adaptive kÂ·Ïƒ window expansion (default 2.5).
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
    if !initial_lower.is_finite() || !initial_upper.is_finite() || initial_lower > initial_upper {
        return Err(RitkPyError::value(format!(
            "initial bounds must be finite and ordered, got [{initial_lower}, {initial_upper}]"
        )));
    }
    if !multiplier.is_finite() || multiplier < 0.0 {
        return Err(RitkPyError::value(format!(
            "multiplier must be finite and non-negative, got {multiplier}"
        )));
    }
    let image = image.inner.clone();
    let filter =
        ConfidenceConnectedFilter::new([seed[0], seed[1], seed[2]], initial_lower, initial_upper)
            .with_multiplier(multiplier)
            .map_err(|error| RitkPyError::value(error.to_string()))?
            .with_max_iterations(max_iterations);
    py.allow_threads(move || filter.apply_native(image.as_ref(), &MoiraiBackend))
        .map(into_py_image)
        .map_err(|error| RitkPyError::value(error.to_string()))
}

/// Neighbourhood-connected region growing.
///
/// Grows a region from a seed: admits voxels whose rectangular neighbourhood
/// (Â±radius in each direction) all satisfy the intensity bounds.
///
/// Args:
///     image:  Input PyImage.
///     seed:   Seed voxel as [z, y, x] integer list.
///     lower:  Inclusive lower intensity bound.
///     upper:  Inclusive upper intensity bound.
///     radius: Neighbourhood half-radius (uniform in all 3 axes, default 1 â†’ 3Ã—3Ã—3).
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
    if !lower.is_finite() || !upper.is_finite() || lower > upper {
        return Err(RitkPyError::value(format!(
            "bounds must be finite and ordered, got [{lower}, {upper}]"
        )));
    }
    let image = image.inner.clone();
    py.allow_threads(move || {
        NeighborhoodConnectedFilter::new([seed[0], seed[1], seed[2]], lower, upper)
            .with_radius([radius, radius, radius])
            .apply_native(image.as_ref(), &MoiraiBackend)
    })
    .map(into_py_image)
    .map_err(|error| RitkPyError::value(error.to_string()))
}

// â”€â”€ IsolatedWatershed â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Isolated watershed segmentation, matching `SimpleITK.IsolatedWatershed`.
///
/// Finds the lowest threshold T* such that `seed1` and `seed2` are in separate
/// 6-connected components of {x : I(x) â‰¤ T*}, then labels those regions.
///
/// Label convention:
///     1 = region reachable from seed1 at T*
///     2 = region reachable from seed2 at T*
///     0 = remaining voxels
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
///     Label PyImage (f32 values 0.0, 1.0, or 2.0).
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
    isolated_value_tolerance: f64,
    upper_value_limit: f64,
) -> RitkResult<PyImage> {
    let image = py_image_to_native(image)?;
    let config =
        IsolatedWatershedConfig::new(threshold, isolated_value_tolerance, upper_value_limit)
            .map_err(|error| RitkPyError::value(error.to_string()))?;
    let result = py.allow_threads(|| {
        IsolatedWatershed::new(seed1, seed2, config).apply_native(&image, &SequentialBackend)
    });
    result
        .map(native_into_py_image)
        .map_err(|error| RitkPyError::value(error.to_string()))
}
