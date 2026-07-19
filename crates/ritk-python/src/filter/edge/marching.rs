//! Fast marching and colliding fronts filters.

use crate::image::{image_from_py, into_py_image, PyImage};
use pyo3::prelude::*;
use ritk_filter::{CollidingFrontsFilter, FastMarchingFilter};

/// Solve the Eikonal arrival-time field by fast marching, matching
/// `SimpleITK.FastMarching`.
///
/// Propagates a front from `trial_points` (seeds) outward through the `image`
/// speed field, solving â€–âˆ‡Tâ€–Â·F = 1. Voxels never reached keep a large sentinel
/// value.
///
/// Args:
///     image: Speed image (non-negative).
///     trial_points: Seed voxels, each `[z, y, x]`.
///     normalization_factor: Speed normalization (default 1.0).
///     stopping_value: Stop once the smallest arrival time exceeds this (default âˆž).
///     initial_trial_values: Initial arrival time per seed; empty â‡’ all 0.
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
    let arc = image_from_py(image);
    let out = py.allow_threads(|| {
        FastMarchingFilter {
            trial_points,
            initial_trial_values,
            normalization_factor,
            stopping_value: stopping_value.unwrap_or(f64::MAX / 2.0),
        }
        .apply(&arc)
    });
    into_py_image(out)
}

/// Colliding-fronts segmentation potential, matching `SimpleITK.CollidingFronts`.
///
/// Two fast-marching fronts are propagated from `seeds1` and `seeds2` through the
/// speed image; the output is the dot product of their upwind gradient fields,
/// strongly negative where the fronts collide. With `apply_connectivity` the
/// result is restricted to the connected region of `P â‰¤ negative_epsilon`
/// reachable from `seeds1`; elsewhere 0.
///
/// Args:
///     image: Speed image (non-negative).
///     seeds1: First front's seed voxels, each `[z, y, x]`.
///     seeds2: Second front's seed voxels, each `[z, y, x]`.
///     apply_connectivity: Restrict to the connected colliding corridor (default True).
///     negative_epsilon: Seed / connectivity threshold (default -1e-6).
///
/// Returns:
///     Colliding-fronts potential PyImage, same shape and metadata as input.
#[pyfunction]
#[pyo3(signature = (image, seeds1, seeds2, apply_connectivity=true, negative_epsilon=-1e-6_f64))]
pub fn colliding_fronts(
    py: Python<'_>,
    image: &PyImage,
    seeds1: Vec<[usize; 3]>,
    seeds2: Vec<[usize; 3]>,
    apply_connectivity: bool,
    negative_epsilon: f64,
) -> PyImage {
    let arc = image_from_py(image);
    let out = py.allow_threads(|| {
        CollidingFrontsFilter {
            seeds1,
            seeds2,
            apply_connectivity,
            negative_epsilon,
        }
        .apply(&arc)
    });
    into_py_image(out)
}
