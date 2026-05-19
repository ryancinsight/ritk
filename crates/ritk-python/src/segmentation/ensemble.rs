//! Python bindings for ensemble segmentation (STAPLE) and GrowCut.

use crate::errors::RitkResult;
use crate::image::{into_py_image, with_tensor_slice, PyImage};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use ritk_core::segmentation::{growcut as core_growcut, staple as core_staple};
use std::sync::Arc;

/// Run the STAPLE algorithm on K binary rater segmentation masks.
///
/// STAPLE (Simultaneous Truth and Performance Level Estimation) by Warfield et al. (2004).
/// Runs an EM algorithm to estimate the probabilistic ground truth W ∈ [0,1]^N and
/// per-rater sensitivity p_k and specificity q_k.
///
/// Args:
///     raters:    list of PyImage, each a binary segmentation mask (0.0=negative, 1.0=positive).
///                All images must have the same shape.
///     max_iter:  Maximum EM iterations (default 100).
///     tol:       Convergence tolerance on per-rater parameter change (default 1e-6).
///
/// Returns:
///     dict with keys:
///     - `probabilistic_truth`: list[float], length N, values in [0,1].
///     - `sensitivity`: list[float], length K, per-rater sensitivity p_k.
///     - `specificity`: list[float], length K, per-rater specificity q_k.
///     - `iterations`: int, number of EM iterations executed.
///     - `converged`: bool, True if algorithm converged before max_iter.
///
/// Raises:
///     ValueError: if raters is empty or images have different shapes.
#[pyfunction]
#[pyo3(signature = (raters, max_iter=100, tol=1e-6))]
pub fn staple_ensemble(
    py: Python<'_>,
    raters: Vec<PyRef<'_, PyImage>>,
    max_iter: usize,
    tol: f64,
) -> RitkResult<Py<PyDict>> {
    if raters.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("raters must not be empty").into());
    }
    // Extract flat f32 vecs from each rater.
    let rater_vecs: Vec<Vec<f32>> = raters
        .iter()
        .map(|r| with_tensor_slice(r.inner.data(), |s| s.to_vec()))
        .collect();

    // Validate lengths match.
    let n = rater_vecs[0].len();
    for (i, v) in rater_vecs.iter().enumerate().skip(1) {
        if v.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "rater {i} has length {} but rater 0 has length {n}",
                v.len()
            ))
            .into());
        }
    }

    let result = py.allow_threads(move || core_staple(&rater_vecs, max_iter, tol));

    let dict = PyDict::new_bound(py);
    dict.set_item(
        "probabilistic_truth",
        result
            .probabilistic_truth
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "sensitivity",
        result
            .sensitivity
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "specificity",
        result
            .specificity
            .iter()
            .map(|&v| v as f64)
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("iterations", result.iterations)?;
    dict.set_item("converged", result.converged)?;
    Ok(dict.into())
}

/// GrowCut interactive segmentation.
///
/// Cellular automaton segmentation (Vezhnevets & Konouchine, GRAPHITE 2005).
/// Propagates seed labels through the image based on intensity similarity.
///
/// Args:
///     image:    Input intensity PyImage.
///     seeds:    Seed label PyImage (same shape as image). Non-zero values are
///               treated as immutable seed labels; 0 = unlabeled.
///     max_iter: Maximum automaton iterations (default 200).
///
/// Returns:
///     PyImage with integer label map (values stored as f32).
///     Shape, spacing, and origin match the input image.
///
/// Raises:
///     RuntimeError: if image and seeds have different shapes.
#[pyfunction]
#[pyo3(signature = (image, seeds, max_iter=200))]
pub fn growcut_segment(
    py: Python<'_>,
    image: &PyImage,
    seeds: &PyImage,
    max_iter: usize,
) -> PyImage {
    let img_arc = Arc::clone(&image.inner);
    let seed_arc = Arc::clone(&seeds.inner);
    let result =
        py.allow_threads(move || core_growcut(img_arc.as_ref(), seed_arc.as_ref(), max_iter));
    into_py_image(result)
}
