//! Python bindings for ensemble segmentation (STAPLE) and GrowCut.

use crate::errors::RitkResult;
use crate::image::vec_to_image_like;
use crate::image::{image_from_py, into_py_image, with_image_slice, PyImage};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use ritk_segmentation::{
    growcut as core_growcut, multi_label_staple as core_multi_label_staple, staple as core_staple,
    StapleConvergence,
};
use std::sync::Arc;

/// Run the STAPLE algorithm on K binary rater segmentation masks.
///
/// STAPLE (Simultaneous Truth and Performance Level Estimation) by Warfield et al. (2004).
/// Runs an EM algorithm to estimate the probabilistic ground truth W âˆˆ \[0,1\]^N and
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
///     - `probabilistic_truth`: list\[float\], length N, values in \[0,1\].
///     - `sensitivity`: list\[float\], length K, per-rater sensitivity p_k.
///     - `specificity`: list\[float\], length K, per-rater specificity q_k.
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
        .map(|r| with_image_slice(r.inner.as_ref(), |s| s.to_vec()))
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
    dict.set_item(
        "converged",
        result.convergence == StapleConvergence::Converged,
    )?;
    Ok(dict.into())
}

/// Run multi-label STAPLE on K integer label maps, returning the hard consensus
/// label image. Matches `SimpleITK.MultiLabelSTAPLE`.
///
/// Args:
///     raters:                list of PyImage, each an integer label map (stored
///                            as f32). All images must have the same shape.
///     max_iter:              Maximum EM iterations; 0 â‡’ iterate to convergence
///                            (default 0).
///     termination_threshold: Stop when the max confusion-matrix change falls
///                            below this (default 1e-5).
///     label_for_undecided:   Label assigned to tie/undecided voxels; None â‡’ L
///                            (max label + 1, the ITK default).
///
/// Returns:
///     PyImage consensus label map (f32), same shape/spacing/origin as `raters[0]`.
///
/// Raises:
///     ValueError: if raters is empty or images have different shapes.
#[pyfunction]
#[pyo3(signature = (raters, max_iter=0, termination_threshold=1e-5, label_for_undecided=None))]
pub fn multi_label_staple(
    py: Python<'_>,
    raters: Vec<PyRef<'_, PyImage>>,
    max_iter: usize,
    termination_threshold: f64,
    label_for_undecided: Option<f32>,
) -> RitkResult<PyImage> {
    if raters.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err("raters must not be empty").into());
    }
    let reference = Arc::clone(&raters[0].inner);
    let shape = reference.shape();
    let rater_vecs: Vec<Vec<f32>> = raters
        .iter()
        .map(|r| with_image_slice(r.inner.as_ref(), |s| s.to_vec()))
        .collect();
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
    let max = (max_iter != 0).then_some(max_iter);
    let result = py.allow_threads(move || {
        core_multi_label_staple(&rater_vecs, max, termination_threshold, label_for_undecided)
    });
    Ok(into_py_image(vec_to_image_like(
        result.labels,
        shape,
        reference.as_ref(),
    )))
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
    let img_arc = image_from_py(image);
    let seed_arc = image_from_py(seeds);
    let result = py.allow_threads(move || core_growcut(&img_arc, &seed_arc, max_iter));
    into_py_image(result)
}
