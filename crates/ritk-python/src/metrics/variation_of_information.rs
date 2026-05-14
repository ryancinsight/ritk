//! Variation of Information (VI) pyfunction wrapper.
//!
//! Delegates to `ritk_core::statistics::information::variation_of_information`.
//! See that module for the mathematical definition (Meilă 2003).

use anyhow::Result;
use pyo3::prelude::*;
use ritk_core::statistics::information::variation_of_information as core_vi;

use crate::image::{image_to_vec, PyImage};

/// VI(X,Y) = H(X) + H(Y) − 2·I(X,Y).
///
/// Delegates to `ritk_core::statistics::information::variation_of_information`.
pub(super) fn variation_of_information_slices(
    a: &[f32],
    b: &[f32],
    num_bins: usize,
) -> Result<f64> {
    core_vi(a, b, num_bins)
}

/// Variation of Information between two images.
///
/// # Formula (Meilă 2003)
/// VI(X, Y) = H(X|Y) + H(Y|X) = H(X) + H(Y) − 2·I(X,Y)
///
/// VI = 0 iff the images have identical intensity distributions.
/// VI is symmetric: VI(X,Y) = VI(Y,X).
///
/// # Arguments
/// - `num_bins`: histogram bins per axis (default 32).
#[pyfunction]
#[pyo3(signature = (fixed, moving, num_bins=32))]
pub fn compute_variation_of_information(
    fixed: &PyImage,
    moving: &PyImage,
    num_bins: usize,
) -> PyResult<f64> {
    let (a, shape_a) = image_to_vec(&fixed.inner)?;
    let (b, shape_b) = image_to_vec(&moving.inner)?;
    if shape_a != shape_b {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: fixed {:?} != moving {:?}",
            shape_a, shape_b
        )));
    }
    if num_bins < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_bins must be >= 2",
        ));
    }
    variation_of_information_slices(&a, &b, num_bins)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}
