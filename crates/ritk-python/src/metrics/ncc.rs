//! NCC (Normalized Cross-Correlation) pyfunction wrapper.
//!
//! # Formula
//! NCC = Σ(aᵢ−ā)(bᵢ−b̄) / √(Σ(aᵢ−ā)² · Σ(bᵢ−b̄)²)

use pyo3::prelude::*;
use ritk_statistics::pearson_correlation;

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{with_image_pair_slices, PyImage};

/// Normalized cross-correlation between two images (Pearson r).
///
/// Both images must have identical shapes. Returns r ∈ [−1, 1].
///
/// # Formula
/// NCC = Σ(aᵢ−ā)(bᵢ−b̄) / √(Σ(aᵢ−ā)² · Σ(bᵢ−b̄)²)
#[pyfunction]
pub fn compute_ncc(py: Python<'_>, fixed: &PyImage, moving: &PyImage) -> RitkResult<f64> {
    let shape_a = fixed.inner.shape();
    let shape_b = moving.inner.shape();
    if shape_a != shape_b {
        return Err(RitkPyError::value(format!(
            "shape mismatch: fixed {:?} != moving {:?}",
            shape_a, shape_b
        )));
    }
    let fixed = fixed.inner.clone();
    let moving = moving.inner.clone();
    py.allow_threads(move || {
        with_image_pair_slices(&fixed, &moving, pearson_correlation)
            .map_err(|e| RitkPyError::runtime(e.to_string()))
    })
}
