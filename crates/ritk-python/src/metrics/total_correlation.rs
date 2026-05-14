//! Total Correlation (Multivariate Mutual Information) pyfunction wrapper.
//!
//! Delegates to `ritk_core::statistics::information::total_correlation`.
//! See that module for the mathematical definition (Watanabe 1960) and
//! complexity constraints (B^n ≤ 4_194_304).

use anyhow::Result;
use pyo3::prelude::*;
use ritk_core::statistics::information::total_correlation as core_tc;

use crate::image::PyImage;
use crate::metrics::image_batch::collect_image_vectors;

/// Total correlation C(X₁,...,Xₙ) = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ).
///
/// Delegates to `ritk_core::statistics::information::total_correlation`.
pub(super) fn total_correlation_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_tc(channels, num_bins)
}

/// Total Correlation (Multivariate Mutual Information) over N images.
///
/// All images must have identical shapes. Returns C ≥ 0.
///
/// # Formula (Watanabe 1960)
/// C(X₁,...,Xₙ) = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ)
///
/// For n=2, this equals standard mutual information I(X₁;X₂).
///
/// # Arguments
/// - `images`: list of PyImage objects (n ≥ 1).
/// - `num_bins`: histogram bins per channel (2 ≤ B ≤ 64, default 32).
#[pyfunction]
#[pyo3(signature = (images, num_bins=32))]
pub fn compute_total_correlation(images: Vec<PyRef<PyImage>>, num_bins: usize) -> PyResult<f64> {
    let (vectors, _) = collect_image_vectors(&images)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    if num_bins < 2 || num_bins > 64 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "num_bins must be in [2, 64], got {}",
            num_bins
        )));
    }

    let slices: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    total_correlation_slices(&slices, num_bins)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}
