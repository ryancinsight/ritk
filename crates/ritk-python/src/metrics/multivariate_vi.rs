//! Multivariate Variation of Information pyfunction wrapper.
//!
//! Delegates to `ritk_core::statistics::information::multivariate_variation_of_information`.
//! VI_n(X₁,...,Xₙ) = (2 / n(n−1)) · Σ_{i<j} VI(Xᵢ,Xⱼ)   (average pairwise VI)

use anyhow::Result;
use pyo3::prelude::*;
use ritk_core::statistics::information::multivariate_variation_of_information as core_mvi;

use crate::image::{image_to_vec, PyImage};

/// Average pairwise VI over n image channels.
///
/// Requires `channels.len() ≥ 2`; all slices must have equal length.
pub(super) fn multivariate_vi_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_mvi(channels, num_bins)
}

/// Multivariate Variation of Information over N images.
///
/// All images must have identical shapes. Returns VI_n ≥ 0.
/// Requires n ≥ 2 images.
///
/// # Formula
/// VI_n(X₁,...,Xₙ) = (2 / n(n−1)) · Σ_{i<j} VI(Xᵢ,Xⱼ)
///
/// # Arguments
/// - `images`: list of PyImage objects (n ≥ 2).
/// - `num_bins`: histogram bins per channel (2 ≤ B ≤ 64, default 32).
#[pyfunction]
#[pyo3(signature = (images, num_bins=32))]
pub fn compute_multivariate_variation_of_information(
    images: Vec<PyRef<PyImage>>,
    num_bins: usize,
) -> PyResult<f64> {
    if images.len() < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "at least 2 images required, got {}",
            images.len()
        )));
    }
    let vecs: Vec<(Vec<f32>, [usize; 3])> = images
        .iter()
        .map(|img| image_to_vec(&img.inner))
        .collect::<Result<_, _>>()?;

    let shape_0 = vecs[0].1;
    for (i, (_, shape)) in vecs.iter().enumerate().skip(1) {
        if *shape != shape_0 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "shape mismatch: images[0] {:?} != images[{}] {:?}",
                shape_0, i, shape
            )));
        }
    }
    if num_bins < 2 || num_bins > 64 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "num_bins must be in [2, 64], got {}",
            num_bins
        )));
    }

    let slices: Vec<&[f32]> = vecs.iter().map(|(v, _)| v.as_slice()).collect();
    multivariate_vi_slices(&slices, num_bins)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mvi_identical_slices_is_zero() {
        let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
        let mvi =
            multivariate_vi_slices(&[x.as_slice(), x.as_slice(), x.as_slice()], 8).unwrap();
        assert!(mvi.abs() < 1e-9, "MVI(X,X,X)={mvi:.10} must be 0");
    }
}
