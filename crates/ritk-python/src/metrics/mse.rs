//! MSE (Mean Squared Error) pyfunction wrapper.
//!
//! # Formula
//! MSE = Σ(aᵢ − bᵢ)² / N

use pyo3::prelude::*;

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{with_image_pair_slices, PyImage};

/// MSE = Σ(aᵢ − bᵢ)² / N.
pub(super) fn mse_slices(a: &[f32], b: &[f32]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| {
            let d = ai as f64 - bi as f64;
            d * d
        })
        .sum();
    sum / n as f64
}

/// Mean squared error between two images.
///
/// Both images must have identical shapes. Returns the scalar MSE as f64.
///
/// # Formula
/// MSE = Σ(fixed_i − moving_i)² / N
#[pyfunction]
pub fn compute_mse(py: Python<'_>, fixed: &PyImage, moving: &PyImage) -> RitkResult<f64> {
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
    Ok(py.allow_threads(move || with_image_pair_slices(&fixed, &moving, mse_slices)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mse_identical_images_returns_zero() {
        let result = mse_slices(&[1.0, 2.0, 3.0, 4.0], &[1.0, 2.0, 3.0, 4.0]);
        assert!(
            result.abs() < 1e-12,
            "MSE of identical must be 0, got {result}"
        );
    }

    #[test]
    fn mse_known_pair_is_analytically_correct() {
        // A=[0,1,2,3], B=[1,2,3,4]; diffs=[-1,-1,-1,-1]; MSE=4/4=1.0
        let a = [0.0_f32, 1.0, 2.0, 3.0];
        let b = [1.0_f32, 2.0, 3.0, 4.0];
        let result = mse_slices(&a, &b);
        assert!(
            (result - 1.0).abs() < 1e-12,
            "MSE([0..3],[1..4]) must be 1.0, got {result}"
        );
    }

    #[test]
    fn mse_empty_returns_zero() {
        assert_eq!(mse_slices(&[], &[]), 0.0);
    }
}
