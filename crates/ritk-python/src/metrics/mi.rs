//! Histogram-based Mutual Information variants plus entropy pyfunction wrappers.
//!
//! Delegates all variants to `ritk_core::statistics::information`:
//! - "standard":   hard nearest-bin assignment — `mutual_information`
//! - "normalized": symmetric uncertainty 2·I/(H(A)+H(B)) ∈ [0,1] — `symmetric_uncertainty`
//! - "mattes":     bilinear soft-binning (Mattes 2003) — `mutual_information_mattes`
//!
//! New entrypoints:
//! - `compute_entropy`: marginal entropy H(X).
//! - `compute_joint_entropy`: joint entropy H(X,Y).
//! - `compute_symmetric_uncertainty`: SU = 2·MI/(H(X)+H(Y)) ∈ [0,1].

use anyhow::Result;
use pyo3::prelude::*;
use ritk_core::statistics::information::{
    joint_entropy as core_joint_entropy, marginal_entropy as core_marginal_entropy,
    mutual_information as core_mi, mutual_information_mattes as core_mi_mattes,
    symmetric_uncertainty as core_su,
};

use crate::image::{image_to_vec, PyImage};

/// Histogram-based MI with configurable binning strategy.
///
/// `variant` must be one of `"mattes"`, `"standard"`, `"normalized"`.
pub(super) fn mi_slices(a: &[f32], b: &[f32], num_bins: usize, variant: &str) -> Result<f64> {
    match variant {
        "mattes" => core_mi_mattes(a, b, num_bins),
        "standard" => core_mi(a, b, num_bins),
        "normalized" => core_su(a, b, num_bins),
        _ => unreachable!("variant validated before mi_slices"),
    }
}

/// Marginal Shannon entropy H(X) of a single image.
///
/// # Formula
/// H(X) = −Σₖ p(k) log p(k)
///
/// # Arguments
/// - `image`: 3D image.
/// - `num_bins`: histogram bins (default 64).
#[pyfunction]
#[pyo3(signature = (image, num_bins=64))]
pub fn compute_entropy(image: &PyImage, num_bins: usize) -> PyResult<f64> {
    let (a, _) = image_to_vec(&image.inner)?;
    super::validate_num_bins(num_bins)?;
    core_marginal_entropy(&a, num_bins)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Joint entropy H(X,Y) between two images.
///
/// Both images must have identical shapes.
///
/// # Formula
/// H(X,Y) = −Σⱼₖ p(j,k) log p(j,k)
///
/// # Arguments
/// - `num_bins`: histogram bins per axis (default 64).
#[pyfunction]
#[pyo3(signature = (fixed, moving, num_bins=64))]
pub fn compute_joint_entropy(fixed: &PyImage, moving: &PyImage, num_bins: usize) -> PyResult<f64> {
    let (a, shape_a) = image_to_vec(&fixed.inner)?;
    let (b, shape_b) = image_to_vec(&moving.inner)?;
    if shape_a != shape_b {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: fixed {:?} != moving {:?}",
            shape_a, shape_b
        )));
    }
    super::validate_num_bins(num_bins)?;
    core_joint_entropy(&a, &b, num_bins)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Symmetric uncertainty SU(X,Y) = 2·I(X;Y) / (H(X) + H(Y)) ∈ [0,1].
///
/// Both images must have identical shapes.
///
/// # Formula (Fayyad & Irani 1993)
/// SU = 2·I(X;Y) / (H(X) + H(Y))
///
/// SU=1 iff X=Y (or X is a deterministic function of Y); SU=0 iff X and Y are independent.
///
/// # Arguments
/// - `num_bins`: histogram bins per axis (default 64).
#[pyfunction]
#[pyo3(signature = (fixed, moving, num_bins=64))]
pub fn compute_symmetric_uncertainty(
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
    super::validate_num_bins(num_bins)?;
    core_su(&a, &b, num_bins).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Mutual information between two images.
///
/// Both images must have identical shapes.
///
/// # Arguments
/// - `num_bins`: histogram bins per axis (default 64).
/// - `variant`: `"mattes"` (default), `"standard"`, or `"normalized"`.
///
/// # Formula
/// MI = H(A) + H(B) − H(A,B)
#[pyfunction]
#[pyo3(signature = (fixed, moving, num_bins=64, variant="mattes"))]
pub fn compute_mutual_information(
    fixed: &PyImage,
    moving: &PyImage,
    num_bins: usize,
    variant: &str,
) -> PyResult<f64> {
    let (a, shape_a) = image_to_vec(&fixed.inner)?;
    let (b, shape_b) = image_to_vec(&moving.inner)?;
    if shape_a != shape_b {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: fixed {:?} != moving {:?}",
            shape_a, shape_b
        )));
    }
    super::validate_num_bins(num_bins)?;
    match variant {
        "mattes" | "standard" | "normalized" => {}
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown variant '{}'; expected one of: mattes, standard, normalized",
                other
            )));
        }
    }
    mi_slices(&a, &b, num_bins, variant)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mi_self_exceeds_constant() {
        // Analytical: MI(A,A) = H(A) > 0 for non-constant A.
        // MI(A, constant) = 0 since H(constant) = 0.
        let a: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let b_const: Vec<f32> = vec![5.0_f32; 32];
        let mi_self = mi_slices(&a, &a, 16, "standard").unwrap();
        let mi_const = mi_slices(&a, &b_const, 16, "standard").unwrap();
        assert!(
            mi_self > 0.0,
            "MI(A,A) must be positive for non-constant A, got {mi_self}"
        );
        assert!(
            mi_const.abs() < 1e-10,
            "MI(A,constant) must be 0, got {mi_const}"
        );
    }

    #[test]
    fn mi_normalized_variant_in_zero_one() {
        // SU = 2·I/(H(A)+H(B)) ∈ [0,1].
        let a: Vec<f32> = (0..64).map(|x| (x % 16) as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| ((x + 4) % 16) as f32).collect();
        let su = mi_slices(&a, &b, 16, "normalized").unwrap();
        assert!(
            (0.0..=1.0).contains(&su),
            "symmetric uncertainty must be in [0,1], got {su}"
        );
    }

    #[test]
    fn mi_normalized_identical_is_one() {
        // SU(X,X) = 2·H(X)/(H(X)+H(X)) = 1.0.
        let a: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let su = mi_slices(&a, &a, 16, "normalized").unwrap();
        assert!((su - 1.0).abs() < 1e-9, "SU(X,X) must equal 1.0, got {su}");
    }

    #[test]
    fn mi_mattes_self_exceeds_zero() {
        let a: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let mi = mi_slices(&a, &a, 16, "mattes").unwrap();
        assert!(mi > 0.0, "Mattes MI(A,A) must be positive, got {mi}");
    }

    #[test]
    fn marginal_entropy_nonconstant_is_positive() {
        // H(X) > 0 for non-constant X.
        let a: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let h = core_marginal_entropy(&a, 16).unwrap();
        assert!(h > 0.0, "H(X) must be positive for non-constant X, got {h}");
    }

    #[test]
    fn marginal_entropy_constant_is_zero() {
        // H(constant) = 0.
        let a = vec![5.0_f32; 32];
        let h = core_marginal_entropy(&a, 16).unwrap();
        assert!(h.abs() < 1e-10, "H(constant) must be 0, got {h}");
    }

    #[test]
    fn joint_entropy_geq_marginal_entropy() {
        // H(X,Y) ≥ H(X) always.
        let a: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| ((x / 8) % 8) as f32).collect();
        let h_xy = core_joint_entropy(&a, &b, 16).unwrap();
        let h_x = core_marginal_entropy(&a, 16).unwrap();
        assert!(
            h_xy >= h_x - 1e-9,
            "H(X,Y) must be >= H(X), got H(X,Y)={h_xy:.6}, H(X)={h_x:.6}"
        );
    }

    #[test]
    fn symmetric_uncertainty_self_is_one() {
        // SU(X,X) = 2·MI(X,X)/(H(X)+H(X)) = 1.0.
        let a: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let su = core_su(&a, &a, 16).unwrap();
        assert!((su - 1.0).abs() < 1e-9, "SU(X,X) must be 1.0, got {su}");
    }
}
