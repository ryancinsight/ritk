//! Python-exposed image similarity metrics.
//!
//! # Functions
//! - `compute_mse`: mean squared error.
//! - `compute_ncc`: normalized cross-correlation (Pearson r).
//! - `compute_mutual_information`: histogram-based MI (mattes / standard / normalized).
//! - `compute_conditional_mutual_information`: I(X;Y|Z) histogram-based CMI.
//! - `compute_interaction_information`: II(X;Y;Z) interaction information (McGill 1954).
//! - `compute_total_correlation`: multivariate MI (total correlation) over N channels.
//! - `compute_variation_of_information`: VI = H(X|Y) + H(Y|X).
//! - `compute_multivariate_variation_of_information`: average pairwise VI over N channels.
//!
//! # Mathematical foundations
//! - MSE   = Σ(aᵢ−bᵢ)² / N
//! - NCC   = Σ(aᵢ−ā)(bᵢ−b̄) / (N·σ_a·σ_b + ε)
//! - MI    = H(A) + H(B) − H(A,B)
//! - CMI   = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)
//! - II    = I(X;Y) − I(X;Y|Z)            (McGill 1954)
//! - TC    = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ)     (Watanabe 1960)
//! - VI    = H(X) + H(Y) − 2·I(X,Y)       (Meilă 2003)
//! - VI_n  = (2/n(n−1)) · Σ_{i<j} VI(Xᵢ,Xⱼ)

mod cmi;
mod mi;
mod mse;
mod multivariate_vi;
mod ncc;
mod total_correlation;
mod variation_of_information;

use pyo3::prelude::*;

use crate::image::image_to_vec;
use crate::image::PyImage;
use cmi::{cmi_slices, ii_slices};
use mi::mi_slices;
use mse::mse_slices;
use multivariate_vi::multivariate_vi_slices;
use ncc::ncc_slices;
use total_correlation::total_correlation_slices;
use variation_of_information::variation_of_information_slices;

// ── PyO3 public functions ─────────────────────────────────────────────────────

/// Mean squared error between two images.
///
/// Both images must have identical shapes. Returns the scalar MSE as f64.
///
/// # Formula
/// MSE = Σ(fixed_i − moving_i)² / N
#[pyfunction]
pub fn compute_mse(fixed: &PyImage, moving: &PyImage) -> PyResult<f64> {
    let (a, shape_a) = image_to_vec(&fixed.inner)?;
    let (b, shape_b) = image_to_vec(&moving.inner)?;
    if shape_a != shape_b {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: fixed {:?} != moving {:?}",
            shape_a, shape_b
        )));
    }
    Ok(mse_slices(&a, &b))
}

/// Normalized cross-correlation between two images (Pearson r).
///
/// Both images must have identical shapes. Returns r ∈ [−1, 1].
///
/// # Formula
/// NCC = Σ(aᵢ−ā)(bᵢ−b̄) / (N·σ_a·σ_b + ε)
#[pyfunction]
pub fn compute_ncc(fixed: &PyImage, moving: &PyImage) -> PyResult<f64> {
    let (a, shape_a) = image_to_vec(&fixed.inner)?;
    let (b, shape_b) = image_to_vec(&moving.inner)?;
    if shape_a != shape_b {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: fixed {:?} != moving {:?}",
            shape_a, shape_b
        )));
    }
    ncc_slices(&a, &b)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
    if num_bins < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_bins must be >= 2",
        ));
    }
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
    if images.is_empty() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "images list must not be empty",
        ));
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
    total_correlation_slices(&slices, num_bins)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Variation of Information between two images.
///
/// Both images must have identical shapes. Returns VI ≥ 0.
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

/// Conditional Mutual Information I(X;Y|Z) between three images.
///
/// All images must have identical shapes. Returns CMI ≥ 0.
///
/// # Formula
/// I(X;Y|Z) = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)
///
/// # Arguments
/// - `num_bins`: histogram bins per axis (default 32).
#[pyfunction]
#[pyo3(signature = (x_img, y_img, z_img, num_bins=32))]
pub fn compute_conditional_mutual_information(
    x_img: &PyImage,
    y_img: &PyImage,
    z_img: &PyImage,
    num_bins: usize,
) -> PyResult<f64> {
    let (x, shape_x) = image_to_vec(&x_img.inner)?;
    let (y, shape_y) = image_to_vec(&y_img.inner)?;
    let (z, shape_z) = image_to_vec(&z_img.inner)?;
    if shape_x != shape_y || shape_x != shape_z {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: x {:?}, y {:?}, z {:?}",
            shape_x, shape_y, shape_z
        )));
    }
    if num_bins < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_bins must be >= 2",
        ));
    }
    cmi_slices(&x, &y, &z, num_bins)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

/// Interaction Information II(X;Y;Z) between three images (McGill 1954).
///
/// All images must have identical shapes. Result may be negative.
///
/// # Formula
/// II(X;Y;Z) = I(X;Y) − I(X;Y|Z)
///
/// - II > 0: Z introduces synergy (knowing Z increases I(X;Y)).
/// - II < 0: Z is redundant (knowing Z decreases apparent I(X;Y)).
/// - II = 0: Z has no net effect on I(X;Y).
///
/// # Arguments
/// - `num_bins`: histogram bins per axis (default 32).
#[pyfunction]
#[pyo3(signature = (x_img, y_img, z_img, num_bins=32))]
pub fn compute_interaction_information(
    x_img: &PyImage,
    y_img: &PyImage,
    z_img: &PyImage,
    num_bins: usize,
) -> PyResult<f64> {
    let (x, shape_x) = image_to_vec(&x_img.inner)?;
    let (y, shape_y) = image_to_vec(&y_img.inner)?;
    let (z, shape_z) = image_to_vec(&z_img.inner)?;
    if shape_x != shape_y || shape_x != shape_z {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "shape mismatch: x {:?}, y {:?}, z {:?}",
            shape_x, shape_y, shape_z
        )));
    }
    if num_bins < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_bins must be >= 2",
        ));
    }
    ii_slices(&x, &y, &z, num_bins)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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

// ── module registration ───────────────────────────────────────────────────────

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "metrics")?;
    m.add_function(wrap_pyfunction!(compute_mse, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_ncc, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_mutual_information, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_conditional_mutual_information, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_interaction_information, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_total_correlation, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_variation_of_information, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_multivariate_variation_of_information, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

// ── integration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Shape, Tensor, TensorData};
    use burn_ndarray::NdArray;
    use ritk_core::{
        image::Image,
        spatial::{Direction, Point, Spacing},
    };
    use std::sync::Arc;

    type B = NdArray<f32>;

    fn make_image(values: Vec<f32>, shape: [usize; 3]) -> PyImage {
        let device = <B as burn::tensor::backend::Backend>::Device::default();
        let td = TensorData::new(values, Shape::new(shape));
        let tensor = Tensor::<B, 3>::from_data(td, &device);
        let img = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        PyImage { inner: Arc::new(img) }
    }

    #[test]
    fn mse_rejects_shape_mismatch() {
        pyo3::prepare_freethreaded_python();
        let a = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let err = compute_mse(&a, &b).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn ncc_rejects_shape_mismatch() {
        pyo3::prepare_freethreaded_python();
        let a = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let err = compute_ncc(&a, &b).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn mi_unknown_variant_errors() {
        pyo3::prepare_freethreaded_python();
        let img = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let err = compute_mutual_information(&img, &img, 32, "bogus").unwrap_err();
        assert!(err.to_string().contains("unknown variant"));
    }

    #[test]
    fn total_correlation_empty_list_errors() {
        pyo3::prepare_freethreaded_python();
        let tc = compute_total_correlation(vec![], 16).unwrap_err();
        assert!(tc.to_string().contains("empty"), "expected empty error: {tc}");
    }

    #[test]
    fn vi_identical_images_is_zero_via_pyfunction() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let img = make_image(v, [4, 4, 4]);
        let vi = compute_variation_of_information(&img, &img, 8).unwrap();
        assert!(vi.abs() < 1e-10, "VI(X,X) must be 0, got {vi}");
    }

    #[test]
    fn vi_rejects_shape_mismatch() {
        pyo3::prepare_freethreaded_python();
        let a = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let err = compute_variation_of_information(&a, &b, 8).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn cmi_identical_images_is_nonnegative() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let img = make_image(v, [4, 4, 4]);
        let cmi = compute_conditional_mutual_information(&img, &img, &img, 8).unwrap();
        assert!(cmi >= 0.0, "CMI must be ≥ 0, got {cmi}");
    }

    #[test]
    fn cmi_rejects_shape_mismatch() {
        pyo3::prepare_freethreaded_python();
        let a = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let err = compute_conditional_mutual_information(&a, &b, &a, 8).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn ii_identical_is_positive() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let img = make_image(v, [4, 4, 4]);
        let ii = compute_interaction_information(&img, &img, &img, 8).unwrap();
        assert!(ii > 0.0, "II(X;X;X) must be positive, got {ii}");
    }

    #[test]
    fn mvi_identical_images_is_zero() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let img = make_image(v, [4, 4, 4]);
        // PyRef requires Python object protocol — test via direct slice logic instead
        let slices: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let mvi = multivariate_vi_slices(
            &[slices.as_slice(), slices.as_slice(), slices.as_slice()],
            8,
        )
        .unwrap();
        assert!(mvi.abs() < 1e-9, "MVI(X,X,X) must be 0, got {mvi}");
        let _ = img; // ensure make_image is used
    }

    #[test]
    fn mvi_rejects_single_image() {
        pyo3::prepare_freethreaded_python();
        let slices: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let err = multivariate_vi_slices(&[slices.as_slice()], 8).unwrap_err();
        assert!(err.to_string().contains("2 channels"), "expected 2-channel error: {err}");
    }
}
