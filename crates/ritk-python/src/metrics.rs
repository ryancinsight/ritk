//! Python-exposed image similarity metrics.
//!
//! # Functions
//! - `compute_mse`: mean squared error between two registered images.
//! - `compute_ncc`: normalized cross-correlation (Pearson r) between two images.
//! - `compute_mutual_information`: histogram-based MI with three variants.
//!
//! # Mathematical foundations
//! - MSE  = Σ(a_i − b_i)² / N
//! - NCC  = Σ(a_i − ā)(b_i − b̄) / (N · σ_a · σ_b + ε)
//! - MI   = H(A) + H(B) − H(A,B)  where H is the Shannon entropy.
//!   - "mattes"     : bilinear soft-binning into joint histogram (Mattes 2003).
//!   - "standard"   : nearest-bin hard assignment into joint histogram.
//!   - "normalized" : 2·MI / (H(A) + H(B))  (Studholme 1999).

use anyhow::{bail, Result};
use pyo3::prelude::*;

use crate::image::PyImage;
use crate::image::image_to_vec;

// ── public PyO3 functions ─────────────────────────────────────────────────────

/// Mean squared error between two images.
///
/// Both images must have identical shapes. Returns the scalar MSE as a float.
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
/// NCC = Σ(a_i − ā)(b_i − b̄) / (N · σ_a · σ_b + ε)
/// where ε = 1e-10 guards against zero-variance inputs.
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
/// - `num_bins`: number of histogram bins per axis (default 64).
/// - `variant`:  `"mattes"` (default), `"standard"`, or `"normalized"`.
///
/// # Formula (all variants)
/// MI = H(A) + H(B) − H(A,B)
///
/// `"mattes"` uses bilinear soft-bin assignment; `"standard"` uses
/// nearest-bin hard assignment; `"normalized"` returns 2·MI/(H(A)+H(B)).
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

// ── private slice-level helpers ───────────────────────────────────────────────

/// MSE = Σ(a_i − b_i)² / N
fn mse_slices(a: &[f32], b: &[f32]) -> f64 {
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

/// Pearson r = cov(a,b) / (std_a · std_b + ε).
fn ncc_slices(a: &[f32], b: &[f32]) -> Result<f64> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        bail!("cannot compute NCC of empty images");
    }
    let n_f = n as f64;
    let mean_a: f64 = a.iter().map(|&x| x as f64).sum::<f64>() / n_f;
    let mean_b: f64 = b.iter().map(|&x| x as f64).sum::<f64>() / n_f;

    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        let da = ai as f64 - mean_a;
        let db = bi as f64 - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    let std_a = (var_a / n_f).sqrt();
    let std_b = (var_b / n_f).sqrt();
    const EPS: f64 = 1e-10;
    Ok(cov / (n_f * (std_a * std_b + EPS)))
}

/// Histogram-based mutual information with configurable binning strategy.
///
/// "mattes":     bilinear soft-binning per Mattes et al. (2003).
/// "standard":   nearest-bin hard assignment.
/// "normalized": hard-bin, returns 2·MI / (H(A) + H(B)).
fn mi_slices(a: &[f32], b: &[f32], num_bins: usize, variant: &str) -> Result<f64> {
    debug_assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 {
        bail!("cannot compute MI of empty images");
    }
    let bins = num_bins;

    // Compute intensity range for each image (guard against degenerate constant inputs).
    let (a_min, a_max) = min_max(a);
    let (b_min, b_max) = min_max(b);
    let a_range = if (a_max - a_min).abs() < f32::EPSILON { 1.0_f64 } else { (a_max - a_min) as f64 };
    let b_range = if (b_max - b_min).abs() < f32::EPSILON { 1.0_f64 } else { (b_max - b_min) as f64 };

    // Allocate joint histogram and marginals.
    let mut joint = vec![0.0_f64; bins * bins];
    let mut hist_a = vec![0.0_f64; bins];
    let mut hist_b = vec![0.0_f64; bins];

    match variant {
        "mattes" => {
            // Bilinear soft-binning: each voxel distributes its weight to up to
            // 4 neighboring bins proportionally (Mattes 2003, eq. 4).
            let scale_a = (bins - 1) as f64 / a_range;
            let scale_b = (bins - 1) as f64 / b_range;
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let fa = ((ai as f64 - a_min as f64) * scale_a).clamp(0.0, (bins - 1) as f64);
                let fb = ((bi as f64 - b_min as f64) * scale_b).clamp(0.0, (bins - 1) as f64);
                let ia = fa.floor() as usize;
                let ib = fb.floor() as usize;
                let wa1 = fa - ia as f64;
                let wb1 = fb - ib as f64;
                let wa0 = 1.0 - wa1;
                let wb0 = 1.0 - wb1;

                let ia1 = (ia + 1).min(bins - 1);
                let ib1 = (ib + 1).min(bins - 1);

                joint[ia * bins + ib]   += wa0 * wb0;
                joint[ia * bins + ib1]  += wa0 * wb1;
                joint[ia1 * bins + ib]  += wa1 * wb0;
                joint[ia1 * bins + ib1] += wa1 * wb1;

                hist_a[ia]  += wa0;
                hist_a[ia1] += wa1;
                hist_b[ib]  += wb0;
                hist_b[ib1] += wb1;
            }
        }
        "standard" | "normalized" => {
            // Hard nearest-bin assignment.
            let scale_a = (bins - 1) as f64 / a_range;
            let scale_b = (bins - 1) as f64 / b_range;
            for (&ai, &bi) in a.iter().zip(b.iter()) {
                let ia = (((ai as f64 - a_min as f64) * scale_a) as usize).min(bins - 1);
                let ib = (((bi as f64 - b_min as f64) * scale_b) as usize).min(bins - 1);
                joint[ia * bins + ib] += 1.0;
                hist_a[ia] += 1.0;
                hist_b[ib] += 1.0;
            }
        }
        _ => unreachable!("variant validated before mi_slices"),
    }

    // Normalize to probability distributions.
    let total = n as f64;
    for v in joint.iter_mut() { *v /= total; }
    for v in hist_a.iter_mut() { *v /= total; }
    for v in hist_b.iter_mut() { *v /= total; }

    // H(A) = −Σ p_a · log(p_a)
    let h_a: f64 = hist_a.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    // H(B) = −Σ p_b · log(p_b)
    let h_b: f64 = hist_b.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
    // H(A,B) = −Σ p_ab · log(p_ab)
    let h_ab: f64 = joint.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();

    let mi = h_a + h_b - h_ab;

    if variant == "normalized" {
        let denom = h_a + h_b;
        if denom < 1e-15 {
            return Ok(0.0);
        }
        Ok(2.0 * mi / denom)
    } else {
        Ok(mi)
    }
}

/// Returns `(min, max)` of a non-empty f32 slice.
fn min_max(data: &[f32]) -> (f32, f32) {
    debug_assert!(!data.is_empty());
    data.iter().fold((data[0], data[0]), |(mn, mx), &v| {
        (mn.min(v), mx.max(v))
    })
}

// ── module registration ───────────────────────────────────────────────────────

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "metrics")?;
    m.add_function(wrap_pyfunction!(compute_mse, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_ncc, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_mutual_information, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

// ── unit tests ────────────────────────────────────────────────────────────────

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
    fn mse_identical_images_returns_zero() {
        let img = make_image(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]);
        let result = mse_slices(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!((result).abs() < 1e-12, "MSE of identical images must be 0, got {result}");
        drop(img);
    }

    #[test]
    fn mse_known_pair_is_analytically_correct() {
        // A = [0,1,2,3], B = [1,2,3,4]; differences = [−1,−1,−1,−1]; MSE = 4/4 = 1.0
        let a = [0.0_f32, 1.0, 2.0, 3.0];
        let b = [1.0_f32, 2.0, 3.0, 4.0];
        let result = mse_slices(&a, &b);
        assert!(
            (result - 1.0).abs() < 1e-12,
            "MSE([0,1,2,3],[1,2,3,4]) must be 1.0, got {result}"
        );
    }

    #[test]
    fn mse_rejects_shape_mismatch() {
        pyo3::prepare_freethreaded_python();
        let a = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let err = compute_mse(&a, &b).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("shape mismatch"), "expected shape mismatch error, got: {msg}");
    }

    #[test]
    fn ncc_identical_images_returns_one() {
        let v: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let result = ncc_slices(&v, &v).expect("NCC must succeed");
        assert!(
            (result - 1.0).abs() < 1e-10,
            "NCC of identical images must be 1.0, got {result}"
        );
    }

    #[test]
    fn ncc_anti_correlated_images_returns_negative_one() {
        let a: Vec<f32> = (1..=8).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=8).rev().map(|x| x as f32).collect();
        let result = ncc_slices(&a, &b).expect("NCC must succeed");
        assert!(
            (result + 1.0).abs() < 1e-10,
            "NCC of anti-correlated images must be −1.0, got {result}"
        );
    }

    #[test]
    fn mi_self_exceeds_constant() {
        // Analytical: MI(A,A) = H(A) > 0 for non-constant A.
        // MI(A, constant) = H(A) + H(const) − H(A,const) = H(A) + 0 − H(A) = 0.
        let a: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let b_const: Vec<f32> = vec![5.0_f32; 32];
        let mi_self = mi_slices(&a, &a, 16, "standard").expect("MI self must succeed");
        let mi_const = mi_slices(&a, &b_const, 16, "standard").expect("MI const must succeed");
        assert!(mi_self > 0.0, "MI(A,A) must be positive for non-constant A, got {mi_self}");
        assert!(
            mi_const.abs() < 1e-10,
            "MI(A,constant) must be 0 (H(B)=0), got {mi_const}"
        );
    }

    #[test]
    fn mi_shape_mismatch_errors() {
        pyo3::prepare_freethreaded_python();
        let a = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let err = compute_mutual_information(&a, &b, 32, "mattes").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("shape mismatch"), "expected shape mismatch error, got: {msg}");
    }

    #[test]
    fn mi_unknown_variant_errors() {
        pyo3::prepare_freethreaded_python();
        let img = make_image(vec![1.0, 2.0, 3.0, 4.0], [1, 2, 2]);
        let err = compute_mutual_information(&img, &img, 32, "bogus").unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("unknown variant") && msg.contains("bogus"),
            "expected unknown variant error, got: {msg}"
        );
    }

    #[test]
    fn mi_normalized_variant_bounded() {
        // Normalized MI ∈ [0, 1] for non-trivial inputs.
        let a: Vec<f32> = (0..64).map(|x| (x % 16) as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| ((x + 4) % 16) as f32).collect();
        let nmi = mi_slices(&a, &b, 16, "normalized").expect("NMI must succeed");
        assert!(
            (0.0..=1.0).contains(&nmi),
            "normalized MI must be in [0,1], got {nmi}"
        );
    }
}
