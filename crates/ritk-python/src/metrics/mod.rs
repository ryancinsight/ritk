//! Python-exposed image similarity and information-theoretic metrics.
//!
//! # Functions
//! - `compute_entropy`: marginal entropy H(X).
//! - `compute_joint_entropy`: joint entropy H(X,Y).
//! - `compute_symmetric_uncertainty`: SU = 2·MI/(H(X)+H(Y)) ∈ \[0,1\].
//! - `compute_mse`: mean squared error.
//! - `compute_ncc`: normalized cross-correlation (Pearson r).
//! - `compute_mutual_information`: histogram-based MI (mattes / standard / normalized).
//! - `compute_conditional_mutual_information`: I(X;Y|Z) histogram-based CMI.
//! - `compute_interaction_information`: II(X;Y;Z) interaction information (McGill 1954).
//! - `compute_total_correlation`: multivariate MI (total correlation) over N channels.
//! - `compute_dual_total_correlation`: DTC(X₁,…,Xₙ) dual total correlation (Han 1978).
//! - `compute_o_information`: Ω(X₁,…,Xₙ) O-information (Rosas 2019).
//! - `compute_variation_of_information`: VI = H(X|Y) + H(Y|X).
//! - `compute_multivariate_variation_of_information`: average pairwise VI over N channels.
//!
//! # Mathematical foundations
//! - Entropy  H(X)  = −Σₖ p(k) log p(k)
//! - H(X,Y)         = −Σⱼₖ p(j,k) log p(j,k)
//! - SU             = 2·I(X;Y) / (H(X) + H(Y))         (Fayyad & Irani 1993)
//! - MSE            = Σ(aᵢ−bᵢ)² / N
//! - NCC            = Σ(aᵢ−ā)(bᵢ−bÌ„) / (N·σ_a·σ_b + ε)
//! - MI             = H(A) + H(B) − H(A,B)
//! - CMI            = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)
//! - II             = I(X;Y) − I(X;Y|Z)                (McGill 1954)
//! - TC             = Σᵢ H(Xᵢ) − H(X₁,...,Xₙ)          (Watanabe 1960)
//! - DTC            = Σᵢ H(X₁,...,Xₙ\Xᵢ) − (n−1)·H(X₁,...,Xₙ)  (Han 1978)
//! - Ω              = TC − DTC                          (Rosas 2019)
//! - VI             = H(X) + H(Y) − 2·I(X,Y)            (Meilă 2003)
//! - VI_n           = (2/n(n−1)) · Σ_{i<j} VI(Xᵢ,Xⱼ)

mod cmi;
mod image_batch;
mod mi;
mod mse;
mod multivariate_vi;
mod ncc;
mod o_information;
mod total_correlation;
mod variation_of_information;

use pyo3::prelude::*;

pub use cmi::{compute_conditional_mutual_information, compute_interaction_information};
pub use mi::{
    compute_entropy, compute_joint_entropy, compute_mutual_information,
    compute_symmetric_uncertainty,
};
pub use mse::compute_mse;
pub use multivariate_vi::compute_multivariate_variation_of_information;
pub use ncc::compute_ncc;
pub use o_information::{compute_dual_total_correlation, compute_o_information};
pub use total_correlation::compute_total_correlation;
pub use variation_of_information::compute_variation_of_information;

// Test-only slice-level bindings in parent scope so `use super::*` in the
// tests child module picks them up without re-exporting beyond pub(super).
#[cfg(test)]
use cmi::{cmi_slices, ii_slices};
#[cfg(test)]
use mi::mi_slices;
#[cfg(test)]
use multivariate_vi::multivariate_vi_slices;
#[cfg(test)]
use o_information::{dtc_slices, oi_slices};

// ── module registration ───────────────────────────────────────────────────────

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "metrics")?;
    m.add_function(wrap_pyfunction!(compute_entropy, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_joint_entropy, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_symmetric_uncertainty, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_mse, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_ncc, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_mutual_information, &m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_conditional_mutual_information,
        &m
    )?)?;
    m.add_function(wrap_pyfunction!(compute_interaction_information, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_total_correlation, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_dual_total_correlation, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_o_information, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_variation_of_information, &m)?)?;
    m.add_function(wrap_pyfunction!(
        compute_multivariate_variation_of_information,
        &m
    )?)?;
    parent.add_submodule(&m)?;
    Ok(())
}

// ── integration tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::{vec_to_image, PyImage};
    use std::sync::Arc;

    fn make_image(values: Vec<f32>, shape: [usize; 3]) -> PyImage {
        PyImage {
            inner: Arc::new(vec_to_image(
                values,
                shape,
                ritk_core::spatial::Point::new([0.0; 3]),
                ritk_core::spatial::Spacing::new([1.0; 3]),
                ritk_core::spatial::Direction::identity(),
            )),
        }
    }

    #[test]
    fn mse_rejects_shape_mismatch() {
        pyo3::prepare_freethreaded_python();
        let a = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let err = Python::with_gil(|py| compute_mse(py, &a, &b)).unwrap_err();
        assert!(err.to_string().contains("shape mismatch"));
    }

    #[test]
    fn ncc_rejects_shape_mismatch() {
        pyo3::prepare_freethreaded_python();
        let a = make_image(vec![1.0, 2.0], [1, 1, 2]);
        let b = make_image(vec![1.0, 2.0, 3.0], [1, 1, 3]);
        let err = Python::with_gil(|py| compute_ncc(py, &a, &b)).unwrap_err();
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
        assert!(
            tc.to_string().contains("empty"),
            "expected empty error: {tc}"
        );
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
        let _ = img;
    }

    #[test]
    fn dtc_two_identical_images_non_negative() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let dtc = dtc_slices(&[v.as_slice(), v.as_slice()], 8).unwrap();
        assert!(dtc >= 0.0, "DTC must be ≥ 0, got {dtc}");
    }

    #[test]
    fn oi_two_identical_images_is_zero() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let oi = oi_slices(&[v.as_slice(), v.as_slice()], 8).unwrap();
        assert!(oi.abs() < 1e-9, "Ω(X,X) must be 0 for n=2, got {oi}");
    }

    #[test]
    fn entropy_nonconstant_image_is_positive() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let img = make_image(v, [4, 4, 4]);
        let h = compute_entropy(&img, 16).unwrap();
        assert!(h > 0.0, "H(X) must be positive for non-constant X, got {h}");
    }

    #[test]
    fn joint_entropy_geq_marginal_via_pyfunction() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let w: Vec<f32> = (0..64).map(|x| ((x / 8) % 8) as f32).collect();
        let img_v = make_image(v, [4, 4, 4]);
        let img_w = make_image(w, [4, 4, 4]);
        let h_xy = compute_joint_entropy(&img_v, &img_w, 16).unwrap();
        let h_x = compute_entropy(&img_v, 16).unwrap();
        assert!(
            h_xy >= h_x - 1e-9,
            "H(X,Y) must be >= H(X), got {h_xy:.6} vs {h_x:.6}"
        );
    }

    #[test]
    fn symmetric_uncertainty_self_is_one_via_pyfunction() {
        pyo3::prepare_freethreaded_python();
        let v: Vec<f32> = (0..64).map(|x| (x % 8) as f32).collect();
        let img = make_image(v, [4, 4, 4]);
        let su = compute_symmetric_uncertainty(&img, &img, 16).unwrap();
        assert!((su - 1.0).abs() < 1e-9, "SU(X,X) must be 1.0, got {su}");
    }

    // Cross-verify: mi_slices / cmi_slices / ii_slices still accessible from this scope.
    #[test]
    fn mi_slices_self_positive() {
        let a: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let mi = mi_slices(&a, &a, 16, "standard").unwrap();
        assert!(mi > 0.0, "mi_slices(A,A) must be positive, got {mi}");
    }

    #[test]
    fn cmi_ii_slices_accessible() {
        let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
        let y: Vec<f32> = (0..64).map(|i| ((i / 8) % 8) as f32).collect();
        let z = vec![2.0_f32; 64];
        let _cmi = cmi_slices(&x, &y, &z, 8).unwrap();
        let _ii = ii_slices(&x, &y, &z, 8).unwrap();
    }
}
