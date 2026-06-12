//! Python-exposed image statistics, comparison, normalization, and Jacobian functions.
//!
//! All functions delegate to `ritk_statistics` implementations (SSOT).
//!
//! # Submodules
//! - `descriptive`:    Descriptive statistics, image comparison, noise estimation, label stats.
//! - `label_overlap`:  Per-label overlap measures (Dice, Jaccard, sensitivity, specificity, etc.).
//! - `normalization`:  Min-max, z-score, histogram matching, white stripe, Nyul-Udupa.
//! - `jacobian`:       Jacobian determinant of displacement fields and analysis.

mod descriptive;
mod jacobian;
mod label_overlap;
mod label_shape_extended;
mod normalization;

pub use descriptive::*;
pub use jacobian::*;
pub use label_overlap::*;
pub use label_shape_extended::*;
pub use normalization::*;

use pyo3::prelude::*;

/// Register the `statistics` submodule and all 18 exposed functions into `parent`.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "statistics")?;
    m.add_function(wrap_pyfunction!(compute_statistics, &m)?)?;
    m.add_function(wrap_pyfunction!(masked_statistics, &m)?)?;
    m.add_function(wrap_pyfunction!(dice_coefficient, &m)?)?;
    m.add_function(wrap_pyfunction!(hausdorff_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(mean_surface_distance, &m)?)?;
    m.add_function(wrap_pyfunction!(psnr, &m)?)?;
    m.add_function(wrap_pyfunction!(ssim, &m)?)?;
    m.add_function(wrap_pyfunction!(estimate_noise, &m)?)?;
    m.add_function(wrap_pyfunction!(minmax_normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(minmax_normalize_range, &m)?)?;
    m.add_function(wrap_pyfunction!(zscore_normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(histogram_match, &m)?)?;
    m.add_function(wrap_pyfunction!(white_stripe_normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(nyul_udupa_normalize, &m)?)?;
    m.add_function(wrap_pyfunction!(compute_label_intensity_statistics, &m)?)?;
    m.add_function(wrap_pyfunction!(jacobian_determinant, &m)?)?;
    m.add_function(wrap_pyfunction!(analyze_jacobian, &m)?)?;
    m.add_function(wrap_pyfunction!(label_overlap_measures, &m)?)?;
    m.add_function(wrap_pyfunction!(extended_label_shape_statistics_py, &m)?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
