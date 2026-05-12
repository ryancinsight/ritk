//! Python-exposed image statistics, comparison, and normalization functions.
//!
//! All functions delegate to `ritk_core::statistics` implementations (SSOT).
//!
//! # Submodules
//! - `descriptive`:    Descriptive statistics, image comparison, noise estimation, label stats.
//! - `normalization`:  Min-max, z-score, histogram matching, white stripe, Nyul-Udupa.

mod descriptive;
mod normalization;

pub use descriptive::*;
pub use normalization::*;

use pyo3::prelude::*;

/// Register the `statistics` submodule and all 15 exposed functions into `parent`.
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
    parent.add_submodule(&m)?;
    Ok(())
}
