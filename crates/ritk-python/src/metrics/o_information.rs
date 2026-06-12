//! O-Information and Dual Total Correlation pyfunction wrappers.
//!
//! Delegates to `ritk_statistics::information`:
//! - DTC(X₁,...,Xₙ) = Σᵢ H(X₁,...,Xₙ\Xᵢ) − (n−1)·H(X₁,...,Xₙ)  (Han 1978)
//! - Ω(X₁,...,Xₙ) = TC − DTC                                        (Rosas 2019)

use anyhow::Result;
use pyo3::prelude::*;
use ritk_statistics::information::{
    dual_total_correlation as core_dtc, o_information as core_oi,
};

use crate::errors::{RitkPyError, RitkResult};
use crate::image::PyImage;
use crate::metrics::image_batch::collect_image_vectors;

pub(super) fn dtc_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_dtc(channels, num_bins)
}

pub(super) fn oi_slices(channels: &[&[f32]], num_bins: usize) -> Result<f64> {
    core_oi(channels, num_bins)
}

/// Dual Total Correlation over N images (Han 1978).
///
/// All images must have identical shapes. Returns DTC ≥ 0.
///
/// # Formula (Han 1978)
/// DTC(X₁,...,Xₙ) = Σᵢ H(X₁,...,Xₙ\Xᵢ) − (n−1)·H(X₁,...,Xₙ)
///
/// For n=2: DTC(X,Y) = I(X;Y) (equals total correlation).
///
/// # Arguments
/// - `images`: list of PyImage objects (n ≥ 2).
/// - `num_bins`: histogram bins per channel (2 ≤ B ≤ 64, default 32).
#[pyfunction]
#[pyo3(signature = (images, num_bins=32))]
pub fn compute_dual_total_correlation(
    images: Vec<PyRef<PyImage>>,
    num_bins: usize,
) -> RitkResult<f64> {
    if images.len() < 2 {
        return Err(RitkPyError::value(format!(
            "at least 2 images required, got {}",
            images.len()
        )));
    }
    let (vectors, _) =
        collect_image_vectors(&images).map_err(|e| RitkPyError::value(e.to_string()))?;
    if !(2..=64).contains(&num_bins) {
        return Err(RitkPyError::value(format!(
            "num_bins must be in [2, 64], got {num_bins}"
        )));
    }
    let slices: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    dtc_slices(&slices, num_bins).map_err(|e| RitkPyError::runtime(e.to_string()))
}

/// O-Information over N images (Rosas et al. 2019).
///
/// All images must have identical shapes. Result may be negative.
///
/// # Formula (Rosas 2019)
/// Ω(X₁,...,Xₙ) = TC(X₁,...,Xₙ) − DTC(X₁,...,Xₙ)
///
/// - Ω > 0: redundancy-dominated system.
/// - Ω < 0: synergy-dominated system.
/// - Ω = 0: balanced / independent.
///
/// For n=3: Ω(X,Y,Z) = II(X;Y;Z) (generalises interaction information).
/// For n=2: Ω(X,Y) = 0 always (TC=DTC=I(X;Y)).
///
/// # Arguments
/// - `images`: list of PyImage objects (n ≥ 2).
/// - `num_bins`: histogram bins per channel (2 ≤ B ≤ 64, default 32).
#[pyfunction]
#[pyo3(signature = (images, num_bins=32))]
pub fn compute_o_information(images: Vec<PyRef<PyImage>>, num_bins: usize) -> RitkResult<f64> {
    if images.len() < 2 {
        return Err(RitkPyError::value(format!(
            "at least 2 images required, got {}",
            images.len()
        )));
    }
    let (vectors, _) =
        collect_image_vectors(&images).map_err(|e| RitkPyError::value(e.to_string()))?;
    if !(2..=64).contains(&num_bins) {
        return Err(RitkPyError::value(format!(
            "num_bins must be in [2, 64], got {num_bins}"
        )));
    }
    let slices: Vec<&[f32]> = vectors.iter().map(|v| v.as_slice()).collect();
    oi_slices(&slices, num_bins).map_err(|e| RitkPyError::runtime(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_statistics::information::o_information_direct as core_oi_direct;

    fn ramp(n: usize, modulus: usize) -> Vec<f32> {
        (0..n).map(|i| (i % modulus) as f32).collect()
    }

    #[test]
    fn dtc_two_identical_slices_is_non_negative() {
        let a = ramp(64, 8);
        let dtc = dtc_slices(&[a.as_slice(), a.as_slice()], 8).unwrap();
        assert!(dtc >= 0.0, "DTC must be ≥ 0, got {dtc}");
    }

    #[test]
    fn oi_two_slices_is_zero() {
        let a = ramp(64, 8);
        let b = ramp(64, 4);
        let oi = oi_slices(&[a.as_slice(), b.as_slice()], 8).unwrap();
        assert!(oi.abs() < 1e-9, "Ω(X,Y) must be 0 for n=2, got {oi}");
    }

    #[test]
    fn oi_direct_matches_oi_three_channels() {
        let a = ramp(128, 8);
        let b = ramp(128, 4);
        let c = ramp(128, 6);
        let channels: &[&[f32]] = &[a.as_slice(), b.as_slice(), c.as_slice()];
        let oi = oi_slices(channels, 8).unwrap();
        let oi_d = core_oi_direct(channels, 8).unwrap();
        assert!(
            (oi - oi_d).abs() < 1e-9,
            "oi={oi:.10} must equal oi_direct={oi_d:.10}"
        );
    }
}
