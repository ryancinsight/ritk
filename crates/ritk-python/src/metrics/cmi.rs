//! Conditional Mutual Information and Interaction Information pyfunction wrappers.
//!
//! Delegates to `ritk_core::statistics::information`:
//! - I(X;Y|Z) = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z)   (conditional MI)
//! - II(X;Y;Z) = I(X;Y) − I(X;Y|Z)                     (interaction info, McGill 1954)

use anyhow::Result;
use pyo3::prelude::*;
use ritk_core::statistics::information::{
    conditional_mutual_information as core_cmi, interaction_information as core_ii,
};

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, PyImage};

/// I(X;Y|Z) via `ritk_core::statistics::information::conditional_mutual_information`.
pub(super) fn cmi_slices(x: &[f32], y: &[f32], z: &[f32], num_bins: usize) -> Result<f64> {
    core_cmi(x, y, z, num_bins)
}

/// II(X;Y;Z) via `ritk_core::statistics::information::interaction_information`.
pub(super) fn ii_slices(x: &[f32], y: &[f32], z: &[f32], num_bins: usize) -> Result<f64> {
    core_ii(x, y, z, num_bins)
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
) -> RitkResult<f64> {
    let (x, shape_x) = image_to_vec(&x_img.inner);
    let (y, shape_y) = image_to_vec(&y_img.inner);
    let (z, shape_z) = image_to_vec(&z_img.inner);
    if shape_x != shape_y || shape_x != shape_z {
        return Err(RitkPyError::value(format!(
            "shape mismatch: x {:?}, y {:?}, z {:?}",
            shape_x, shape_y, shape_z
        )));
    }
    if num_bins < 2 {
        return Err(RitkPyError::value("num_bins must be >= 2"));
    }
    cmi_slices(&x, &y, &z, num_bins)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
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
) -> RitkResult<f64> {
    let (x, shape_x) = image_to_vec(&x_img.inner);
    let (y, shape_y) = image_to_vec(&y_img.inner);
    let (z, shape_z) = image_to_vec(&z_img.inner);
    if shape_x != shape_y || shape_x != shape_z {
        return Err(RitkPyError::value(format!(
            "shape mismatch: x {:?}, y {:?}, z {:?}",
            shape_x, shape_y, shape_z
        )));
    }
    if num_bins < 2 {
        return Err(RitkPyError::value("num_bins must be >= 2"));
    }
    ii_slices(&x, &y, &z, num_bins)
        .map_err(|e| RitkPyError::runtime(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cmi_constant_z_equals_mi_slice() {
        // I(X;Y|const) = I(X;Y) (validated analytically in ritk-core; cross-check slice path).
        use ritk_core::statistics::information::mutual_information as core_mi;
        let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
        let y: Vec<f32> = (0..64).map(|i| ((i / 8) % 8) as f32).collect();
        let z_const = vec![2.0_f32; 64];
        let cmi = cmi_slices(&x, &y, &z_const, 8).unwrap();
        let mi = core_mi(&x, &y, 8).unwrap();
        assert!(
            (cmi - mi).abs() < 1e-9,
            "CMI(X,Y|const)={cmi:.9} must equal MI(X,Y)={mi:.9}"
        );
    }

    #[test]
    fn ii_constant_z_is_zero_slice() {
        let x: Vec<f32> = (0..64).map(|i| (i % 8) as f32).collect();
        let y: Vec<f32> = (0..64).map(|i| ((i / 8) % 8) as f32).collect();
        let z_const = vec![2.0_f32; 64];
        let ii = ii_slices(&x, &y, &z_const, 8).unwrap();
        assert!(ii.abs() < 1e-9, "II(X;Y;const)={ii:.10} must be 0");
    }
}
