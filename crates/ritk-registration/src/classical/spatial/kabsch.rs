//! Kabsch SVD rigid-body alignment and Fiducial Registration Error.
//!
//! # Algorithm
//! Given centred point sets P_f (fixed) and P_m (moving), compute:
//!   H = P_m^T · P_f
//!   H = U · Σ · V^T  (SVD)
//!   R = V · U^T       (optimal rotation; reflect if det < 0)
//!
//! Reference: Kabsch (1976), *Acta Crystallogr.* A32:922–923.

use nalgebra::{Matrix3, Vector3};
use ndarray::Array2;

use super::error::SpatialError;

/// Compute optimal rigid-body rotation using the Kabsch SVD algorithm.
///
/// Both `fixed_centered` and `moving_centered` must be (N×3) with identical N.
/// Returns a row-major 9-element rotation matrix.
#[rustfmt::skip]
pub(crate) fn kabsch_algorithm(
    fixed_centered: &Array2<f64>,
    moving_centered: &Array2<f64>,
) -> Result<[f64; 9], SpatialError> {
    if fixed_centered.nrows() != moving_centered.nrows() {
        return Err(SpatialError::InvalidPointSet(
            "Point sets must have same number of rows".to_string(),
        ));
    }
    if fixed_centered.ncols() != 3 || moving_centered.ncols() != 3 {
        return Err(SpatialError::InvalidPointSet(
            "Points must have 3 columns".to_string(),
        ));
    }

    let mut h = Matrix3::zeros();
    for i in 0..fixed_centered.nrows() {
        let pf = fixed_centered.row(i);
        let pm = moving_centered.row(i);
        h += Matrix3::new(
            pm[0] * pf[0], pm[0] * pf[1], pm[0] * pf[2],
            pm[1] * pf[0], pm[1] * pf[1], pm[1] * pf[2],
            pm[2] * pf[0], pm[2] * pf[1], pm[2] * pf[2],
        );
    }

    let svd = h.svd(true, true);
    let u   = svd.u.ok_or_else(|| SpatialError::SvdConvergence("U not found".to_string()))?;
    let v_t = svd.v_t.ok_or_else(|| SpatialError::SvdConvergence("V not found".to_string()))?;

    let mut r = v_t.transpose() * u.transpose();

    if r.determinant() < 0.0 {
        let mut v_corrected = v_t.transpose();
        v_corrected.set_column(2, &(-v_t.transpose().column(2)));
        r = v_corrected * u.transpose();
    }

    Ok([
        r[(0, 0)], r[(0, 1)], r[(0, 2)],
        r[(1, 0)], r[(1, 1)], r[(1, 2)],
        r[(2, 0)], r[(2, 1)], r[(2, 2)],
    ])
}

/// Compute Fiducial Registration Error (FRE).
///
/// FRE = sqrt(mean‖R·p_moving + t − p_fixed‖²) over N landmarks.
pub(crate) fn compute_fre(
    fixed: &Array2<f64>,
    moving: &Array2<f64>,
    rotation: &[f64; 9],
    translation: &[f64; 3],
) -> f64 {
    let r = Matrix3::new(
        rotation[0],
        rotation[1],
        rotation[2],
        rotation[3],
        rotation[4],
        rotation[5],
        rotation[6],
        rotation[7],
        rotation[8],
    );
    let t = Vector3::new(translation[0], translation[1], translation[2]);

    let mut sum_sq = 0.0_f64;
    for i in 0..fixed.nrows() {
        let pf = Vector3::new(fixed.row(i)[0], fixed.row(i)[1], fixed.row(i)[2]);
        let pm = Vector3::new(moving.row(i)[0], moving.row(i)[1], moving.row(i)[2]);
        let err = pf - (r * pm + t);
        sum_sq += err.dot(&err);
    }
    (sum_sq / fixed.nrows() as f64).sqrt()
}
