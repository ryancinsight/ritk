//! Kabsch SVD rigid-body alignment and Fiducial Registration Error.
//!
//! # Algorithm
//! Given centred point sets P_f (fixed) and P_m (moving), compute:
//!   H = P_m^T · P_f
//!   H = U · Σ · V^T  (SVD)
//!   R = V · U^T       (optimal rotation; reflect if det < 0)
//!
//! Reference: Kabsch (1976), *Acta Crystallogr.* A32:922–923.

use leto::Array2;
use nalgebra::{Matrix3, Vector3};

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
    let shape_f = fixed_centered.shape();
    let shape_m = moving_centered.shape();

    if shape_f[0] != shape_m[0] {
        return Err(SpatialError::InvalidPointSet(
            "Point sets must have same number of rows".to_string(),
        ));
    }
    if shape_f[1] != 3 || shape_m[1] != 3 {
        return Err(SpatialError::InvalidPointSet(
            "Points must have 3 columns".to_string(),
        ));
    }

    let mut h = Matrix3::zeros();
    for i in 0..shape_f[0] {
        let pf0 = *fixed_centered.get([i, 0]).unwrap();
        let pf1 = *fixed_centered.get([i, 1]).unwrap();
        let pf2 = *fixed_centered.get([i, 2]).unwrap();

        let pm0 = *moving_centered.get([i, 0]).unwrap();
        let pm1 = *moving_centered.get([i, 1]).unwrap();
        let pm2 = *moving_centered.get([i, 2]).unwrap();

        h += Matrix3::new(
            pm0 * pf0, pm0 * pf1, pm0 * pf2,
            pm1 * pf0, pm1 * pf1, pm1 * pf2,
            pm2 * pf0, pm2 * pf1, pm2 * pf2,
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

    let shape = fixed.shape();
    let mut sum_sq = 0.0_f64;
    for i in 0..shape[0] {
        let pf = Vector3::new(
            *fixed.get([i, 0]).unwrap(),
            *fixed.get([i, 1]).unwrap(),
            *fixed.get([i, 2]).unwrap(),
        );
        let pm = Vector3::new(
            *moving.get([i, 0]).unwrap(),
            *moving.get([i, 1]).unwrap(),
            *moving.get([i, 2]).unwrap(),
        );
        let err = pf - (r * pm + t);
        sum_sq += err.dot(&err);
    }
    (sum_sq / shape[0] as f64).sqrt()
}
