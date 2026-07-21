//! Kabsch SVD rigid-body alignment and Fiducial Registration Error.
//!
//! # Algorithm
//! Given centred point sets P_f (fixed) and P_m (moving), compute:
//!   H = P_m^T · P_f
//!   H = U · Σ · V^T  (SVD)
//!   R = V · U^T       (optimal rotation; reflect if det < 0)
//!
//! Reference: Kabsch (1976), *Acta Crystallogr.* A32:922–923.

use leto::{Array2, FixedMatrix, FixedVector};
use leto_ops::svd_rank_revealing;

use super::error::SpatialError;

type Matrix3 = FixedMatrix<f64, 3, 3>;
type Vector3 = FixedVector<f64, 3>;

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
    if point_sets_are_exactly_equal(fixed_centered, moving_centered) {
        return Ok([
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]);
    }

    let mut h = Matrix3::zeros();
    for i in 0..shape_f[0] {
        let pf0 = *fixed_centered.get([i, 0]).expect("valid index");
        let pf1 = *fixed_centered.get([i, 1]).expect("valid index");
        let pf2 = *fixed_centered.get([i, 2]).expect("valid index");

        let pm0 = *moving_centered.get([i, 0]).expect("valid index");
        let pm1 = *moving_centered.get([i, 1]).expect("valid index");
        let pm2 = *moving_centered.get([i, 2]).expect("valid index");

        h += Matrix3::from_rows([
            [pm0 * pf0, pm0 * pf1, pm0 * pf2],
            [pm1 * pf0, pm1 * pf1, pm1 * pf2],
            [pm2 * pf0, pm2 * pf1, pm2 * pf2],
        ]);
    }

    let h_array = Array2::from_vec(
        [3, 3],
        vec![
            h[(0, 0)], h[(0, 1)], h[(0, 2)],
            h[(1, 0)], h[(1, 1)], h[(1, 2)],
            h[(2, 0)], h[(2, 1)], h[(2, 2)],
        ],
    )
    .map_err(|err| SpatialError::SvdConvergence(format!("Kabsch covariance layout: {err}")))?;
    let svd = svd_rank_revealing(&h_array.view())
        .map_err(|err| SpatialError::SvdConvergence(format!("Kabsch SVD failed: {err}")))?;

    let u = matrix_from_svd_columns(&svd.left_singular_vectors)?;
    let mut v = matrix_from_svd_columns(&svd.right_singular_vectors)?;

    let mut r = v * u.transpose();
    if r.determinant() < 0.0 {
        v.set_column(2, -Vector3::new([v[(0, 2)], v[(1, 2)], v[(2, 2)]]));
        r = v * u.transpose();
    }

    Ok([
        r[(0, 0)], r[(0, 1)], r[(0, 2)],
        r[(1, 0)], r[(1, 1)], r[(1, 2)],
        r[(2, 0)], r[(2, 1)], r[(2, 2)],
    ])
}

fn point_sets_are_exactly_equal(lhs: &Array2<f64>, rhs: &Array2<f64>) -> bool {
    lhs.shape() == rhs.shape()
        && lhs
            .iter()
            .zip(rhs.iter())
            .all(|(&left, &right)| left == right)
}

fn matrix_from_svd_columns(matrix: &Array2<f64>) -> Result<Matrix3, SpatialError> {
    let shape = matrix.shape();
    if shape != [3, 3] {
        return Err(SpatialError::SvdConvergence(format!(
            "Kabsch SVD factor shape must be [3, 3], got {shape:?}"
        )));
    }

    Ok(Matrix3::from_rows([
        [
            *matrix.get([0, 0]).expect("valid index"),
            *matrix.get([0, 1]).expect("valid index"),
            *matrix.get([0, 2]).expect("valid index"),
        ],
        [
            *matrix.get([1, 0]).expect("valid index"),
            *matrix.get([1, 1]).expect("valid index"),
            *matrix.get([1, 2]).expect("valid index"),
        ],
        [
            *matrix.get([2, 0]).expect("valid index"),
            *matrix.get([2, 1]).expect("valid index"),
            *matrix.get([2, 2]).expect("valid index"),
        ],
    ]))
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
    let r = Matrix3::from_rows([
        [rotation[0], rotation[1], rotation[2]],
        [rotation[3], rotation[4], rotation[5]],
        [rotation[6], rotation[7], rotation[8]],
    ]);
    let t = Vector3::new([translation[0], translation[1], translation[2]]);

    let shape = fixed.shape();
    let mut sum_sq = 0.0_f64;
    for i in 0..shape[0] {
        let pf = Vector3::new([
            *fixed.get([i, 0]).expect("valid index"),
            *fixed.get([i, 1]).expect("valid index"),
            *fixed.get([i, 2]).expect("valid index"),
        ]);
        let pm = Vector3::new([
            *moving.get([i, 0]).expect("valid index"),
            *moving.get([i, 1]).expect("valid index"),
            *moving.get([i, 2]).expect("valid index"),
        ]);
        let err = pf - (r * pm + t);
        sum_sq += err.dot(&err);
    }
    (sum_sq / shape[0] as f64).sqrt()
}
