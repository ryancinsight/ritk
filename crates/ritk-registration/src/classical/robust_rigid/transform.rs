use leto::{Array2, FixedMatrix, FixedVector};

use super::super::error::{RegistrationError, Result};
use super::super::spatial::{
    build_homogeneous_matrix, center_points, compute_centroid, kabsch_algorithm,
};
use super::correspondence::RigidCorrespondence;
use super::limits::RANK_TOLERANCE;
use crate::types::AffineTransform;

type Matrix3 = FixedMatrix<f64, 3, 3>;
type Vector3 = FixedVector<f64, 3>;

pub(super) fn fit_indices(
    correspondences: &[RigidCorrespondence],
    indices: &[usize],
) -> Result<AffineTransform> {
    if indices.len() < 3 {
        return Err(RegistrationError::InvalidInput(format!(
            "rigid fitting needs at least three retained points, got {}",
            indices.len()
        )));
    }
    let value_count = indices.len().checked_mul(3).ok_or_else(|| {
        RegistrationError::InvalidInput("rigid coordinate count overflows usize".to_owned())
    })?;
    let mut fixed_values = Vec::new();
    fixed_values
        .try_reserve_exact(value_count)
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {value_count} fixed rigid coordinates: {error}"
            ))
        })?;
    let mut moving_values = Vec::new();
    moving_values
        .try_reserve_exact(value_count)
        .map_err(|error| {
            RegistrationError::InvalidInput(format!(
                "cannot allocate {value_count} moving rigid coordinates: {error}"
            ))
        })?;
    for &index in indices {
        let pair = correspondences
            .get(index)
            .expect("invariant: retained correspondence index came from this slice");
        fixed_values.extend_from_slice(&pair.source_mm);
        moving_values.extend_from_slice(&pair.target_mm);
    }
    let fixed = Array2::from_vec([indices.len(), 3], fixed_values).map_err(|error| {
        RegistrationError::NumericalFailure(format!(
            "cannot lay out fixed rigid correspondences: {error}"
        ))
    })?;
    let moving = Array2::from_vec([indices.len(), 3], moving_values).map_err(|error| {
        RegistrationError::NumericalFailure(format!(
            "cannot lay out moving rigid correspondences: {error}"
        ))
    })?;
    let fixed_centroid = compute_centroid(&fixed);
    let moving_centroid = compute_centroid(&moving);
    let fixed_centered = center_points(&fixed, &fixed_centroid);
    let moving_centered = center_points(&moving, &moving_centroid);
    ensure_non_collinear(&fixed_centered, "fixed")?;
    ensure_non_collinear(&moving_centered, "moving")?;

    // `kabsch_algorithm(target, source)` maps source to target. The public
    // correspondence convention here is fixed→moving.
    let rotation = kabsch_algorithm(&moving_centered, &fixed_centered)?;
    let matrix = Matrix3::from_rows([
        [rotation[0], rotation[1], rotation[2]],
        [rotation[3], rotation[4], rotation[5]],
        [rotation[6], rotation[7], rotation[8]],
    ]);
    let translation = moving_centroid - matrix * fixed_centroid;
    let transform =
        build_homogeneous_matrix(&rotation, &[translation[0], translation[1], translation[2]]);
    if transform.as_array().iter().all(|value| value.is_finite()) {
        Ok(transform)
    } else {
        Err(RegistrationError::NumericalFailure(
            "rigid fit produced a non-finite transform".to_owned(),
        ))
    }
}

fn ensure_non_collinear(points: &Array2<f64>, context: &str) -> Result<()> {
    let mut covariance = Matrix3::zeros();
    for row in 0..points.shape()[0] {
        let point = Vector3::new([
            *points
                .get([row, 0])
                .expect("invariant: three-column point array"),
            *points
                .get([row, 1])
                .expect("invariant: three-column point array"),
            *points
                .get([row, 2])
                .expect("invariant: three-column point array"),
        ]);
        covariance += Matrix3::from_rows([
            [
                point[0] * point[0],
                point[0] * point[1],
                point[0] * point[2],
            ],
            [
                point[1] * point[0],
                point[1] * point[1],
                point[1] * point[2],
            ],
            [
                point[2] * point[0],
                point[2] * point[1],
                point[2] * point[2],
            ],
        ]);
    }
    let trace = covariance[(0, 0)] + covariance[(1, 1)] + covariance[(2, 2)];
    let frobenius_squared = (0..3)
        .flat_map(|row| (0..3).map(move |column| covariance[(row, column)].powi(2)))
        .sum::<f64>();
    // For PSD covariance with eigenvalues λᵢ, this is Σᵢ<ⱼ λᵢλⱼ.
    // It is zero exactly for rank < 2; sqrt(epsilon) rejects numerically
    // unresolved second axes without assigning a dimensional scale.
    let second_elementary = ((trace * trace - frobenius_squared) * 0.5).max(0.0);
    if trace <= 0.0 || second_elementary <= RANK_TOLERANCE * trace * trace {
        return Err(RegistrationError::InvalidInput(format!(
            "{context} rigid correspondences are collinear or numerically rank deficient"
        )));
    }
    Ok(())
}

pub(super) fn squared_residual(transform: &AffineTransform, pair: &RigidCorrespondence) -> f64 {
    let matrix = transform.as_array();
    let mapped = [
        matrix[0] * pair.source_mm[0]
            + matrix[1] * pair.source_mm[1]
            + matrix[2] * pair.source_mm[2]
            + matrix[3],
        matrix[4] * pair.source_mm[0]
            + matrix[5] * pair.source_mm[1]
            + matrix[6] * pair.source_mm[2]
            + matrix[7],
        matrix[8] * pair.source_mm[0]
            + matrix[9] * pair.source_mm[1]
            + matrix[10] * pair.source_mm[2]
            + matrix[11],
    ];
    mapped
        .iter()
        .zip(pair.target_mm.iter())
        .map(|(actual, expected)| (actual - expected).powi(2))
        .sum()
}
