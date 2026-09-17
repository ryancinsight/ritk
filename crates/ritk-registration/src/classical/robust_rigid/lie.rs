use super::super::error::{RegistrationError, Result};
use super::limits::{RANK_TOLERANCE, ROTATION_LOG_BRANCH_TOLERANCE};
use crate::types::AffineTransform;

pub(super) fn invert_rigid(transform: &AffineTransform) -> Result<AffineTransform> {
    let matrix = transform.as_array();
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(RegistrationError::NumericalFailure(
            "cannot invert a non-finite rigid transform".to_owned(),
        ));
    }
    let translation = [matrix[3], matrix[7], matrix[11]];
    let inverse_translation = [
        -(matrix[0] * translation[0] + matrix[4] * translation[1] + matrix[8] * translation[2]),
        -(matrix[1] * translation[0] + matrix[5] * translation[1] + matrix[9] * translation[2]),
        -(matrix[2] * translation[0] + matrix[6] * translation[1] + matrix[10] * translation[2]),
    ];
    Ok(AffineTransform([
        matrix[0],
        matrix[4],
        matrix[8],
        inverse_translation[0],
        matrix[1],
        matrix[5],
        matrix[9],
        inverse_translation[1],
        matrix[2],
        matrix[6],
        matrix[10],
        inverse_translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]))
}

pub(super) fn log_euclidean_mean(
    forward: &AffineTransform,
    reverse_inverse: &AffineTransform,
) -> Result<AffineTransform> {
    let forward_log = rigid_logarithm(forward)?;
    let reverse_log = rigid_logarithm(reverse_inverse)?;
    let mean = std::array::from_fn(|index| (forward_log[index] + reverse_log[index]) * 0.5);
    rigid_exponential(mean)
}

/// Return the principal `se(3)` logarithm as angular and translational parts.
fn rigid_logarithm(transform: &AffineTransform) -> Result<[f64; 6]> {
    let matrix = transform.as_array();
    if !matrix.iter().all(|value| value.is_finite()) {
        return Err(RegistrationError::NumericalFailure(
            "rigid logarithm received a non-finite transform".to_owned(),
        ));
    }

    let sine_axis = [
        (matrix[9] - matrix[6]) * 0.5,
        (matrix[2] - matrix[8]) * 0.5,
        (matrix[4] - matrix[1]) * 0.5,
    ];
    let sine = vector_norm(sine_axis);
    let cosine = ((matrix[0] + matrix[5] + matrix[10] - 1.0) * 0.5).clamp(-1.0, 1.0);
    if cosine < 0.0 && sine <= ROTATION_LOG_BRANCH_TOLERANCE {
        return Err(RegistrationError::NumericalFailure(
            "principal rigid logarithm is unresolved at the 180-degree rotation branch".to_owned(),
        ));
    }
    let angle = sine.atan2(cosine);
    let angular = if sine == 0.0 {
        [0.0; 3]
    } else {
        let scale = angle / sine;
        sine_axis.map(|value| value * scale)
    };
    let translation = [matrix[3], matrix[7], matrix[11]];
    let angular_squared = dot(angular, angular);
    let jacobian_inverse_coefficient = if angular_squared <= RANK_TOLERANCE {
        // Taylor series for
        // `(1 - theta/2 * cot(theta/2)) / theta^2` avoids cancellation.
        1.0 / 12.0 + angular_squared / 720.0 + angular_squared.powi(2) / 30_240.0
    } else {
        let half_angle = angle * 0.5;
        (1.0 - half_angle * (half_angle.cos() / half_angle.sin())) / angular_squared
    };
    let cross_once = cross(angular, translation);
    let cross_twice = cross(angular, cross_once);
    let tangent_translation: [f64; 3] = std::array::from_fn(|axis| {
        translation[axis] - 0.5 * cross_once[axis]
            + jacobian_inverse_coefficient * cross_twice[axis]
    });
    let logarithm = [
        angular[0],
        angular[1],
        angular[2],
        tangent_translation[0],
        tangent_translation[1],
        tangent_translation[2],
    ];
    if logarithm.iter().all(|value| value.is_finite()) {
        Ok(logarithm)
    } else {
        Err(RegistrationError::NumericalFailure(
            "rigid logarithm produced a non-finite tangent vector".to_owned(),
        ))
    }
}

/// Exponentiate an `se(3)` tangent vector with stable small-angle series.
fn rigid_exponential(tangent: [f64; 6]) -> Result<AffineTransform> {
    if !tangent.iter().all(|value| value.is_finite()) {
        return Err(RegistrationError::NumericalFailure(
            "rigid exponential received a non-finite tangent vector".to_owned(),
        ));
    }
    let angular = [tangent[0], tangent[1], tangent[2]];
    let tangent_translation = [tangent[3], tangent[4], tangent[5]];
    let angle_squared = dot(angular, angular);
    let (rotation_linear, rotation_quadratic, translation_quadratic) =
        if angle_squared <= RANK_TOLERANCE {
            let angle_fourth = angle_squared * angle_squared;
            (
                1.0 - angle_squared / 6.0 + angle_fourth / 120.0,
                0.5 - angle_squared / 24.0 + angle_fourth / 720.0,
                1.0 / 6.0 - angle_squared / 120.0 + angle_fourth / 5_040.0,
            )
        } else {
            let angle = angle_squared.sqrt();
            (
                angle.sin() / angle,
                (1.0 - angle.cos()) / angle_squared,
                (angle - angle.sin()) / (angle_squared * angle),
            )
        };
    let [x, y, z] = angular;
    let angular_square = [
        [-(y * y + z * z), x * y, x * z],
        [x * y, -(x * x + z * z), y * z],
        [x * z, y * z, -(x * x + y * y)],
    ];
    let angular_matrix = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]];
    let rotation: [[f64; 3]; 3] = std::array::from_fn(|row| {
        std::array::from_fn(|column| {
            (if row == column { 1.0 } else { 0.0 })
                + rotation_linear * angular_matrix[row][column]
                + rotation_quadratic * angular_square[row][column]
        })
    });
    let cross_once = cross(angular, tangent_translation);
    let cross_twice = cross(angular, cross_once);
    let translation: [f64; 3] = std::array::from_fn(|axis| {
        tangent_translation[axis]
            + rotation_quadratic * cross_once[axis]
            + translation_quadratic * cross_twice[axis]
    });
    let transform = AffineTransform([
        rotation[0][0],
        rotation[0][1],
        rotation[0][2],
        translation[0],
        rotation[1][0],
        rotation[1][1],
        rotation[1][2],
        translation[1],
        rotation[2][0],
        rotation[2][1],
        rotation[2][2],
        translation[2],
        0.0,
        0.0,
        0.0,
        1.0,
    ]);
    if transform.as_array().iter().all(|value| value.is_finite()) {
        Ok(transform)
    } else {
        Err(RegistrationError::NumericalFailure(
            "rigid exponential produced a non-finite transform".to_owned(),
        ))
    }
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left.into_iter().zip(right).map(|(a, b)| a * b).sum()
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    dot(vector, vector).sqrt()
}
