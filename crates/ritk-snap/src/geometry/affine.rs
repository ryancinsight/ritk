//! Validated Cartesian voxel and patient-coordinate transforms.

use thiserror::Error;

/// Failure while constructing a physical-space affine transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum AffineError {
    /// The origin contains a non-finite coordinate.
    #[error("physical origin contains a non-finite coordinate")]
    NonFiniteOrigin,
    /// The direction matrix contains a non-finite coordinate.
    #[error("direction matrix contains a non-finite coordinate")]
    NonFiniteDirection,
    /// At least one voxel spacing is not positive and finite.
    #[error("voxel spacing must be positive and finite")]
    InvalidSpacing,
    /// The scaled direction matrix cannot be inverted safely.
    #[error("direction and spacing form a singular affine")]
    Singular,
}

/// A validated affine map between continuous voxel and patient coordinates.
///
/// The voxel axes use the viewer's `[depth, row, column]` order. `direction`
/// is row-major and its columns are those three physical axis directions;
/// `spacing` is `[dz, dy, dx]` in millimetres. The forward map is
/// `origin + direction * diag(spacing) * voxel`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineTransform {
    origin: [f64; 3],
    voxel_to_patient: [f64; 9],
    patient_to_voxel: [f64; 9],
}

impl AffineTransform {
    /// Construct and validate an affine transform from image metadata.
    ///
    /// # Errors
    /// Returns [`AffineError`] when metadata is non-finite, has non-positive
    /// spacing, or cannot be inverted.
    pub fn from_parts(
        origin: [f64; 3],
        direction: [f64; 9],
        spacing: [f64; 3],
    ) -> Result<Self, AffineError> {
        if !origin.iter().all(|value| value.is_finite()) {
            return Err(AffineError::NonFiniteOrigin);
        }
        if !direction.iter().all(|value| value.is_finite()) {
            return Err(AffineError::NonFiniteDirection);
        }
        if !spacing
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(AffineError::InvalidSpacing);
        }

        let voxel_to_patient = [
            direction[0] * spacing[0],
            direction[1] * spacing[1],
            direction[2] * spacing[2],
            direction[3] * spacing[0],
            direction[4] * spacing[1],
            direction[5] * spacing[2],
            direction[6] * spacing[0],
            direction[7] * spacing[1],
            direction[8] * spacing[2],
        ];
        let patient_to_voxel = invert3x3(voxel_to_patient).ok_or(AffineError::Singular)?;
        Ok(Self {
            origin,
            voxel_to_patient,
            patient_to_voxel,
        })
    }

    /// Map a continuous voxel coordinate to patient coordinates.
    #[must_use]
    pub fn voxel_to_patient(&self, voxel: [f64; 3]) -> [f64; 3] {
        [
            self.origin[0]
                + self.voxel_to_patient[0] * voxel[0]
                + self.voxel_to_patient[1] * voxel[1]
                + self.voxel_to_patient[2] * voxel[2],
            self.origin[1]
                + self.voxel_to_patient[3] * voxel[0]
                + self.voxel_to_patient[4] * voxel[1]
                + self.voxel_to_patient[5] * voxel[2],
            self.origin[2]
                + self.voxel_to_patient[6] * voxel[0]
                + self.voxel_to_patient[7] * voxel[1]
                + self.voxel_to_patient[8] * voxel[2],
        ]
    }

    /// Map a patient coordinate to a continuous voxel coordinate.
    #[must_use]
    pub fn patient_to_voxel(&self, patient: [f64; 3]) -> [f64; 3] {
        let centered = [
            patient[0] - self.origin[0],
            patient[1] - self.origin[1],
            patient[2] - self.origin[2],
        ];
        [
            self.patient_to_voxel[0] * centered[0]
                + self.patient_to_voxel[1] * centered[1]
                + self.patient_to_voxel[2] * centered[2],
            self.patient_to_voxel[3] * centered[0]
                + self.patient_to_voxel[4] * centered[1]
                + self.patient_to_voxel[5] * centered[2],
            self.patient_to_voxel[6] * centered[0]
                + self.patient_to_voxel[7] * centered[1]
                + self.patient_to_voxel[8] * centered[2],
        ]
    }

    /// Return a unit vector for one voxel axis in patient space.
    #[must_use]
    pub fn axis_direction(&self, axis: usize) -> Option<[f64; 3]> {
        if axis > 2 {
            return None;
        }
        let vector = [
            self.voxel_to_patient[axis],
            self.voxel_to_patient[3 + axis],
            self.voxel_to_patient[6 + axis],
        ];
        let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
        (norm.is_finite() && norm > 0.0).then_some([
            vector[0] / norm,
            vector[1] / norm,
            vector[2] / norm,
        ])
    }
}

fn invert3x3(matrix: [f64; 9]) -> Option<[f64; 9]> {
    let determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    let scale = matrix
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    let threshold = 128.0 * f64::EPSILON * scale * scale * scale;
    if !determinant.is_finite()
        || !scale.is_finite()
        || scale == 0.0
        || determinant.abs() <= threshold
    {
        return None;
    }
    let inverse_determinant = 1.0 / determinant;
    Some([
        (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inverse_determinant,
        (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inverse_determinant,
        (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inverse_determinant,
        (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inverse_determinant,
        (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inverse_determinant,
        (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inverse_determinant,
        (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inverse_determinant,
        (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inverse_determinant,
        (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inverse_determinant,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anisotropic_rotated_round_trip_preserves_patient_point() {
        let transform = AffineTransform::from_parts(
            [10.0, 20.0, 30.0],
            [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 3.0, 0.5],
        )
        .expect("valid transform");
        let voxel = [4.0, 5.0, 6.0];
        let patient = transform.voxel_to_patient(voxel);
        let recovered = transform.patient_to_voxel(patient);
        for (actual, expected) in recovered.into_iter().zip(voxel) {
            assert!((actual - expected).abs() <= 32.0 * f64::EPSILON);
        }
    }

    #[test]
    fn invalid_spacing_and_singular_direction_are_rejected() {
        assert_eq!(
            AffineTransform::from_parts([0.0; 3], [1.0; 9], [1.0, 0.0, 1.0]),
            Err(AffineError::InvalidSpacing)
        );
        assert_eq!(
            AffineTransform::from_parts([0.0; 3], [0.0; 9], [1.0; 3]),
            Err(AffineError::Singular)
        );
    }
}
