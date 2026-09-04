use super::super::error::{RegistrationError, Result};
use super::pose::{multiply_3x3, rigid_about_centroid};
use crate::types::AffineTransform;

/// `sqrt(f64::EPSILON)`, balancing roundoff and perturbation in 3×3 rigid checks.
const RIGID_TOLERANCE: f64 = 1.490_116_119_384_765_6e-8;

/// Full rigid transform about which residual pose parameters are searched.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RigidSearchAnchor {
    pub(super) transform: AffineTransform,
    pub(super) fixed_center_mm: [f64; 3],
    pub(super) moving_center_mm: [f64; 3],
}

impl RigidSearchAnchor {
    /// Construct the legacy identity-rotation anchor from corresponding centres.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] when either centre contains
    /// a non-finite coordinate or forming their translation overflows.
    pub fn from_centers(fixed_center_mm: [f64; 3], moving_center_mm: [f64; 3]) -> Result<Self> {
        let transform = rigid_about_centroid(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            fixed_center_mm,
            moving_center_mm,
        );
        Self::try_new(transform, fixed_center_mm)
    }

    /// Validate an externally estimated rigid search anchor.
    ///
    /// `transform` maps fixed to moving physical coordinates. Residual rotations
    /// are right-composed in the fixed frame about `fixed_center_mm`; residual
    /// translations are expressed in moving-frame millimetres. The zero residual
    /// therefore reproduces `transform` exactly.
    ///
    /// # Errors
    ///
    /// Returns [`RegistrationError::InvalidInput`] for a non-finite centre or a
    /// matrix that is not finite, homogeneous, orthonormal, and proper within a
    /// relative `sqrt(f64::EPSILON)` numerical threshold.
    pub fn try_new(transform: AffineTransform, fixed_center_mm: [f64; 3]) -> Result<Self> {
        if fixed_center_mm.iter().any(|value| !value.is_finite()) {
            return Err(RegistrationError::InvalidInput(format!(
                "rigid-search fixed center must be finite, got {fixed_center_mm:?}"
            )));
        }
        validate_rigid(transform.as_array())?;
        let matrix = transform.as_array();
        let moving_center_mm = [
            matrix[0] * fixed_center_mm[0]
                + matrix[1] * fixed_center_mm[1]
                + matrix[2] * fixed_center_mm[2]
                + matrix[3],
            matrix[4] * fixed_center_mm[0]
                + matrix[5] * fixed_center_mm[1]
                + matrix[6] * fixed_center_mm[2]
                + matrix[7],
            matrix[8] * fixed_center_mm[0]
                + matrix[9] * fixed_center_mm[1]
                + matrix[10] * fixed_center_mm[2]
                + matrix[11],
        ];
        if moving_center_mm.iter().any(|value| !value.is_finite()) {
            return Err(RegistrationError::InvalidInput(
                "rigid-search anchor maps its fixed center to a non-finite point".to_owned(),
            ));
        }
        Ok(Self {
            transform,
            fixed_center_mm,
            moving_center_mm,
        })
    }

    /// Return the fixed→moving anchor transform.
    #[must_use]
    pub const fn transform(self) -> AffineTransform {
        self.transform
    }

    /// Return the fixed-frame rotation center in millimetres.
    #[must_use]
    pub const fn fixed_center_mm(self) -> [f64; 3] {
        self.fixed_center_mm
    }

    /// Return the mapped moving-frame center in millimetres.
    #[must_use]
    pub const fn moving_center_mm(self) -> [f64; 3] {
        self.moving_center_mm
    }
}

fn validate_rigid(matrix: &[f64; 16]) -> Result<()> {
    if matrix.iter().any(|value| !value.is_finite()) {
        return Err(RegistrationError::InvalidInput(
            "rigid-search anchor matrix must be finite".to_owned(),
        ));
    }
    if matrix[12].abs() > RIGID_TOLERANCE
        || matrix[13].abs() > RIGID_TOLERANCE
        || matrix[14].abs() > RIGID_TOLERANCE
        || (matrix[15] - 1.0).abs() > RIGID_TOLERANCE
    {
        return Err(RegistrationError::InvalidInput(
            "rigid-search anchor must have homogeneous bottom row [0, 0, 0, 1]".to_owned(),
        ));
    }
    let rotation = [
        [matrix[0], matrix[1], matrix[2]],
        [matrix[4], matrix[5], matrix[6]],
        [matrix[8], matrix[9], matrix[10]],
    ];
    let gram = multiply_3x3(rotation, transpose_3x3(rotation));
    for (row, values) in gram.iter().enumerate() {
        for (column, &actual) in values.iter().enumerate() {
            let expected = if row == column { 1.0 } else { 0.0 };
            if (actual - expected).abs() > RIGID_TOLERANCE {
                return Err(RegistrationError::InvalidInput(
                    "rigid-search anchor rotation must be orthonormal".to_owned(),
                ));
            }
        }
    }
    let determinant = rotation[0][0]
        * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0]);
    if (determinant - 1.0).abs() > RIGID_TOLERANCE {
        return Err(RegistrationError::InvalidInput(
            "rigid-search anchor rotation must have determinant +1".to_owned(),
        ));
    }
    Ok(())
}

fn transpose_3x3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|row| std::array::from_fn(|column| matrix[column][row]))
}
