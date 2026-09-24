//! Validated coordinates in DICOM patient millimetres.

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// A finite point in patient space, measured in millimetres.
///
/// Coordinate order follows DICOM patient axes `[x, y, z]`. The validated
/// representation is used for persisted measurements that remain meaningful
/// independently of a displayed slice or reslice plane.
///
/// # Examples
///
/// ```
/// use ritk_snap::geometry::{PatientPointError, PatientPointMm};
///
/// let point = PatientPointMm::try_new([1.0, 2.0, 3.0])?;
/// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
/// # Ok::<(), PatientPointError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "[f64; 3]")]
pub struct PatientPointMm([f64; 3]);

impl PatientPointMm {
    /// Construct a patient-space point after validating all coordinates.
    ///
    /// # Errors
    ///
    /// Returns [`PatientPointError::NonFiniteCoordinate`] when any component
    /// is NaN or infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::geometry::{PatientPointError, PatientPointMm};
    ///
    /// let point = PatientPointMm::try_new([1.0, 2.0, 3.0])?;
    /// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
    /// # Ok::<(), PatientPointError>(())
    /// ```
    pub fn try_new(coordinates: [f64; 3]) -> Result<Self, PatientPointError> {
        for (axis, value) in coordinates.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(PatientPointError::NonFiniteCoordinate { axis, value });
            }
        }
        Ok(Self(coordinates))
    }

    /// Return patient coordinates in millimetres as `[x, y, z]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::geometry::{PatientPointError, PatientPointMm};
    ///
    /// let point = PatientPointMm::try_new([1.0, 2.0, 3.0])?;
    /// assert_eq!(point.coordinates(), [1.0, 2.0, 3.0]);
    /// # Ok::<(), PatientPointError>(())
    /// ```
    #[must_use]
    pub const fn coordinates(self) -> [f64; 3] {
        self.0
    }
}

impl TryFrom<[f64; 3]> for PatientPointMm {
    type Error = PatientPointError;

    fn try_from(coordinates: [f64; 3]) -> Result<Self, Self::Error> {
        Self::try_new(coordinates)
    }
}

/// Invalid construction or deserialization of a patient-space point.
///
/// # Examples
///
/// ```
/// use ritk_snap::geometry::{PatientPointError, PatientPointMm};
///
/// assert!(matches!(
///     PatientPointMm::try_new([0.0, f64::NAN, 1.0]),
///     Err(PatientPointError::NonFiniteCoordinate { axis: 1, .. })
/// ));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum PatientPointError {
    /// A patient coordinate is NaN or infinite.
    #[error("patient coordinate {axis} is not finite: {value}")]
    NonFiniteCoordinate {
        /// Coordinate index in `[x, y, z]` order.
        axis: usize,
        /// Rejected coordinate value.
        value: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::{PatientPointError, PatientPointMm};

    #[test]
    fn point_round_trips_through_array_json() {
        let point = PatientPointMm::try_new([1.25, -2.5, 3.75]).expect("finite point");
        let json = serde_json::to_string(&point).expect("serialize point");
        assert_eq!(json, "[1.25,-2.5,3.75]");
        let recovered: PatientPointMm = serde_json::from_str(&json).expect("deserialize point");
        assert_eq!(recovered, point);
    }

    #[test]
    fn point_constructor_rejects_non_finite_values_on_every_axis() {
        for axis in 0..3 {
            for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let coordinates: [f64; 3] =
                    std::array::from_fn(|component| if component == axis { invalid } else { 0.0 });
                let Err(PatientPointError::NonFiniteCoordinate {
                    axis: rejected_axis,
                    value,
                }) = PatientPointMm::try_new(coordinates)
                else {
                    panic!("non-finite coordinate must be rejected");
                };
                assert_eq!(rejected_axis, axis);
                assert_eq!(value.to_bits(), invalid.to_bits());
            }
        }
    }

    #[test]
    fn point_deserialization_rejects_invalid_array_shape() {
        let error = serde_json::from_str::<PatientPointMm>("[0.0,1.0]")
            .expect_err("patient point requires exactly three coordinates");
        assert!(error.to_string().contains("length 2"));
    }
}
