//! Validated lengths measured in DICOM patient space.

use super::MeasurementError;
use crate::geometry::PatientPointMm;
use serde::{Deserialize, Serialize};

/// A validated straight-line distance between two points in DICOM patient space.
///
/// The endpoints use the DICOM patient coordinate order `[x, y, z]` in
/// millimetres. Construction rejects a displacement whose Euclidean norm
/// cannot be represented as a finite `f64`.
///
/// # Examples
///
/// ```
/// use ritk_snap::geometry::PatientPointMm;
/// use ritk_snap::tools::interaction::{MeasurementError, PatientLength};
///
/// let start = PatientPointMm::try_new([0.0, 0.0, 0.0]).expect("finite point");
/// let end = PatientPointMm::try_new([3.0, 4.0, 0.0]).expect("finite point");
/// let length = PatientLength::try_new(start, end)?;
/// assert_eq!(length.start_mm(), start);
/// assert_eq!(length.end_mm(), end);
/// assert_eq!(length.length_mm(), 5.0);
/// # Ok::<(), MeasurementError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(
    try_from = "PatientLengthRepresentation",
    into = "PatientLengthRepresentation"
)]
pub struct PatientLength {
    start_mm: PatientPointMm,
    end_mm: PatientPointMm,
}

impl PatientLength {
    /// Construct a patient-space length from two validated endpoints.
    ///
    /// # Errors
    ///
    /// Returns [`MeasurementError::NonFiniteResult`] when the three-dimensional
    /// Euclidean distance overflows `f64`.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::geometry::PatientPointMm;
    /// use ritk_snap::tools::interaction::{MeasurementError, PatientLength};
    ///
    /// let start = PatientPointMm::try_new([0.0, 0.0, 0.0]).expect("finite point");
    /// let end = PatientPointMm::try_new([3.0, 4.0, 0.0]).expect("finite point");
    /// let length = PatientLength::try_new(start, end)?;
    /// assert_eq!(length.length_mm(), 5.0);
    /// # Ok::<(), MeasurementError>(())
    /// ```
    pub fn try_new(
        start_mm: PatientPointMm,
        end_mm: PatientPointMm,
    ) -> Result<Self, MeasurementError> {
        let measurement = Self { start_mm, end_mm };
        measurement
            .distance()
            .is_finite()
            .then_some(measurement)
            .ok_or(MeasurementError::NonFiniteResult {
                kind: "patient length",
            })
    }

    /// Return the start point in DICOM patient millimetres.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::geometry::PatientPointMm;
    /// use ritk_snap::tools::interaction::{MeasurementError, PatientLength};
    ///
    /// let start = PatientPointMm::try_new([1.0, 2.0, 3.0]).expect("finite point");
    /// let end = PatientPointMm::try_new([4.0, 6.0, 3.0]).expect("finite point");
    /// let length = PatientLength::try_new(start, end)?;
    /// assert_eq!(length.start_mm(), start);
    /// # Ok::<(), MeasurementError>(())
    /// ```
    #[must_use]
    pub const fn start_mm(self) -> PatientPointMm {
        self.start_mm
    }

    /// Return the end point in DICOM patient millimetres.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::geometry::PatientPointMm;
    /// use ritk_snap::tools::interaction::{MeasurementError, PatientLength};
    ///
    /// let start = PatientPointMm::try_new([1.0, 2.0, 3.0]).expect("finite point");
    /// let end = PatientPointMm::try_new([4.0, 6.0, 3.0]).expect("finite point");
    /// let length = PatientLength::try_new(start, end)?;
    /// assert_eq!(length.end_mm(), end);
    /// # Ok::<(), MeasurementError>(())
    /// ```
    #[must_use]
    pub const fn end_mm(self) -> PatientPointMm {
        self.end_mm
    }

    /// Return the three-dimensional Euclidean distance in millimetres.
    ///
    /// # Examples
    ///
    /// ```
    /// use ritk_snap::geometry::PatientPointMm;
    /// use ritk_snap::tools::interaction::{MeasurementError, PatientLength};
    ///
    /// let start = PatientPointMm::try_new([0.0, 0.0, 0.0]).expect("finite point");
    /// let end = PatientPointMm::try_new([3.0, 4.0, 0.0]).expect("finite point");
    /// let length = PatientLength::try_new(start, end)?;
    /// assert_eq!(length.length_mm(), 5.0);
    /// # Ok::<(), MeasurementError>(())
    /// ```
    #[must_use]
    pub fn length_mm(self) -> f64 {
        self.distance()
    }

    fn distance(self) -> f64 {
        let start = self.start_mm.coordinates();
        let end = self.end_mm.coordinates();
        let dx = end[0] - start[0];
        let dy = end[1] - start[1];
        let dz = end[2] - start[2];
        dx.hypot(dy).hypot(dz)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct PatientLengthRepresentation {
    start_mm: PatientPointMm,
    end_mm: PatientPointMm,
}

impl TryFrom<PatientLengthRepresentation> for PatientLength {
    type Error = MeasurementError;

    fn try_from(value: PatientLengthRepresentation) -> Result<Self, Self::Error> {
        Self::try_new(value.start_mm, value.end_mm)
    }
}

impl From<PatientLength> for PatientLengthRepresentation {
    fn from(value: PatientLength) -> Self {
        Self {
            start_mm: value.start_mm,
            end_mm: value.end_mm,
        }
    }
}
