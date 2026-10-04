//! Explicit transformations from stored samples to calibrated intensity.

use thiserror::Error;

/// The precision of values emitted by a modality lookup table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LutOutputBits {
    /// Eight-bit unsigned table entries.
    Eight,
    /// Sixteen-bit unsigned table entries.
    Sixteen,
}

/// A validated linear intensity transform.
///
/// The calibrated value is stored value times slope plus intercept.
/// Coefficients must be finite. Zero slope remains valid because it is a
/// defined constant transform.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCalibration {
    slope: f64,
    intercept: f64,
}

impl LinearCalibration {
    /// Creates a linear calibration with finite coefficients.
    ///
    /// # Errors
    ///
    /// Returns [`CalibrationError::NonFiniteSlope`] or
    /// [`CalibrationError::NonFiniteIntercept`] for a non-finite coefficient.
    pub fn new(slope: f64, intercept: f64) -> Result<Self, CalibrationError> {
        if !slope.is_finite() {
            return Err(CalibrationError::NonFiniteSlope);
        }
        if !intercept.is_finite() {
            return Err(CalibrationError::NonFiniteIntercept);
        }
        Ok(Self { slope, intercept })
    }

    /// Returns the multiplicative coefficient.
    #[must_use]
    pub const fn slope(self) -> f64 {
        self.slope
    }

    /// Returns the additive coefficient.
    #[must_use]
    pub const fn intercept(self) -> f64 {
        self.intercept
    }

    /// Reports whether this transform leaves every finite value unchanged.
    #[must_use]
    pub const fn is_identity(self) -> bool {
        self.slope == 1.0 && self.intercept == 0.0
    }
}

/// A validated modality lookup table over signed stored-value coordinates.
///
/// Values below the first mapped input use the first entry. Values above the
/// table's mapped range use its last entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModalityLookupTable {
    first_mapped_value: i64,
    last_mapped_value: i64,
    entries: Box<[u16]>,
    output_bits: LutOutputBits,
}

impl ModalityLookupTable {
    /// Creates a table with one through 65,536 entries.
    ///
    /// # Errors
    ///
    /// Returns an error for an empty or oversized table, values wider than an
    /// eight-bit output, or an input range that overflows i64.
    pub fn new(
        first_mapped_value: i64,
        entries: Box<[u16]>,
        output_bits: LutOutputBits,
    ) -> Result<Self, CalibrationError> {
        if entries.is_empty() {
            return Err(CalibrationError::EmptyLookupTable);
        }
        if entries.len() > 65_536 {
            return Err(CalibrationError::LookupTableTooLarge {
                entries: entries.len(),
            });
        }
        if matches!(output_bits, LutOutputBits::Eight)
            && let Some(value) = entries
                .iter()
                .copied()
                .find(|value| *value > u16::from(u8::MAX))
        {
            return Err(CalibrationError::LookupValueExceedsEightBits { value });
        }
        let last_offset = i64::try_from(entries.len() - 1)
            .map_err(|_| CalibrationError::LookupInputRangeOverflow)?;
        let last_mapped_value = first_mapped_value
            .checked_add(last_offset)
            .ok_or(CalibrationError::LookupInputRangeOverflow)?;
        Ok(Self {
            first_mapped_value,
            last_mapped_value,
            entries,
            output_bits,
        })
    }

    /// Returns the first stored input represented by the table.
    #[must_use]
    pub const fn first_mapped_value(&self) -> i64 {
        self.first_mapped_value
    }

    /// Returns the final stored input represented by the table.
    #[must_use]
    pub const fn last_mapped_value(&self) -> i64 {
        self.last_mapped_value
    }

    /// Returns the table entries as unsigned values.
    #[must_use]
    pub fn entries(&self) -> &[u16] {
        &self.entries
    }

    /// Returns the output precision declared by the table.
    #[must_use]
    pub const fn output_bits(&self) -> LutOutputBits {
        self.output_bits
    }

    /// Maps a stored signed value, clamping outside the table's input range.
    #[must_use]
    pub fn map(&self, stored_value: i64) -> u16 {
        let index = if stored_value <= self.first_mapped_value {
            0
        } else if stored_value >= self.last_mapped_value {
            self.entries.len() - 1
        } else {
            usize::try_from(stored_value - self.first_mapped_value)
                .expect("invariant: mapped LUT input difference is non-negative")
        };
        self.entries
            .get(index)
            .copied()
            .expect("invariant: validated lookup range indexes an entry")
    }
}

/// An explicit calibration carried with stored image samples.
#[derive(Debug, Clone, PartialEq)]
pub enum IntensityCalibration {
    /// Stored samples already represent the values consumed by the caller.
    Identity,
    /// One linear transform applies to every frame.
    Linear(LinearCalibration),
    /// A separate linear transform applies to each depth-axis frame.
    PerFrameLinear(Box<[LinearCalibration]>),
    /// A nonlinear unsigned lookup table maps stored pixel values.
    ModalityLookup(ModalityLookupTable),
}

impl IntensityCalibration {
    /// Checks calibration dimensions against a depth, row, column shape.
    ///
    /// # Errors
    ///
    /// Returns CalibrationShapeError::FrameCountMismatch when per-frame
    /// calibration does not contain one entry per depth frame.
    pub fn validate_for_shape(&self, shape: [usize; 3]) -> Result<(), CalibrationShapeError> {
        if let Self::PerFrameLinear(frames) = self
            && frames.len() != shape[0]
        {
            return Err(CalibrationShapeError::FrameCountMismatch {
                expected: shape[0],
                actual: frames.len(),
            });
        }
        Ok(())
    }

    /// Reports whether a target that stores only samples can represent this
    /// calibration without changing the image.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        match self {
            Self::Identity => true,
            Self::Linear(calibration) => calibration.is_identity(),
            Self::PerFrameLinear(calibrations) => calibrations
                .iter()
                .all(|calibration| calibration.is_identity()),
            Self::ModalityLookup(_) => false,
        }
    }
}

/// A calibration coefficient or table violates its declared representation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CalibrationError {
    /// The linear slope is not finite.
    #[error("calibration slope must be finite")]
    NonFiniteSlope,
    /// The linear intercept is not finite.
    #[error("calibration intercept must be finite")]
    NonFiniteIntercept,
    /// A modality lookup table has no entries.
    #[error("modality lookup table must contain at least one entry")]
    EmptyLookupTable,
    /// A modality lookup table exceeds the DICOM 16-bit descriptor limit.
    #[error("modality lookup table has {entries} entries; maximum is 65,536")]
    LookupTableTooLarge {
        /// The attempted table length.
        entries: usize,
    },
    /// An eight-bit table contains a larger output value.
    #[error("eight-bit modality lookup table contains value {value}")]
    LookupValueExceedsEightBits {
        /// The invalid table value.
        value: u16,
    },
    /// The mapped input range cannot be represented by i64.
    #[error("modality lookup table input range overflows i64")]
    LookupInputRangeOverflow,
}

/// Per-frame calibration does not match the volume's depth axis.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CalibrationShapeError {
    /// There is not one calibration per stored frame.
    #[error("per-frame calibration count is {actual}; volume depth is {expected}")]
    FrameCountMismatch {
        /// Required frame count.
        expected: usize,
        /// Supplied calibration count.
        actual: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::{
        CalibrationError, CalibrationShapeError, IntensityCalibration, LinearCalibration,
        LutOutputBits, ModalityLookupTable,
    };

    #[test]
    fn linear_coefficients_reject_nonfinite_values() {
        assert!(matches!(
            LinearCalibration::new(f64::INFINITY, 0.0),
            Err(CalibrationError::NonFiniteSlope)
        ));
        assert!(matches!(
            LinearCalibration::new(1.0, f64::NAN),
            Err(CalibrationError::NonFiniteIntercept)
        ));
        assert_eq!(
            LinearCalibration::new(0.0, -1024.0)
                .expect("finite coefficients")
                .intercept(),
            -1024.0
        );
    }

    #[test]
    fn lookup_clamps_and_preserves_declared_precision() {
        let table =
            ModalityLookupTable::new(-1, Box::from([0_u16, 255, 65_535]), LutOutputBits::Sixteen)
                .expect("valid lookup table");
        assert_eq!(table.map(i64::MIN), 0);
        assert_eq!(table.map(-1), 0);
        assert_eq!(table.map(0), 255);
        assert_eq!(table.map(1), 65_535);
        assert_eq!(table.map(i64::MAX), 65_535);
    }

    #[test]
    fn lookup_rejects_invalid_descriptor_ranges_and_values() {
        assert!(matches!(
            ModalityLookupTable::new(0, Box::new([]), LutOutputBits::Eight),
            Err(CalibrationError::EmptyLookupTable)
        ));
        assert!(matches!(
            ModalityLookupTable::new(i64::MAX, Box::from([0_u16, 1]), LutOutputBits::Sixteen),
            Err(CalibrationError::LookupInputRangeOverflow)
        ));
        assert!(matches!(
            ModalityLookupTable::new(0, Box::from([256_u16]), LutOutputBits::Eight),
            Err(CalibrationError::LookupValueExceedsEightBits { value: 256 })
        ));
    }

    #[test]
    fn per_frame_calibration_matches_depth_axis() {
        let one_frame = LinearCalibration::new(1.0, 0.0).expect("identity linear transform");
        let calibration = IntensityCalibration::PerFrameLinear(Box::from([one_frame, one_frame]));
        assert_eq!(calibration.validate_for_shape([2, 3, 4]), Ok(()));
        assert_eq!(
            calibration.validate_for_shape([1, 3, 4]),
            Err(CalibrationShapeError::FrameCountMismatch {
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn identity_linear_calibrations_have_identity_value_semantics() {
        let identity = LinearCalibration::new(1.0, 0.0).expect("finite identity");
        let scaled = LinearCalibration::new(2.0, 0.0).expect("finite scaling");
        assert!(IntensityCalibration::Linear(identity).is_identity());
        assert!(
            IntensityCalibration::PerFrameLinear(Box::from([identity, identity])).is_identity()
        );
        assert!(!IntensityCalibration::PerFrameLinear(Box::from([identity, scaled])).is_identity());
    }
}
