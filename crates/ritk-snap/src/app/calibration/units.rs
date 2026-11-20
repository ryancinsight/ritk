//! Physical units accepted for a manually measured reference.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CalibrationUnit {
    Millimetres,
    Inches,
    French,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CalibrationInputError {
    InvalidDistance,
    OutsideRange,
}

const MILLIMETRES_PER_INCH: f32 = 25.4;
const FRENCH_PER_MILLIMETRE: f32 = 3.0;

impl CalibrationUnit {
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Millimetres => "mm",
            Self::Inches => "inches",
            Self::French => "French",
        }
    }

    pub(crate) const fn next(self) -> Self {
        match self {
            Self::Millimetres => Self::Inches,
            Self::Inches => Self::French,
            Self::French => Self::Millimetres,
        }
    }

    pub(crate) const fn previous(self) -> Self {
        self.next().next()
    }

    /// Converts the entered reference to the annotation distance precision.
    ///
    /// One inch is exactly 25.4 mm (NIST, SI Units – Length, FAQ):
    /// <https://www.nist.gov/pml/owm/si-units-length>.
    /// French size is three times the outer diameter in mm (EAUN 2024,
    /// section 5.2.2):
    /// <https://nurses.uroweb.org/guidelines/indwelling-catheterisation-in-adults-urethral-and-suprapubic/chapter/equipment-and-products>.
    pub(crate) fn millimetres(self, distance: f32) -> Result<f32, CalibrationInputError> {
        if !distance.is_finite() || distance <= 0.0 {
            return Err(CalibrationInputError::InvalidDistance);
        }
        let millimetres = match self {
            Self::Millimetres => distance,
            Self::Inches => distance * MILLIMETRES_PER_INCH,
            Self::French => distance / FRENCH_PER_MILLIMETRE,
        };
        if !millimetres.is_finite() || millimetres <= 0.0 {
            return Err(CalibrationInputError::OutsideRange);
        }
        Ok(millimetres)
    }
}

#[cfg(test)]
mod tests {
    use super::CalibrationInputError::{InvalidDistance, OutsideRange};
    use super::CalibrationUnit::{French, Inches, Millimetres};

    #[test]
    fn reference_units_follow_published_dimensions() {
        for (unit, entered, expected) in [
            (Millimetres, 12.7, 12.7),
            (Inches, 0.5, 12.7),
            (French, 39.0, 13.0),
        ] {
            assert_eq!(unit.millimetres(entered), Ok(expected));
        }
    }

    #[test]
    fn entered_and_converted_limits_are_distinct_errors() {
        for unit in [Millimetres, Inches, French] {
            for invalid in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert_eq!(unit.millimetres(invalid), Err(InvalidDistance));
            }
        }
        for (unit, entered, expected) in [
            (Inches, f32::MAX, Err(OutsideRange)),
            (French, f32::from_bits(1), Err(OutsideRange)),
            (French, f32::from_bits(3), Ok(f32::from_bits(1))),
            (Millimetres, f32::MAX, Ok(f32::MAX)),
            (Millimetres, f32::from_bits(1), Ok(f32::from_bits(1))),
        ] {
            assert_eq!(unit.millimetres(entered), expected);
        }
    }
}
