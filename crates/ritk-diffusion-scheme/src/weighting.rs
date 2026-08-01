//! Physically typed diffusion weighting.

use aequitas::Quantity;
use aequitas::dimension::DivideDimension;
use aequitas::systems::si::dimensions::{Area, Time};

use crate::GradientSchemeError;

type TimePerArea = <Time as DivideDimension<Area>>::Output;

const SQUARE_METERS_PER_SQUARE_MILLIMETER: f64 = 1.0e-6;
const SQUARE_MILLIMETERS_PER_SQUARE_METER: f64 = 1.0 / SQUARE_METERS_PER_SQUARE_MILLIMETER;

/// Diffusion sensitization factor with dimension time per area.
///
/// Storage uses canonical SI seconds per square meter. Use
/// [`from_seconds_per_square_millimeter`](Self::from_seconds_per_square_millimeter)
/// and [`seconds_per_square_millimeter`](Self::seconds_per_square_millimeter)
/// at MRI format and user-interface boundaries.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct DiffusionWeighting(Quantity<f64, TimePerArea>);

impl DiffusionWeighting {
    /// Construct from the MRI convention seconds per square millimeter.
    ///
    /// # Errors
    ///
    /// Returns [`GradientSchemeError::InvalidWeighting`] for a negative,
    /// NaN, or infinite value.
    pub fn from_seconds_per_square_millimeter(value: f64) -> Result<Self, GradientSchemeError> {
        Self::at_index(value, 0)
    }

    pub(crate) fn at_index(value: f64, index: usize) -> Result<Self, GradientSchemeError> {
        if !value.is_finite() || value < 0.0 {
            return Err(GradientSchemeError::InvalidWeighting { index, value });
        }
        Ok(Self(Quantity::from_base(
            value * SQUARE_MILLIMETERS_PER_SQUARE_METER,
        )))
    }

    /// Return the value in seconds per square millimeter.
    #[must_use]
    pub fn seconds_per_square_millimeter(self) -> f64 {
        *self.0.as_base() * SQUARE_METERS_PER_SQUARE_MILLIMETER
    }

    /// Return the canonical SI value in seconds per square meter.
    #[must_use]
    pub fn seconds_per_square_meter(self) -> f64 {
        *self.0.as_base()
    }

    /// Return whether this is an exact unweighted acquisition.
    #[must_use]
    pub fn is_unweighted(self) -> bool {
        self.seconds_per_square_millimeter() == 0.0
    }
}
