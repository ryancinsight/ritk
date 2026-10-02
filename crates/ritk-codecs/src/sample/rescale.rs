//! Linear maps from stored sample values to physical values.

use coeus_core::Scalar;

use super::{Sample, SampleError};

/// The linear map `y = slope · x + intercept` from a stored sample `x` to the
/// physical value `y` it encodes.
///
/// Formats store it beside integer samples (NIfTI `scl_slope`/`scl_inter`,
/// DICOM Rescale Slope/Intercept, Analyze `funused1`); each format decides
/// how its header fields map onto a `Rescale`, including which field values
/// mean "no rescale". The coefficients are `f64`, wide enough for every
/// header's field; [`apply`](Self::apply) evaluates the map in the sample
/// type's own arithmetic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rescale {
    slope: f64,
    intercept: f64,
}

impl Rescale {
    /// The map that leaves every sample unchanged.
    pub const IDENTITY: Self = Self {
        slope: 1.0,
        intercept: 0.0,
    };

    /// The map `y = slope · x + intercept`.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::NonFiniteRescale`] when either coefficient is
    /// NaN or infinite.
    pub fn new(slope: f64, intercept: f64) -> Result<Self, SampleError> {
        if slope.is_finite() && intercept.is_finite() {
            Ok(Self { slope, intercept })
        } else {
            Err(SampleError::NonFiniteRescale { slope, intercept })
        }
    }

    /// The multiplier applied to each stored sample.
    #[must_use]
    pub fn slope(self) -> f64 {
        self.slope
    }

    /// The offset added after the multiplier.
    #[must_use]
    pub fn intercept(self) -> f64 {
        self.intercept
    }

    /// Whether the map leaves every sample unchanged.
    #[must_use]
    pub fn is_identity(self) -> bool {
        self == Self::IDENTITY
    }

    /// Map each sample to its physical value in place.
    ///
    /// The identity map changes nothing for any `T`. Any other map is
    /// evaluated as `x · slope + intercept` in `T`, with both coefficients
    /// converted to `T` once.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::IntegerRescale`], leaving `values` unchanged,
    /// when the map is not the identity and `T` is an integer type: a
    /// fractional or offset physical value has no faithful integer form, so
    /// such a caller reads the stored samples and the map separately.
    /// Returns [`SampleError::RescaleOutOfRange`], leaving `values`
    /// unchanged, when a coefficient overflows `T` or a nonzero slope
    /// underflows to zero in `T`.
    pub fn apply<T: Sample + Scalar>(self, values: &mut [T]) -> Result<(), SampleError> {
        if self.is_identity() {
            return Ok(());
        }
        if !T::TYPE.is_float() {
            return Err(SampleError::IntegerRescale {
                sample_type: T::TYPE,
                slope: self.slope,
                intercept: self.intercept,
            });
        }
        let slope = T::from_real_sample(self.slope);
        let intercept = T::from_real_sample(self.intercept);
        let slope_is_finite = slope.is_finite();
        let intercept_is_finite = intercept.is_finite();
        let slope_is_zero = slope == T::zero();
        // A nonzero slope that rounds to zero in `T` would map every sample
        // to the intercept: an underflow as unfaithful as an overflow.
        if (slope_is_zero && self.slope != 0.0) || !(slope_is_finite && intercept_is_finite) {
            return Err(SampleError::RescaleOutOfRange {
                sample_type: T::TYPE,
                slope: self.slope,
                intercept: self.intercept,
            });
        }
        for value in values {
            *value = *value * slope + intercept;
        }
        Ok(())
    }
}
