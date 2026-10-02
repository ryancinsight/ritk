//! The MINC real-value map of one slice.

use ritk_codecs::sample::{Rescale, Sample, SampleError};

/// The MINC pixel conversion of one slice, from a stored sample to its real
/// intensity:
///
/// ```text
/// real = (stored - valid_minimum) * slope + intercept
/// ```
///
/// where `slope = (image_max - image_min) / (valid_max - valid_min)` and
/// `intercept = image_min`. This is libminc's
/// `(v - vmin) / (vmax - vmin) * (rmax - rmin) + rmin` with the quotient and
/// the real-range width folded into one multiplier.
///
/// The stored value is offset by `valid_minimum` before it is scaled, never
/// folded into the intercept: `stored - valid_minimum` is exact in floating
/// point for stored integers, whereas the folded intercept
/// `image_min - valid_minimum * slope` cancels catastrophically when
/// `valid_minimum` is far from zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RealValueMap {
    shift: Rescale,
    scale: Rescale,
}

impl RealValueMap {
    /// The map that leaves every stored value unchanged.
    pub const IDENTITY: Self = Self {
        shift: Rescale::IDENTITY,
        scale: Rescale::IDENTITY,
    };

    /// The map `real = (stored - valid_minimum) * slope + intercept`.
    ///
    /// A map with `slope == 1` and `intercept == valid_minimum` is the
    /// identity and is stored as [`IDENTITY`](Self::IDENTITY), whose
    /// [`valid_minimum`](Self::valid_minimum) is zero.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::NonFiniteRescale`] when any argument is NaN or
    /// infinite.
    pub fn new(valid_minimum: f64, slope: f64, intercept: f64) -> Result<Self, SampleError> {
        let scale = Rescale::new(slope, intercept)?;
        let shift = Rescale::new(1.0, -valid_minimum)?;
        if slope == 1.0 && intercept == valid_minimum {
            return Ok(Self::IDENTITY);
        }
        Ok(Self { shift, scale })
    }

    /// The stored value that maps to the intercept, `valid_min`.
    #[must_use]
    pub fn valid_minimum(self) -> f64 {
        -self.shift.intercept()
    }

    /// The real-range width per stored step, `(image_max - image_min) /
    /// (valid_max - valid_min)`.
    #[must_use]
    pub fn slope(self) -> f64 {
        self.scale.slope()
    }

    /// The real value of a stored `valid_min`, `image_min`.
    #[must_use]
    pub fn intercept(self) -> f64 {
        self.scale.intercept()
    }

    /// Whether the map leaves every stored value unchanged.
    #[must_use]
    pub fn is_identity(self) -> bool {
        self == Self::IDENTITY
    }

    /// Map each stored value to its real intensity in place, in `T`'s
    /// arithmetic: `(x - valid_minimum)` first, then `* slope + intercept`.
    ///
    /// The identity map changes nothing for any `T`. Rounding: the offset is
    /// exact for a stored integer `T` represents exactly and a `valid_minimum`
    /// likewise; the coefficients convert to `T` once each, and the multiply
    /// and the add round once each.
    ///
    /// # Errors
    ///
    /// Returns [`SampleError::IntegerRescale`], leaving `values` unchanged,
    /// when the map is not the identity and `T` is an integer type: a
    /// fractional or offset real value has no faithful integer form. Returns
    /// [`SampleError::RescaleOutOfRange`], leaving `values` unchanged, when a
    /// coefficient overflows `T` or a nonzero slope underflows to zero in `T`.
    pub fn apply<T: Sample>(self, values: &mut [T]) -> Result<(), SampleError> {
        if self.is_identity() {
            return Ok(());
        }
        if !T::TYPE.is_float() {
            return Err(SampleError::IntegerRescale {
                sample_type: T::TYPE,
                slope: self.slope(),
                intercept: self.intercept(),
            });
        }
        // Applying to no values checks the scale's coefficients against `T`
        // first, so the offset below never changes `values` and then leaves the
        // scale refused.
        self.scale.apply::<T>(&mut [])?;
        self.shift.apply(values)?;
        self.scale.apply(values)
    }
}

#[cfg(test)]
#[path = "tests_real_map.rs"]
mod tests;
