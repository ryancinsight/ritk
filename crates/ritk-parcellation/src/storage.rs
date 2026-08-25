//! Reading labels out of volumes that store them as floating-point values.
//!
//! Label volumes are integers, but every general image format and every image
//! pipeline in RITK carries `f32` samples, so a label arrives as a float on the
//! way in from NIfTI, from a Python array, or from a warped atlas. Recovering
//! the integer is a decision with a wrong answer — truncation turns a value a
//! hair under its label into the label below — so it lives here once rather
//! than at each boundary that faces the problem.

use crate::BACKGROUND;

/// The label a stored floating-point sample represents.
///
/// Rounds to the nearest integer, so a value displaced by interpolation or by
/// the format's own precision recovers the label it was written as. Anything at
/// or below zero is [`BACKGROUND`]: a negative sample cannot be a label, and
/// silently wrapping it into a large positive one would invent a region.
///
/// Non-finite samples are background for the same reason — a NaN carries no
/// label, and every comparison against it is false, so it is rejected
/// explicitly rather than falling through the sign test. A sample beyond
/// `u32::MAX` saturates there; no real label volume reaches four billion
/// regions, and saturating is the defined behaviour of the cast rather than a
/// silent wrap to a small label.
///
/// # Examples
///
/// ```
/// use ritk_parcellation::storage::label_from_stored;
///
/// assert_eq!(label_from_stored(17.000_002), 17);
/// assert_eq!(label_from_stored(16.999_998), 17);
/// assert_eq!(label_from_stored(-1.0), 0);
/// assert_eq!(label_from_stored(f32::NAN), 0);
/// ```
#[must_use]
pub fn label_from_stored(value: f32) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        return BACKGROUND;
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the value is positive and a label volume holds integers"
    )]
    let label = value.round() as u32;
    label
}

#[cfg(test)]
mod tests;
