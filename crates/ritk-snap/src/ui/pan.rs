//! Checked arithmetic for viewport pan offsets.

use crate::tools::interaction::ViewportOffset;
use thiserror::Error;

/// Failure to apply a pan displacement in viewer precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum PanError {
    /// The captured viewport offset is not finite.
    #[error("the captured pan offset must be finite")]
    InvalidOrigin,
    /// The supplied displacement is not finite.
    #[error("the pan displacement must be finite")]
    InvalidDisplacement,
    /// Adding the displacement overflows the viewport range.
    #[error("the pan result is outside the viewport range")]
    UnrepresentableResult,
}

/// Add a press-relative displacement to a captured viewport offset.
///
/// The two `f32` components are returned together or not at all.
///
/// # Examples
/// ```
/// use ritk_snap::tools::interaction::ViewportOffset;
/// use ritk_snap::ui::{pan_from_drag_delta, PanError};
///
/// let moved = pan_from_drag_delta(
///     ViewportOffset::new(4.0, -2.0),
///     ViewportOffset::new(3.0, 5.0),
/// )?;
/// assert_eq!(moved, ViewportOffset::new(7.0, 3.0));
///
/// let overflow = pan_from_drag_delta(
///     ViewportOffset::new(f32::MAX, 0.0),
///     ViewportOffset::new(f32::MAX, 0.0),
/// );
/// assert_eq!(overflow, Err(PanError::UnrepresentableResult));
/// # Ok::<(), PanError>(())
/// ```
///
/// # Errors
/// Returns [`PanError`] when the origin, displacement, or result is not finite.
pub fn pan_from_drag_delta(
    viewport_origin: ViewportOffset,
    displacement: ViewportOffset,
) -> Result<ViewportOffset, PanError> {
    if !viewport_origin.x().is_finite() || !viewport_origin.y().is_finite() {
        return Err(PanError::InvalidOrigin);
    }
    if !displacement.x().is_finite() || !displacement.y().is_finite() {
        return Err(PanError::InvalidDisplacement);
    }
    let result = ViewportOffset::new(
        viewport_origin.x() + displacement.x(),
        viewport_origin.y() + displacement.y(),
    );
    if !result.x().is_finite() || !result.y().is_finite() {
        return Err(PanError::UnrepresentableResult);
    }
    Ok(result)
}

#[cfg(test)]
#[path = "tests_pan.rs"]
mod tests;
