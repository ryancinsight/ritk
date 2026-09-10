//! Pan viewport drag interaction as a testable SSOT.
//!
//! # Mathematical Specification
//!
//! Given:
//! - `viewport_origin: ViewportOffset` — the viewport pan offset at the time the drag started
//! - `start: ImagePoint` — the pointer position (screen pixels) where the drag started
//! - `current: ImagePoint` — the current pointer position (screen pixels)
//!
//! The new pan offset is:
//!
//! ```text
//! delta = current − start
//! new_offset = viewport_origin + delta
//! ```
//!
//! Pan offset is additive: each pixel of pointer motion in a direction translates
//! the viewport by one pixel in that same direction (no sensitivity scaling).
//!
//! # Property-Based Invariants
//!
//! **Identity**: When `current == start`, then `pan_from_drag_delta(...)` returns
//! `viewport_origin` unchanged (zero delta produces zero change).
//!
//! **Additive commutativity**: If two disjoint drags produce deltas δ₁ and δ₂,
//! their combined effect is the same as applying δ₁ + δ₂ in a single drag.
//!
//! **Directional independence**: Horizontal and vertical components are computed
//! independently; movement in the x direction does not affect the y component
//! of the result and vice versa.

use crate::tools::interaction::{ImagePoint, ViewportOffset};

/// Calculate the new pan offset given drag start, current pointer position, and
/// the original viewport origin at drag start.
///
/// # Mathematical Contract
///
/// This function computes:
/// ```text
/// delta = current − start
/// new_pan = viewport_origin + delta
/// ```
///
/// The return value is the viewport's new [`ViewportOffset`].
///
/// # Arguments
///
/// - `viewport_origin`: The pan offset at the time the drag started.
/// - `start`: The pointer position (screen pixels) where the drag started.
/// - `current`: The current pointer position (screen pixels).
///
/// # Returns
///
/// The new viewport pan offset.
///
/// # Examples
///
/// ```
/// # use ritk_snap::tools::interaction::{ImagePoint, ViewportOffset};
/// # use ritk_snap::ui::pan::pan_from_drag_delta;
/// let origin = ViewportOffset::new(100.0, 50.0);
/// let start = ImagePoint::new(200.0, 150.0);
/// let current = ImagePoint::new(210.0, 145.0);
/// let new_pan = pan_from_drag_delta(origin, start, current);
/// // delta = (10.0, -5.0)
/// // new_pan = (110.0, 45.0)
/// assert_eq!(new_pan, ViewportOffset::new(110.0, 45.0));
/// ```
///
/// # Property: Identity
///
/// When `current == start`, the result is `viewport_origin` (zero delta).
pub fn pan_from_drag_delta(
    viewport_origin: ViewportOffset,
    start: ImagePoint,
    current: ImagePoint,
) -> ViewportOffset {
    let delta_x = current.x() - start.x();
    let delta_y = current.y() - start.y();
    ViewportOffset::new(viewport_origin.x() + delta_x, viewport_origin.y() + delta_y)
}

#[cfg(test)]
#[path = "tests_pan.rs"]
mod tests;
