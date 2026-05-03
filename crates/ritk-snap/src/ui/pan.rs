//! Pan viewport drag interaction as a testable SSOT.
//!
//! # Mathematical Specification
//!
//! Given:
//! - `viewport_origin: Pos2` — the viewport pan offset at the time the drag started (stored as screen pixels)
//! - `start: Pos2` — the pointer position (screen pixels) where the drag started
//! - `current: Pos2` — the current pointer position (screen pixels)
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

use egui::{Pos2, Vec2};

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
/// The return value is the viewport's new pan offset as a `Vec2`.
///
/// # Arguments
///
/// - `viewport_origin`: The pan offset at the time the drag started, represented as a `Pos2`.
/// - `start`: The pointer position (screen pixels) where the drag started.
/// - `current`: The current pointer position (screen pixels).
///
/// # Returns
///
/// The new viewport pan offset as a `Vec2`.
///
/// # Examples
///
/// ```ignore
/// let origin = Pos2::new(100.0, 50.0);
/// let start = Pos2::new(200.0, 150.0);
/// let current = Pos2::new(210.0, 145.0);
/// let new_pan = pan_from_drag_delta(origin, start, current);
/// // delta = (10.0, -5.0)
/// // new_pan = (110.0, 45.0)
/// assert_eq!(new_pan, Vec2::new(110.0, 45.0));
/// ```
///
/// # Property: Identity
///
/// When `current == start`, the result is `viewport_origin` converted to `Vec2` (zero delta).
pub fn pan_from_drag_delta(viewport_origin: Pos2, start: Pos2, current: Pos2) -> Vec2 {
    let delta = current - start;
    Vec2::new(
        viewport_origin.x + delta.x,
        viewport_origin.y + delta.y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_zero_delta_preserves_viewport_origin() {
        let origin = Pos2::new(100.0, 50.0);
        let start = Pos2::new(200.0, 150.0);
        let result = pan_from_drag_delta(origin, start, start);
        assert_eq!(result.x, 100.0);
        assert_eq!(result.y, 50.0);
    }

    #[test]
    fn rightward_drag_increases_x_offset() {
        let origin = Pos2::new(0.0, 0.0);
        let start = Pos2::new(100.0, 100.0);
        let current = Pos2::new(130.0, 100.0);
        let result = pan_from_drag_delta(origin, start, current);
        assert_eq!(result.x, 30.0);
        assert_eq!(result.y, 0.0);
    }

    #[test]
    fn leftward_drag_decreases_x_offset() {
        let origin = Pos2::new(100.0, 100.0);
        let start = Pos2::new(100.0, 100.0);
        let current = Pos2::new(60.0, 100.0);
        let result = pan_from_drag_delta(origin, start, current);
        assert_eq!(result.x, 60.0);
        assert_eq!(result.y, 100.0);
    }

    #[test]
    fn downward_drag_increases_y_offset() {
        let origin = Pos2::new(0.0, 0.0);
        let start = Pos2::new(100.0, 100.0);
        let current = Pos2::new(100.0, 130.0);
        let result = pan_from_drag_delta(origin, start, current);
        assert_eq!(result.x, 0.0);
        assert_eq!(result.y, 30.0);
    }

    #[test]
    fn upward_drag_decreases_y_offset() {
        let origin = Pos2::new(100.0, 100.0);
        let start = Pos2::new(100.0, 100.0);
        let current = Pos2::new(100.0, 60.0);
        let result = pan_from_drag_delta(origin, start, current);
        assert_eq!(result.x, 100.0);
        assert_eq!(result.y, 60.0);
    }

    #[test]
    fn diagonal_drag_updates_both_components_independently() {
        let origin = Pos2::new(50.0, 75.0);
        let start = Pos2::new(200.0, 150.0);
        let current = Pos2::new(220.0, 130.0);
        let result = pan_from_drag_delta(origin, start, current);
        // delta = (20.0, -20.0)
        // new_pan = (70.0, 55.0)
        assert_eq!(result.x, 70.0);
        assert_eq!(result.y, 55.0);
    }

    #[test]
    fn large_positive_drag_produces_proportional_offset_increase() {
        let origin = Pos2::new(0.0, 0.0);
        let start = Pos2::new(0.0, 0.0);
        let current = Pos2::new(100.0, 200.0);
        let result = pan_from_drag_delta(origin, start, current);
        assert_eq!(result.x, 100.0);
        assert_eq!(result.y, 200.0);
    }

    #[test]
    fn large_negative_drag_produces_proportional_offset_decrease() {
        let origin = Pos2::new(500.0, 400.0);
        let start = Pos2::new(100.0, 100.0);
        let current = Pos2::new(0.0, 0.0);
        let result = pan_from_drag_delta(origin, start, current);
        assert_eq!(result.x, 400.0);
        assert_eq!(result.y, 300.0);
    }

    #[test]
    fn fractional_delta_is_preserved() {
        let origin = Pos2::new(10.0, 20.0);
        let start = Pos2::new(50.0, 50.0);
        let current = Pos2::new(52.5, 48.3);
        let result = pan_from_drag_delta(origin, start, current);
        // delta = (2.5, -1.7)
        // new_pan = (12.5, 18.3)
        assert_eq!(result.x, 12.5);
        assert!((result.y - 18.3).abs() < 1e-6);
    }
}
