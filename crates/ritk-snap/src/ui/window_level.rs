//! Window/Level (W/L) drag interaction SSOT.
//!
//! # Mathematical specification
//!
//! ITK-SNAP convention: horizontal drag adjusts window **width**; vertical
//! drag adjusts window **center**.  Both axes are proportional with a shared
//! sensitivity constant `s` (HU per screen pixel):
//!
//! ```text
//! new_width  = max(1.0, original_width  + dx · s)
//! new_center = original_center − dy · s    (y-axis inverted: drag up → higher center)
//! ```
//!
//! Width is clamped to `[1.0, ∞)` so the sigmoid-like windowing transform
//! `output = (value − center + width/2) / width` never produces a zero
//! denominator in the renderer.  Center is unbounded.
//!
//! # Proof of monotonicity (width)
//!
//! For fixed `dx > 0`, `new_width = original_width + dx · s ≥ original_width`
//! (monotone non-decreasing in `dx`).  The `max(1.0, ...)` clamp is the
//! identity for all physically meaningful starting widths (`width ≥ 1.0`), so
//! it does not break monotonicity.
//!
//! # Proof of monotonicity (center)
//!
//! For fixed `dy > 0` (downward screen drag), `new_center = original_center −
//! dy · s < original_center` (monotone decreasing in `dy`), matching the
//! screen convention where dragging upward raises center.

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default drag sensitivity: HU per screen pixel.
///
/// Chosen to match ITK-SNAP interactive behaviour: 4 HU of adjustment per
/// pixel of pointer displacement.
pub const WINDOW_LEVEL_SENSITIVITY: f64 = 4.0;

/// Minimum physically meaningful window width.
///
/// Guaranteed by [`clamp_window_width`]; passed through by
/// [`window_level_from_drag_delta`].
pub const MIN_WINDOW_WIDTH: f64 = 1.0;

// ── Core SSOT functions ────────────────────────────────────────────────────────

/// Clamp `width` to `[MIN_WINDOW_WIDTH, ∞)`.
///
/// The renderer divides by `width`; zero or negative widths are invalid and
/// must never reach the pixel pipeline.
#[inline]
pub fn clamp_window_width(width: f64) -> f64 {
    width.max(MIN_WINDOW_WIDTH)
}

/// Compute updated `(center, width)` from a drag delta `(dx, dy)`.
///
/// `dx` is the screen-space horizontal displacement (positive = right).
/// `dy` is the screen-space vertical displacement (positive = down, egui
/// convention).  The y-axis is inverted so dragging **up** increases center.
///
/// `sensitivity` controls HU adjustment per pixel; use
/// [`WINDOW_LEVEL_SENSITIVITY`] for the default.
///
/// # Arguments
/// - `original_center` — W/L center at drag start (HU or normalised units).
/// - `original_width`  — W/L width at drag start; must be ≥ [`MIN_WINDOW_WIDTH`].
/// - `dx`              — Horizontal drag displacement in screen pixels.
/// - `dy`              — Vertical drag displacement in screen pixels.
/// - `sensitivity`     — HU per pixel (positive).
///
/// # Returns
/// `(new_center, new_width)` with `new_width ≥ MIN_WINDOW_WIDTH`.
#[inline]
pub fn window_level_from_drag_delta(
    original_center: f64,
    original_width: f64,
    dx: f32,
    dy: f32,
    sensitivity: f64,
) -> (f64, f64) {
    let new_width = clamp_window_width(original_width + dx as f64 * sensitivity);
    let new_center = original_center - dy as f64 * sensitivity;
    (new_center, new_width)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Zero drag leaves (center, width) unchanged.
    #[test]
    fn zero_drag_is_identity() {
        let (c, w) = window_level_from_drag_delta(40.0, 400.0, 0.0, 0.0, WINDOW_LEVEL_SENSITIVITY);
        assert_eq!(c, 40.0);
        assert_eq!(w, 400.0);
    }

    /// Positive horizontal drag (right) increases window width proportionally.
    #[test]
    fn rightward_drag_increases_width() {
        let (c, w) = window_level_from_drag_delta(40.0, 400.0, 10.0, 0.0, WINDOW_LEVEL_SENSITIVITY);
        // new_width = 400.0 + 10.0 * 4.0 = 440.0
        assert_eq!(w, 440.0);
        assert_eq!(c, 40.0);
    }

    /// Negative horizontal drag (left) decreases window width, floored at MIN.
    #[test]
    fn leftward_drag_decreases_width_clamped() {
        // Starting width = 400, dx = -200 → raw = 400 − 800 = −400 → clamp to 1.0
        let (_c, w) =
            window_level_from_drag_delta(40.0, 400.0, -200.0, 0.0, WINDOW_LEVEL_SENSITIVITY);
        assert_eq!(w, MIN_WINDOW_WIDTH);
    }

    /// Positive vertical drag (down) decreases window center (y-axis inverted).
    #[test]
    fn downward_drag_decreases_center() {
        let (c, w) = window_level_from_drag_delta(40.0, 400.0, 0.0, 10.0, WINDOW_LEVEL_SENSITIVITY);
        // new_center = 40.0 − 10.0 * 4.0 = 0.0
        assert_eq!(c, 0.0);
        assert_eq!(w, 400.0);
    }

    /// Negative vertical drag (up) increases window center.
    #[test]
    fn upward_drag_increases_center() {
        let (c, _w) =
            window_level_from_drag_delta(40.0, 400.0, 0.0, -10.0, WINDOW_LEVEL_SENSITIVITY);
        // new_center = 40.0 − (−10.0) * 4.0 = 80.0
        assert_eq!(c, 80.0);
    }

    /// Width is monotone non-decreasing in dx for dx ≥ 0.
    #[test]
    fn width_monotone_nondecreasing_in_positive_dx() {
        let widths: Vec<f64> = (0..=20)
            .map(|i| {
                let (_, w) = window_level_from_drag_delta(
                    0.0,
                    100.0,
                    i as f32,
                    0.0,
                    WINDOW_LEVEL_SENSITIVITY,
                );
                w
            })
            .collect();
        for pair in widths.windows(2) {
            assert!(
                pair[1] >= pair[0],
                "width decreased: {} → {}",
                pair[0],
                pair[1]
            );
        }
    }

    /// Center is monotone decreasing in dy for dy > 0 (downward = lower center).
    #[test]
    fn center_monotone_decreasing_in_positive_dy() {
        let centers: Vec<f64> = (0..=20)
            .map(|i| {
                let (c, _) = window_level_from_drag_delta(
                    0.0,
                    100.0,
                    0.0,
                    i as f32,
                    WINDOW_LEVEL_SENSITIVITY,
                );
                c
            })
            .collect();
        for pair in centers.windows(2) {
            assert!(
                pair[1] <= pair[0],
                "center increased on downward drag: {} → {}",
                pair[0],
                pair[1]
            );
        }
    }

    /// Diagonal drag updates both center and width independently (no coupling).
    #[test]
    fn diagonal_drag_updates_center_and_width_independently() {
        let (c, w) =
            window_level_from_drag_delta(100.0, 200.0, 5.0, -5.0, WINDOW_LEVEL_SENSITIVITY);
        // new_width  = 200 + 5*4 = 220
        assert_eq!(w, 220.0);
        // new_center = 100 − (−5)*4 = 120
        assert_eq!(c, 120.0);
    }

    /// clamp_window_width enforces MIN_WINDOW_WIDTH.
    #[test]
    fn clamp_window_width_enforces_minimum() {
        assert_eq!(clamp_window_width(0.0), MIN_WINDOW_WIDTH);
        assert_eq!(clamp_window_width(-100.0), MIN_WINDOW_WIDTH);
        assert_eq!(clamp_window_width(0.5), MIN_WINDOW_WIDTH);
        assert_eq!(clamp_window_width(1.0), 1.0);
        assert_eq!(clamp_window_width(500.0), 500.0);
    }
}
