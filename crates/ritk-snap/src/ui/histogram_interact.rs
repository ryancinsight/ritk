//! Histogram canvas interaction SSOT.
//!
//! # Overview
//!
//! Provides pure functions for translating pointer events on the histogram
//! canvas into updated window/level (W/L) parameters, and for mapping a
//! canvas x-coordinate back to its corresponding intensity value.
//!
//! # Mathematical specification
//!
//! The histogram canvas renders intensity range `[hist_min, hist_max]` over
//! pixel x-range `[x_left, x_right]`.  The inverse linear map is:
//!
//! ```text
//! t   = clamp((x − x_left) / (x_right − x_left), 0, 1)
//! v(x) = hist_min + t × (hist_max − hist_min)
//! ```
//!
//! **Drag interaction** maps a 2D pointer delta `(dx, dy)` to:
//!
//! ```text
//! Δcenter = (dx / canvas_width)  × (hist_max − hist_min)   [horizontal shift]
//! scale   = 1 − dy / canvas_height                          [up = narrow, down = widen]
//! new_width = max(1, current_width × scale)
//! ```
//!
//! **Click interaction** (no drag threshold crossed) sets:
//!
//! ```text
//! new_center = v(click_x)
//! ```
//!
//! # ITK-SNAP parity
//!
//! ITK-SNAP's histogram widget supports direct mouse interaction to adjust
//! window/level by dragging horizontally (center) and vertically (width).
//! [`wl_from_histogram_drag`] reproduces the ITK-SNAP proportional mapping.
//!
//! # Formal invariants
//!
//! - `x_to_intensity(x_left, ...)  = hist_min`
//! - `x_to_intensity(x_right, ...) = hist_max`
//! - `x_to_intensity` ∘ `wl_to_x`  = identity on `[hist_min, hist_max]`
//! - `wl_from_histogram_drag(0, 0, ...) = (current_center, current_width)` (zero delta invariant)
//! - `new_width ≥ 1.0` always (enforced by clamp)

// ── x_to_intensity ─────────────────────────────────────────────────────────────

/// Map a canvas x-coordinate back to its corresponding intensity value.
///
/// This is the inverse of [`crate::ui::histogram::wl_to_x`]:
///
/// ```text
/// t   = clamp((x − x_left) / (x_right − x_left), 0.0, 1.0)
/// v   = hist_min + t × (hist_max − hist_min)
/// ```
///
/// # Degenerate input
///
/// Returns `hist_min` when `x_right − x_left < ε` or `hist_max − hist_min < ε`.
#[inline]
pub fn x_to_intensity(
    x: f32,
    hist_min: f32,
    hist_max: f32,
    x_left: f32,
    x_right: f32,
) -> f32 {
    let canvas_span = x_right - x_left;
    let intensity_span = hist_max - hist_min;
    if canvas_span.abs() < f32::EPSILON || intensity_span.abs() < f32::EPSILON {
        return hist_min;
    }
    let t = ((x - x_left) / canvas_span).clamp(0.0, 1.0);
    hist_min + t * intensity_span
}

// ── wl_from_histogram_drag ─────────────────────────────────────────────────────

/// Compute updated `(window_center, window_width)` from a histogram canvas
/// pointer drag delta.
///
/// # Contract
///
/// - `dx > 0` shifts center toward higher intensity: `Δcenter = (dx/canvas_width) × span`.
/// - `dy < 0` (upward drag) narrows the window: `scale = 1 − dy/canvas_height > 1`.
/// - `dy > 0` (downward drag) widens the window: `scale = 1 − dy/canvas_height < 1`.
/// - `new_width ≥ 1.0` always, regardless of how extreme the drag is.
///
/// Returns `(current_center, current_width)` unchanged when `canvas_width ≤ 0`
/// or `intensity_span ≤ 0`.
pub fn wl_from_histogram_drag(
    dx: f32,
    dy: f32,
    canvas_width: f32,
    canvas_height: f32,
    hist_min: f32,
    hist_max: f32,
    current_center: f32,
    current_width: f32,
) -> (f32, f32) {
    let intensity_span = hist_max - hist_min;
    if intensity_span <= 0.0 || canvas_width <= 0.0 {
        return (current_center, current_width);
    }
    // Horizontal drag: proportional center shift
    let center_delta = (dx / canvas_width) * intensity_span;
    let new_center = current_center + center_delta;

    // Vertical drag: multiplicative width scale
    let scale = if canvas_height > 0.0 {
        1.0 - dy / canvas_height
    } else {
        1.0
    };
    let new_width = (current_width * scale).max(1.0);

    (new_center, new_width)
}

// ── wl_center_from_click ───────────────────────────────────────────────────────

/// Compute a new window center from a single click on the histogram canvas.
///
/// Equivalent to `x_to_intensity(click_x, ...)`.  Width is unchanged; the
/// caller retains the existing window width.
#[inline]
pub fn wl_center_from_click(
    click_x: f32,
    hist_min: f32,
    hist_max: f32,
    x_left: f32,
    x_right: f32,
) -> f32 {
    x_to_intensity(click_x, hist_min, hist_max, x_left, x_right)
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── x_to_intensity ──────────────────────────────────────────────────────────

    /// Left edge of canvas maps to hist_min.
    #[test]
    fn x_to_intensity_left_edge_returns_hist_min() {
        let v = x_to_intensity(10.0, -1000.0, 3000.0, 10.0, 210.0);
        assert_eq!(v, -1000.0);
    }

    /// Right edge of canvas maps to hist_max.
    #[test]
    fn x_to_intensity_right_edge_returns_hist_max() {
        let v = x_to_intensity(210.0, -1000.0, 3000.0, 10.0, 210.0);
        assert_eq!(v, 3000.0);
    }

    /// Midpoint of canvas maps to midpoint of intensity range.
    ///
    /// Analytical: t = 0.5, v = -1000 + 0.5 × 4000 = 1000.
    #[test]
    fn x_to_intensity_midpoint_maps_analytically() {
        let v = x_to_intensity(110.0, -1000.0, 3000.0, 10.0, 210.0);
        let expected = -1000.0_f32 + 0.5 * 4000.0_f32;
        assert!((v - expected).abs() < 1e-3, "got {v}, expected {expected}");
    }

    /// x below x_left clamps to hist_min.
    #[test]
    fn x_to_intensity_below_left_clamps_to_min() {
        let v = x_to_intensity(0.0, -1000.0, 3000.0, 10.0, 210.0);
        assert_eq!(v, -1000.0);
    }

    /// x above x_right clamps to hist_max.
    #[test]
    fn x_to_intensity_above_right_clamps_to_max() {
        let v = x_to_intensity(999.0, -1000.0, 3000.0, 10.0, 210.0);
        assert_eq!(v, 3000.0);
    }

    /// Degenerate canvas (x_left == x_right) returns hist_min.
    #[test]
    fn x_to_intensity_degenerate_canvas_returns_min() {
        let v = x_to_intensity(50.0, -500.0, 500.0, 50.0, 50.0);
        assert_eq!(v, -500.0);
    }

    /// Degenerate intensity span (min == max) returns hist_min.
    #[test]
    fn x_to_intensity_degenerate_span_returns_min() {
        let v = x_to_intensity(100.0, 500.0, 500.0, 0.0, 200.0);
        assert_eq!(v, 500.0);
    }

    // ── wl_from_histogram_drag ──────────────────────────────────────────────────

    /// Zero drag delta leaves center and width unchanged.
    #[test]
    fn wl_from_drag_zero_delta_identity() {
        let (c, w) = wl_from_histogram_drag(0.0, 0.0, 200.0, 80.0, -1000.0, 3000.0, 40.0, 400.0);
        assert_eq!(c, 40.0);
        assert_eq!(w, 400.0);
    }

    /// Rightward drag (dx = canvas_width/2) shifts center by span/2.
    ///
    /// Analytical: Δcenter = (100/200) × 4000 = 2000; new center = 40 + 2000 = 2040.
    #[test]
    fn wl_from_drag_rightward_shifts_center() {
        let (c, _) =
            wl_from_histogram_drag(100.0, 0.0, 200.0, 80.0, -1000.0, 3000.0, 40.0, 400.0);
        let expected = 40.0_f32 + 0.5 * 4000.0_f32;
        assert!((c - expected).abs() < 1e-3, "got {c}, expected {expected}");
    }

    /// Leftward drag shifts center toward lower intensity.
    #[test]
    fn wl_from_drag_leftward_shifts_center_negative() {
        let (c, _) =
            wl_from_histogram_drag(-200.0, 0.0, 200.0, 80.0, -1000.0, 3000.0, 40.0, 400.0);
        // Δcenter = (-200/200) × 4000 = -4000; new center = 40 - 4000 = -3960
        let expected = 40.0_f32 - 4000.0_f32;
        assert!((c - expected).abs() < 1e-3, "got {c}, expected {expected}");
    }

    /// Upward drag (dy < 0) narrows the window width.
    ///
    /// Analytical: scale = 1 - (-40/80) = 1.5; new_width = 400 × 1.5 = 600.
    #[test]
    fn wl_from_drag_upward_narrows_width() {
        let (_, w) =
            wl_from_histogram_drag(0.0, -40.0, 200.0, 80.0, -1000.0, 3000.0, 40.0, 400.0);
        let expected = 400.0_f32 * 1.5_f32;
        assert!((w - expected).abs() < 1e-3, "got {w}, expected {expected}");
    }

    /// Downward drag (dy > 0) widens the window width.
    ///
    /// Analytical: scale = 1 - (80/80) = 0.0; new_width = max(1, 400 × 0.0) = 1.
    #[test]
    fn wl_from_drag_extreme_downward_clamps_to_min_width() {
        let (_, w) =
            wl_from_histogram_drag(0.0, 80.0, 200.0, 80.0, -1000.0, 3000.0, 40.0, 400.0);
        assert!(w >= 1.0, "width {w} below minimum");
        assert!((w - 1.0).abs() < 1e-3, "got {w}, expected 1.0");
    }

    /// Degenerate canvas_width returns input unchanged.
    #[test]
    fn wl_from_drag_degenerate_canvas_width_identity() {
        let (c, w) = wl_from_histogram_drag(10.0, 5.0, 0.0, 80.0, -1000.0, 3000.0, 40.0, 400.0);
        assert_eq!(c, 40.0);
        assert_eq!(w, 400.0);
    }

    /// Degenerate intensity span (hist_min == hist_max) returns input unchanged.
    #[test]
    fn wl_from_drag_degenerate_span_identity() {
        let (c, w) =
            wl_from_histogram_drag(50.0, 20.0, 200.0, 80.0, 1000.0, 1000.0, 40.0, 400.0);
        assert_eq!(c, 40.0);
        assert_eq!(w, 400.0);
    }

    // ── wl_center_from_click ─────────────────────────────────────────────────

    /// Click at canvas left gives hist_min.
    #[test]
    fn wl_center_from_click_left_returns_min() {
        let c = wl_center_from_click(0.0, -500.0, 500.0, 0.0, 100.0);
        assert_eq!(c, -500.0);
    }

    /// Click at canvas right gives hist_max.
    #[test]
    fn wl_center_from_click_right_returns_max() {
        let c = wl_center_from_click(100.0, -500.0, 500.0, 0.0, 100.0);
        assert_eq!(c, 500.0);
    }

    /// Click at canvas midpoint gives intensity midpoint.
    #[test]
    fn wl_center_from_click_midpoint_analytical() {
        let c = wl_center_from_click(50.0, -500.0, 500.0, 0.0, 100.0);
        assert!((c - 0.0_f32).abs() < 1e-3, "got {c}, expected 0.0");
    }
}
