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
pub fn x_to_intensity(x: f32, hist_min: f32, hist_max: f32, x_left: f32, x_right: f32) -> f32 {
    let canvas_span = x_right - x_left;
    let intensity_span = hist_max - hist_min;
    if canvas_span.abs() < f32::EPSILON || intensity_span.abs() < f32::EPSILON {
        return hist_min;
    }
    let t = ((x - x_left) / canvas_span).clamp(0.0, 1.0);
    hist_min + t * intensity_span
}

// ── wl_from_histogram_drag ─────────────────────────────────────────────────────

/// Canvas geometry and intensity range for histogram interaction.
///
/// Groups the four canvas/range parameters so `wl_from_histogram_drag` stays
/// within the function argument limit.
#[derive(Clone, Copy)]
pub struct HistogramCanvasGeometry {
    pub canvas_width: f32,
    pub canvas_height: f32,
    pub hist_min: f32,
    pub hist_max: f32,
}

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
    canvas: HistogramCanvasGeometry,
    current_center: f32,
    current_width: f32,
) -> (f32, f32) {
    let HistogramCanvasGeometry {
        canvas_width,
        canvas_height,
        hist_min,
        hist_max,
    } = canvas;
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

#[cfg(test)]
#[path = "tests_histogram_interact.rs"]
mod tests;
