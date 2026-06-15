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

#[cfg(test)]
#[path = "tests_window_level.rs"]
mod tests;
