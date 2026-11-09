//! Histogram display geometry.
//!
//! Bars use a log-plus-one scale so high-count bins (air background in CT) do
//! not visually dominate the display. Both functions are O(1), allocation-free
//! and deterministic.

/// Compute the display height of a histogram bar using a log₁₊₁ scale.
///
/// # Contract
///
/// `height(count, peak, h) = ln(count + 1) / ln(peak + 1) × h`
///
/// where `ln` is the natural logarithm. The invariants are:
/// - `bar_height_log(peak, peak, h) = h` (tallest bar fills the canvas).
/// - `bar_height_log(0, peak, h) = 0.0` (empty bin has zero height).
/// - `bar_height_log(count, 0, h) = 0.0` (no data, no bars).
///
/// Uses `f64` internally to avoid significant rounding errors for large
/// counts (up to ~2³² per bin).
#[inline]
pub fn bar_height_log(count: u64, peak: u64, available_height: f32) -> f32 {
    if peak == 0 || count == 0 {
        return 0.0;
    }
    let ratio = ((count as f64 + 1.0).ln()) / ((peak as f64 + 1.0).ln());
    (ratio * available_height as f64) as f32
}

/// Map an intensity `value` from histogram range `[hist_min, hist_max]`
/// linearly to a pixel x-coordinate in `[x_left, x_right]`.
///
/// # Contract
///
/// `x(v) = x_left + (v − hist_min) / (hist_max − hist_min) × (x_right − x_left)`
///
/// clamped to `[x_left, x_right]`.
///
/// Returns `x_left` when the range is degenerate (`hist_max − hist_min < ε`).
#[inline]
pub fn wl_to_x(value: f32, hist_min: f32, hist_max: f32, x_left: f32, x_right: f32) -> f32 {
    let span = hist_max - hist_min;
    if span.abs() < f32::EPSILON {
        return x_left;
    }
    let t = (value - hist_min) / span;
    let x = x_left + t * (x_right - x_left);
    x.clamp(x_left, x_right)
}

#[cfg(test)]
#[path = "tests_histogram.rs"]
mod tests;
