//! Voxel intensity histogram widget and rendering utilities.
//!
//! # Overview
//!
//! Provides [`draw_histogram`], a stateless function that renders an
//! intensity histogram bar chart with a W/L range overlay into any egui
//! `Ui`. The histogram bars use a log₁₊₁ scale so that high-count bins
//! (e.g. air background in CT) do not visually dominate the display.
//!
//! # Pure helper functions (tested independently)
//!
//! | Function | Contract |
//! |----------|---------|
//! | [`bar_height_log`] | log₁₊₁-scaled bar height from count and peak |
//! | [`wl_to_x`] | linear mapping of an intensity value to an x-pixel |
//!
//! Both functions are O(1), allocation-free, and deterministic; they are
//! suitable for per-frame calls in the egui render loop.
//!
//! # ITK-SNAP parity
//!
//! ITK-SNAP displays a histogram with the current W/L window shaded.
//! [`draw_histogram`] reproduces this with: grey bars (log-scaled) + blue
//! semi-transparent W/L band + orange centre line.

use egui::{Color32, Painter, Pos2, Rect, Stroke, Ui};

use crate::render::histogram::{histogram_peak_count, Histogram};
use crate::ui::histogram_interact::{wl_center_from_click, wl_from_histogram_drag, HistogramCanvasGeometry};

// ── visual constants ───────────────────────────────────────────────────────────

/// Background fill for the histogram canvas.
const BACKGROUND_COLOR: Color32 = Color32::from_gray(20);

/// Fill colour for histogram bars.
const BAR_COLOR: Color32 = Color32::from_gray(160);

/// Fill colour for the W/L range band (semi-transparent blue).
const WL_BAND_COLOR: Color32 = Color32::from_rgba_premultiplied(40, 100, 200, 50);

/// Stroke colour for the W/L range border.
const WL_BORDER_COLOR: Color32 = Color32::from_rgb(80, 150, 255);

/// Stroke colour for the W/L centre line.
const WL_CENTER_COLOR: Color32 = Color32::from_rgb(255, 190, 60);

/// Height in pixels allocated for the histogram canvas.
const CANVAS_HEIGHT: f32 = 80.0;

// ── bar_height_log ─────────────────────────────────────────────────────────────

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

// ── wl_to_x ───────────────────────────────────────────────────────────────────

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

// ── draw_histogram ─────────────────────────────────────────────────────────────

/// Render an intensity histogram with a W/L range overlay into `ui`.
///
/// Allocates a rectangle of height [`CANVAS_HEIGHT`] and full available
/// width, then:
///
/// 1. Fills the background.
/// 2. Draws one bar per bin (log₁₊₁-scaled height).
/// 3. Overlays the W/L window band (centre ± width/2) as a semi-transparent
///    blue rectangle with an orange centre line.
/// 4. Draws `min` and `max` intensity labels below the canvas.
///
/// Does nothing if the histogram is empty (zero bins).
///
/// # Return value
///
/// Returns `Some((new_center, new_width))` when the user has interacted
/// with the canvas during this frame (drag or click), and `None` otherwise.
/// The caller is responsible for clamping and applying the returned values.
pub fn draw_histogram(
    histogram: &Histogram,
    window_center: f32,
    window_width: f32,
    ui: &mut Ui,
) -> Option<(f32, f32)> {
    if histogram.bins == 0 {
        ui.label("(no histogram — load a study first)");
        return None;
    }

    let avail_width = ui.available_width().max(1.0);
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(avail_width, CANVAS_HEIGHT),
        egui::Sense::click_and_drag(),
    );

    let painter: Painter = ui.painter_at(rect);

    // ── background ────────────────────────────────────────────────────────────
    painter.rect_filled(rect, 2.0, BACKGROUND_COLOR);

    // ── bars ──────────────────────────────────────────────────────────────────
    let peak = histogram_peak_count(histogram);
    let n = histogram.bins;
    let bar_w = rect.width() / n as f32;

    for (i, &count) in histogram.counts.iter().enumerate() {
        let h = bar_height_log(count, peak, rect.height());
        if h <= 0.0 {
            continue;
        }
        let x0 = rect.left() + i as f32 * bar_w;
        let x1 = (x0 + bar_w).min(rect.right());
        let bar_rect = Rect::from_min_max(
            Pos2::new(x0, rect.bottom() - h),
            Pos2::new(x1, rect.bottom()),
        );
        painter.rect_filled(bar_rect, 0.0, BAR_COLOR);
    }

    // ── W/L range overlay ─────────────────────────────────────────────────────
    let hist_min = histogram.min();
    let hist_max = histogram.max();
    let span = hist_max - hist_min;
    if span.abs() > f32::EPSILON {
        let l_val = window_center - window_width / 2.0;
        let r_val = window_center + window_width / 2.0;

        let x_l = wl_to_x(l_val, hist_min, hist_max, rect.left(), rect.right());
        let x_r = wl_to_x(r_val, hist_min, hist_max, rect.left(), rect.right());
        let x_c = wl_to_x(window_center, hist_min, hist_max, rect.left(), rect.right());

        // Semi-transparent band
        if (x_r - x_l).abs() > 0.5 {
            let band =
                Rect::from_min_max(Pos2::new(x_l, rect.top()), Pos2::new(x_r, rect.bottom()));
            painter.rect_filled(band, 0.0, WL_BAND_COLOR);
            painter.rect_stroke(band, 0.0, Stroke::new(1.0_f32, WL_BORDER_COLOR));
        }

        // Centre line
        painter.line_segment(
            [Pos2::new(x_c, rect.top()), Pos2::new(x_c, rect.bottom())],
            Stroke::new(1.0_f32, WL_CENTER_COLOR),
        );
    }

    // ── axis labels ───────────────────────────────────────────────────────────
    ui.horizontal(|ui| {
        ui.label(
            egui::RichText::new(format!("{:.0}", hist_min))
                .size(10.0)
                .color(Color32::GRAY),
        );
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(
                egui::RichText::new(format!("{:.0}", hist_max))
                    .size(10.0)
                    .color(Color32::GRAY),
            );
        });
    });

    // ── interaction ───────────────────────────────────────────────────────────
    if response.dragged() {
        let delta = response.drag_delta();
        let (new_c, new_w) = wl_from_histogram_drag(
            delta.x,
            delta.y,
            HistogramCanvasGeometry {
                canvas_width: rect.width(),
                canvas_height: rect.height(),
                hist_min,
                hist_max,
            },
            window_center,
            window_width,
        );
        return Some((new_c, new_w));
    }
    if response.clicked() {
        if let Some(pos) = response.interact_pointer_pos() {
            let new_c = wl_center_from_click(pos.x, hist_min, hist_max, rect.left(), rect.right());
            return Some((new_c, window_width));
        }
    }
    None
}

// ── tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── bar_height_log tests ──────────────────────────────────────────────────

    /// Peak count returns exactly available_height.
    ///
    /// Proof: ln(peak+1)/ln(peak+1) = 1.0; 1.0 × h = h.
    #[test]
    fn bar_height_log_at_peak_returns_available_height() {
        let h = bar_height_log(100, 100, 80.0);
        assert!((h - 80.0).abs() < 1e-4, "expected 80.0, got {h}");
    }

    /// Zero count returns 0.0 regardless of peak.
    ///
    /// Proof: ln(0+1) = ln(1) = 0.0; 0.0/anything = 0.0.
    #[test]
    fn bar_height_log_zero_count_returns_zero() {
        let h = bar_height_log(0, 100, 80.0);
        assert_eq!(h, 0.0, "zero count must produce 0 height");
    }

    /// Zero peak returns 0.0 (guard against division).
    #[test]
    fn bar_height_log_zero_peak_returns_zero() {
        let h = bar_height_log(50, 0, 80.0);
        assert_eq!(h, 0.0, "zero peak must produce 0 height");
    }

    /// Half-peak bin has height strictly between 0 and available_height.
    ///
    /// Analytical: ln(51) / ln(101) × 80.0 ≈ 3.932 / 4.615 × 80.0 ≈ 68.18
    #[test]
    fn bar_height_log_half_peak_is_strictly_between_zero_and_max() {
        let h = bar_height_log(50, 100, 80.0);
        assert!(
            h > 0.0 && h < 80.0,
            "half-peak height {h} must be in (0, 80)"
        );
        // Analytical: ln(51)/ln(101) × 80.0
        let expected = (51f64.ln() / 101f64.ln() * 80.0) as f32;
        assert!(
            (h - expected).abs() < 1e-3,
            "expected {expected:.4}, got {h:.4}"
        );
    }

    // ── wl_to_x tests ────────────────────────────────────────────────────────

    /// Centre value maps to the exact mid-point of the pixel range.
    ///
    /// Analytical: t = (50 − 0) / (100 − 0) = 0.5; x = 0.0 + 0.5 × 200.0 = 100.0.
    #[test]
    fn wl_to_x_centre_value_maps_to_midpoint() {
        let x = wl_to_x(50.0, 0.0, 100.0, 0.0, 200.0);
        assert!((x - 100.0).abs() < 1e-4, "expected x=100.0, got {x}");
    }

    /// Value below hist_min clamps to x_left.
    #[test]
    fn wl_to_x_below_range_clamps_to_left() {
        let x = wl_to_x(-10.0, 0.0, 100.0, 10.0, 110.0);
        assert!(
            (x - 10.0).abs() < 1e-4,
            "expected x=10.0 (clamped), got {x}"
        );
    }

    /// Value above hist_max clamps to x_right.
    #[test]
    fn wl_to_x_above_range_clamps_to_right() {
        let x = wl_to_x(110.0, 0.0, 100.0, 10.0, 110.0);
        assert!(
            (x - 110.0).abs() < 1e-4,
            "expected x=110.0 (clamped), got {x}"
        );
    }
}
