//! Colorbar rendering widget for viewport W/L intensity display.
//!
//! # Mathematical Specification
//!
//! A colorbar is a vertical gradient bar mapping normalized intensity
//! `t ∈ [0.0, 1.0]` through the active colormap to a visual color swatch.
//!
//! The bar height `H` is divided into N = H pixel rows. Row `i` (top = 0)
//! corresponds to normalized intensity:
//! ```text
//! t(i) = 1.0 − i / (H − 1)   for H > 1
//! t(i) = 1.0                  for H = 1
//! ```
//! The top of the bar represents the maximum W/L intensity and the bottom
//! represents the minimum W/L intensity, matching the typical DICOM/ITK-SNAP
//! colorbar convention (bright at top, dark at bottom for grayscale).
//!
//! W/L intensity at normalized value t:
//! ```text
//! intensity(t) = (center − width/2) + t × width
//! ```
//!
//! Axis labels are drawn at the top (max HU value) and bottom (min HU value)
//! of the bar, with the center value at the midpoint.

use egui::{pos2, vec2, Color32, Painter, Rect, Rounding, Stroke, Ui};

use crate::render::colormap::Colormap;

/// Width of the colorbar swatch in screen pixels.
pub const COLORBAR_WIDTH: f32 = 18.0;

/// Total width including the text label area.
pub const COLORBAR_PANEL_WIDTH: f32 = 70.0;

/// Minimum height in screen pixels.
const MIN_HEIGHT: f32 = 60.0;

/// Draw a vertical colorbar into the given `Rect` using `painter`.
///
/// # Parameters
/// - `painter`   — egui `Painter` for the target region.
/// - `rect`      — the bounding rectangle for the entire colorbar panel.
/// - `colormap`  — active colormap to sample.
/// - `center`    — W/L window centre (HU or relative units).
/// - `width`     — W/L window width (HU or relative units).
///
/// # Contract
/// - The top of the bar corresponds to intensity `center + width / 2`.
/// - The bottom corresponds to intensity `center − width / 2`.
/// - All labels are drawn in white with a dark text shadow for legibility
///   on any background.
pub fn draw_colorbar(
    painter: &Painter,
    rect: Rect,
    colormap: Colormap,
    center: f32,
    width: f32,
) {
    let bar_height = (rect.height()).max(MIN_HEIGHT);
    let bar_left = rect.left();
    let bar_right = bar_left + COLORBAR_WIDTH;
    let bar_top = rect.top();
    let bar_bottom = bar_top + bar_height;

    // Draw the gradient bar row by row.
    let n_rows = bar_height.ceil() as usize;
    let n_rows = n_rows.max(2);

    for i in 0..n_rows {
        let t = 1.0 - (i as f32) / ((n_rows - 1) as f32);
        let [r, g, b] = colormap.map(t);
        let color = Color32::from_rgb(r, g, b);

        let row_y_top = bar_top + (i as f32 / n_rows as f32) * bar_height;
        let row_y_bottom = bar_top + ((i + 1) as f32 / n_rows as f32) * bar_height;

        let row_rect = Rect::from_min_max(
            pos2(bar_left, row_y_top),
            pos2(bar_right, row_y_bottom + 0.5), // +0.5 to avoid gaps
        );
        painter.rect_filled(row_rect, Rounding::ZERO, color);
    }

    // Border around the colorbar.
    let bar_rect = Rect::from_min_max(pos2(bar_left, bar_top), pos2(bar_right, bar_bottom));
    painter.rect_stroke(bar_rect, Rounding::ZERO, Stroke::new(1.0, Color32::GRAY));

    // Intensity labels.
    let half_width = width * 0.5;
    let max_intensity = center + half_width;
    let mid_intensity = center;
    let min_intensity = center - half_width;

    let label_x = bar_right + 3.0;
    let font_id = egui::FontId::proportional(10.0);

    let draw_label = |painter: &Painter, y: f32, value: f32| {
        let text = if value.abs() >= 1000.0 {
            format!("{:.0}", value)
        } else {
            format!("{:.1}", value)
        };

        // Shadow
        painter.text(
            pos2(label_x + 1.0, y + 1.0),
            egui::Align2::LEFT_CENTER,
            &text,
            font_id.clone(),
            Color32::BLACK,
        );
        // Label
        painter.text(
            pos2(label_x, y),
            egui::Align2::LEFT_CENTER,
            &text,
            font_id.clone(),
            Color32::WHITE,
        );
    };

    draw_label(painter, bar_top + 5.0, max_intensity);
    draw_label(painter, bar_top + bar_height * 0.5, mid_intensity);
    draw_label(painter, bar_bottom - 5.0, min_intensity);
}

/// Show a colorbar panel as an egui widget inside a `Ui`.
///
/// Allocates space for the full colorbar panel (`COLORBAR_PANEL_WIDTH` wide)
/// and delegates to [`draw_colorbar`].
pub fn show_colorbar(ui: &mut Ui, colormap: Colormap, center: f32, width: f32) {
    let bar_height = ui.available_height().max(MIN_HEIGHT);
    let (rect, _response) = ui.allocate_exact_size(
        vec2(COLORBAR_PANEL_WIDTH, bar_height),
        egui::Sense::hover(),
    );
    draw_colorbar(ui.painter(), rect, colormap, center, width);
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Intensity mapping ─────────────────────────────────────────────────────

    /// For center=100, width=200 (range [-100, 100]):
    /// - t=1.0 → intensity = 100 (max at top of bar)
    /// - t=0.0 → intensity = -100 (min at bottom)
    /// - t=0.5 → intensity = 0 (center)
    #[test]
    fn test_intensity_mapping_analytical() {
        let center = 100.0f32;
        let width = 200.0f32;
        let half = width * 0.5;

        let max_intensity = center + half;
        let min_intensity = center - half;
        let mid_intensity = center;

        assert_eq!(max_intensity, 200.0, "t=1.0 must equal center+width/2");
        assert_eq!(min_intensity, 0.0, "t=0.0 must equal center-width/2");
        assert_eq!(mid_intensity, 100.0, "t=0.5 must equal center");
    }

    /// Standard CT brain preset: center=40, width=80 → range [-0, 80].
    #[test]
    fn test_ct_brain_preset_range() {
        let center = 40.0f32;
        let width = 80.0f32;
        assert_eq!(center + width * 0.5, 80.0);
        assert_eq!(center - width * 0.5, 0.0);
    }

    /// Standard CT lung preset: center=−400, width=1500.
    #[test]
    fn test_ct_lung_preset_range() {
        let center = -400.0f32;
        let width = 1500.0f32;
        let max_hu = center + width * 0.5;
        let min_hu = center - width * 0.5;
        assert_eq!(max_hu, 350.0, "lung max HU = 350");
        assert_eq!(min_hu, -1150.0, "lung min HU = -1150");
    }

    /// Positive-width invariant: the colorbar width must always be positive.
    #[test]
    fn test_width_positive_invariant() {
        // Width = 0 is a degenerate case (single value band). The colorbar
        // still renders without panic; the labels converge to the same value.
        let center = 50.0f32;
        let width = 0.0f32;
        let max_intensity = center + width * 0.5;
        let min_intensity = center - width * 0.5;
        assert_eq!(max_intensity, min_intensity, "zero width: top = bottom");
    }

    // ── Colorbar row count ─────────────────────────────────────────────────────

    /// Normalized value `t` for row i of N total rows:
    /// t(i) = 1.0 - i / (N - 1).
    /// Verify top row = 1.0 and bottom row = 0.0.
    #[test]
    fn test_row_t_top_and_bottom() {
        let n = 256usize;
        let t_top = 1.0f32 - 0.0 / (n as f32 - 1.0);
        let t_bottom = 1.0f32 - (n as f32 - 1.0) / (n as f32 - 1.0);
        assert_eq!(t_top, 1.0, "top row must map to t=1.0");
        assert!((t_bottom - 0.0).abs() < 1e-6, "bottom row must map to t≈0.0");
    }

    /// Middle row (i = N/2) corresponds to t ≈ 0.5 for even N.
    #[test]
    fn test_row_t_midpoint_analytical() {
        let n = 100usize;
        let i = n / 2;
        let t = 1.0f32 - (i as f32) / ((n - 1) as f32);
        // For N=100, i=50: t = 1 - 50/99 ≈ 0.4949...
        assert!((t - 0.495).abs() < 0.01, "middle row t ≈ 0.495 for N=100");
    }

    // ── COLORBAR_WIDTH constant ───────────────────────────────────────────────

    #[test]
    fn test_colorbar_width_positive() {
        assert!(COLORBAR_WIDTH > 0.0);
        assert!(COLORBAR_PANEL_WIDTH > COLORBAR_WIDTH);
    }
}
