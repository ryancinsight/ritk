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
use iris::color::{ColorMap, Normalized};

use crate::render::NamedColorMap;

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
    colormap: NamedColorMap,
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
        let t = Normalized::new(t).expect("invariant: generated colorbar coordinate is normalized");
        let [r, g, b, _] = colormap.sample(t).to_rgba8();
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
    painter.rect_stroke(
        bar_rect,
        Rounding::ZERO,
        Stroke::new(1.0_f32, Color32::GRAY),
    );

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
pub fn show_colorbar(ui: &mut Ui, colormap: NamedColorMap, center: f32, width: f32) {
    let bar_height = ui.available_height().max(MIN_HEIGHT);
    let (rect, _response) =
        ui.allocate_exact_size(vec2(COLORBAR_PANEL_WIDTH, bar_height), egui::Sense::hover());
    draw_colorbar(ui.painter(), rect, colormap, center, width);
}

#[cfg(test)]
#[path = "tests_colorbar.rs"]
mod tests;
