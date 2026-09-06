//! DICOM-style 4-corner information overlay and patient-orientation labels.
//!
//! # Overlay layout
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │ Patient Name          Series Desc       │
//! │ Patient ID            Modality / Date   │
//! │                                         │
//! │                                         │
//! │ Slice N/M             W: WWWW C: CCCC   │
//! │ Spacing               Zoom: ZZZ%        │
//! │ Dimensions            Value: VVVV          │
//! └─────────────────────────────────────────┘
//! ```
//!
//! # Orientation label convention
//!
//! For each axis the displayed labels are derived from the direction cosine
//! matrix columns:
//!
//! | Axis | Fixed index | Left edge | Right edge | Top edge | Bottom edge |
//! |------|-------------|-----------|------------|----------|-------------|
//! | 0    | depth d     | R or L    | L or R     | A or P   | P or A      |
//! | 1    | row r       | R or L    | L or R     | S or I   | I or S      |
//! | 2    | col c       | A or P    | P or A     | S or I   | I or S      |
//!
//! Labels are determined by the dominant component of the relevant direction
//! cosine column.

use egui::{Align2, Color32, FontId, Painter, Pos2, Rect};

mod details;

use super::anatomical_label_for_axis;
use crate::render::slice_render::WindowLevel;
use crate::LoadedVolume;

// ── constants ──────────────────────────────────────────────────────────────────

/// Font size for overlay text (points).
const OVERLAY_FONT_SIZE: f32 = 12.0;

/// Colour for all overlay text.
const OVERLAY_TEXT_COLOR: Color32 = Color32::from_rgb(255, 255, 160); // warm yellow

/// Colour for orientation labels (brighter white for visibility).
const ORIENT_LABEL_COLOR: Color32 = Color32::WHITE;

/// Margin from the viewport edge (pixels).
const MARGIN: f32 = 6.0;

/// Backing inset keeps antialiased glyph edges away from image pixels.
const TEXT_PADDING: f32 = 2.0;
const ORIENTATION_FONT_SIZE: f32 = 14.0;

// ── OverlayRenderer ───────────────────────────────────────────────────────────

/// Per-frame display state passed to `OverlayRenderer::draw`.
///
/// Groups axis/slice/WL/zoom/cursor parameters so the function stays within
/// the argument limit.
pub struct OverlayContext {
    pub axis: usize,
    pub slice_index: usize,
    pub wl: WindowLevel,
    pub zoom: f32,
    pub cursor_value: Option<f32>,
    pub pointer_intensity: f32,
    pub pointer_suv: Option<f32>,
    pub cursor_suv: Option<f32>,
}

/// Renders DICOM-style information overlays on a viewport rectangle.
///
/// All methods are stateless; call them with the current render state on
/// every frame.
pub struct OverlayRenderer;

impl OverlayRenderer {
    // ── Public API ────────────────────────────────────────────────────────────

    /// Draw the standard 4-corner DICOM text overlay.
    ///
    /// Corner assignments:
    ///
    /// | Corner      | Content                                          |
    /// |-------------|--------------------------------------------------|
    /// | Top-left    | Patient Name, Patient ID                         |
    /// | Top-right   | Series description, Modality, Study date         |
    /// | Bottom-left | Slice N/M, Voxel spacing, Image dimensions       |
    /// | Bottom-right| Window width/centre, Zoom %, cursor value    |
    ///
    /// # Parameters
    /// - `painter`      — egui painter for the viewport.
    /// - `rect`         — viewport rectangle in screen coordinates.
    /// - `volume`       — loaded volume supplying metadata.
    /// - `ctx`          — axis, slice, presentation and sampled-value context.
    ///
    /// Returns all annotation text when the measured layout cannot fit without
    /// overlap. The caller must expose it with [`Self::show_details`]. Orientation
    /// labels participate in this same layout, rather than painting independently.
    #[must_use]
    pub fn draw(
        painter: &Painter,
        rect: Rect,
        volume: &LoadedVolume,
        ctx: OverlayContext,
    ) -> Option<String> {
        let rect = rect.intersect(painter.clip_rect());
        let OverlayContext {
            axis,
            slice_index,
            wl,
            zoom,
            cursor_value,
            pointer_intensity,
            pointer_suv,
            cursor_suv,
        } = ctx;
        let [depth, rows, cols] = volume.shape;

        // ── Top-left: patient information ──────────────────────────────────
        let patient_name = volume
            .patient_name
            .as_deref()
            .unwrap_or("(no name)")
            .to_string();
        let patient_id = volume.patient_id.as_deref().unwrap_or("").to_string();

        let tl_text = if patient_id.is_empty() {
            patient_name
        } else {
            format!("{}\nID: {}", patient_name, patient_id)
        };
        // ── Top-right: series / modality / date ────────────────────────────
        let series_desc = volume
            .series_description
            .as_deref()
            .unwrap_or("")
            .to_string();
        let modality = volume.modality.as_deref().unwrap_or("").to_string();
        let study_date = volume.study_date.as_deref().unwrap_or("").to_string();

        let mut tr_lines: Vec<String> = Vec::new();
        if !series_desc.is_empty() {
            tr_lines.push(series_desc);
        }
        if !modality.is_empty() {
            tr_lines.push(modality);
        }
        if !study_date.is_empty() {
            tr_lines.push(format!("Date: {}", study_date));
        }
        // ── Bottom-left: slice, spacing, dimensions ────────────────────────
        let (total_slices, dim_w, dim_h) = match axis {
            0 => (depth, cols, rows),
            1 => (rows, cols, depth),
            _ => (cols, rows, depth),
        };
        let axis_name = anatomical_label_for_axis(Some(volume), axis);

        let [dz, dy, dx] = volume.spacing;
        let spacing_str = format!("{:.2} × {:.2} × {:.2} mm", dx, dy, dz);
        let dims_str = format!("{}×{}×{}", cols, rows, depth);
        let slice_str = format!(
            "{}: {}/{}   {}×{}",
            axis_name,
            slice_index + 1,
            total_slices,
            dim_w,
            dim_h
        );
        let bl_text = format!(
            "{}\nSpacing: {}\nDims: {}",
            slice_str, spacing_str, dims_str
        );
        // ── Bottom-right: W/L, zoom, cursor, pointer ──────────────────────
        let wl_str = format!("W:{:.0} C:{:.0}", wl.width, wl.center);
        let zoom_str = format!("Zoom: {:.0}%", zoom * 100.0);
        let cursor_val_str = format_cursor_str(cursor_value, cursor_suv);
        let pointer_val_str = format_pointer_str(pointer_intensity, pointer_suv);
        let br_lines: Vec<&str> = [
            &wl_str as &str,
            &zoom_str as &str,
            &cursor_val_str as &str,
            &pointer_val_str as &str,
        ]
        .iter()
        .filter(|s| !s.is_empty())
        .copied()
        .collect();
        let labels = orientation_labels(axis, &volume.direction);
        let blocks = [
            (Align2::LEFT_TOP, tl_text),
            (Align2::RIGHT_TOP, tr_lines.join("\n")),
            (Align2::LEFT_BOTTOM, bl_text),
            (Align2::RIGHT_BOTTOM, br_lines.join("\n")),
        ];
        let mut layout = Vec::with_capacity(8);
        for (anchor, text) in &blocks {
            if !text.is_empty() {
                let galley = Self::layout_corner(painter, rect, text, OVERLAY_TEXT_COLOR);
                let bounds = anchor.anchor_size(Self::anchor_pos(rect, *anchor), galley.size());
                layout.push((bounds, galley, OVERLAY_TEXT_COLOR));
            }
        }
        for (anchor, label) in [
            (Align2::LEFT_CENTER, labels.left),
            (Align2::RIGHT_CENTER, labels.right),
            (Align2::CENTER_TOP, labels.top),
            (Align2::CENTER_BOTTOM, labels.bottom),
        ] {
            let galley = painter.layout_no_wrap(
                label.to_owned(),
                FontId::proportional(ORIENTATION_FONT_SIZE),
                ORIENT_LABEL_COLOR,
            );
            let bounds = anchor.anchor_size(Self::anchor_pos(rect, anchor), galley.size());
            layout.push((bounds, galley, ORIENT_LABEL_COLOR));
        }
        let fits = layout.iter().enumerate().all(|(index, (bounds, _, _))| {
            let backing = bounds.expand(TEXT_PADDING);
            rect.contains_rect(backing)
                && layout[index + 1..]
                    .iter()
                    .all(|(other, _, _)| !backing.intersects(other.expand(TEXT_PADDING)))
        });
        if !fits {
            return Some(format!(
                "{}\n\nOrientation: left {}, right {}, top {}, bottom {}",
                blocks
                    .iter()
                    .map(|(_, text)| text.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
                labels.left,
                labels.right,
                labels.top,
                labels.bottom
            ));
        }
        for (bounds, galley, color) in layout {
            Self::paint_bounds(painter, bounds, galley, color);
        }
        None
    }

    /// Draw `text` anchored at the given corner of `rect`.
    ///
    /// `anchor` controls both which corner is used as the anchor point and
    /// how the text is aligned relative to that point. Text wraps within its
    /// half of the viewport, with a central orientation-label lane reserved.
    /// An opaque dark backing preserves contrast across the grayscale range.
    pub fn draw_text_anchored(
        painter: &Painter,
        rect: Rect,
        anchor: Align2,
        text: &str,
        color: Color32,
    ) {
        let rect = rect.intersect(painter.clip_rect());
        let galley = Self::layout_corner(painter, rect, text, color);
        Self::paint_galley(painter, rect, anchor, galley, color);
    }

    fn layout_corner(
        painter: &Painter,
        rect: Rect,
        text: &str,
        color: Color32,
    ) -> std::sync::Arc<egui::Galley> {
        // Separate columns leave a measured central lane for orientation labels.
        // Wrapping preserves every character, including long identifiers.
        let orientation_width = ["L", "R", "P", "A", "S", "I"]
            .into_iter()
            .map(|label| {
                painter
                    .layout_no_wrap(
                        label.to_owned(),
                        FontId::proportional(ORIENTATION_FONT_SIZE),
                        ORIENT_LABEL_COLOR,
                    )
                    .size()
                    .x
            })
            .fold(0.0_f32, f32::max);
        let center_lane = orientation_width + 2.0 * (TEXT_PADDING + MARGIN);
        let column_width = (rect.width() - 2.0 * MARGIN - center_lane) * 0.5;
        let wrap_width = (column_width - 2.0 * TEXT_PADDING).max(1.0);
        painter.layout(
            text.to_owned(),
            FontId::proportional(OVERLAY_FONT_SIZE),
            color,
            wrap_width,
        )
    }

    fn paint_galley(
        painter: &Painter,
        rect: Rect,
        anchor: Align2,
        galley: std::sync::Arc<egui::Galley>,
        color: Color32,
    ) {
        let bounds = anchor.anchor_size(Self::anchor_pos(rect, anchor), galley.size());
        Self::paint_bounds(painter, bounds, galley, color);
    }

    fn paint_bounds(
        painter: &Painter,
        bounds: Rect,
        galley: std::sync::Arc<egui::Galley>,
        color: Color32,
    ) {
        painter.rect_filled(bounds.expand(TEXT_PADDING), 2.0, Color32::BLACK);
        painter.galley(bounds.min, galley, color);
    }

    /// Compute the screen position for the given anchor within `rect`,
    /// inset by [`MARGIN`] pixels from each edge.
    fn anchor_pos(rect: Rect, anchor: Align2) -> Pos2 {
        let x = match anchor.x() {
            egui::Align::Min => rect.min.x + MARGIN,
            egui::Align::Center => rect.center().x,
            egui::Align::Max => rect.max.x - MARGIN,
        };
        let y = match anchor.y() {
            egui::Align::Min => rect.min.y + MARGIN,
            egui::Align::Center => rect.center().y,
            egui::Align::Max => rect.max.y - MARGIN,
        };
        Pos2::new(x, y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrientationLabels {
    left: &'static str,
    right: &'static str,
    top: &'static str,
    bottom: &'static str,
}

fn orientation_labels(axis: usize, direction: &[f64; 9]) -> OrientationLabels {
    let col = |j: usize| -> [f64; 3] { [direction[j], direction[3 + j], direction[6 + j]] };

    let depth_axis = col(0);
    let row_axis = col(1);
    let col_axis = col(2);

    let (horiz, vert) = match axis {
        0 => (col_axis, row_axis),
        1 => (col_axis, depth_axis),
        _ => (row_axis, depth_axis),
    };

    OrientationLabels {
        left: lps_label(horiz, false),
        right: lps_label(horiz, true),
        top: lps_label(vert, false),
        bottom: lps_label(vert, true),
    }
}

fn lps_label(v: [f64; 3], positive: bool) -> &'static str {
    let v = if positive { v } else { [-v[0], -v[1], -v[2]] };
    let abs = [v[0].abs(), v[1].abs(), v[2].abs()];
    let max_idx = if abs[0] >= abs[1] && abs[0] >= abs[2] {
        0
    } else if abs[1] >= abs[2] {
        1
    } else {
        2
    };
    match (max_idx, v[max_idx] >= 0.0) {
        (0, true) => "L",
        (0, false) => "R",
        (1, true) => "P",
        (1, false) => "A",
        (2, true) => "S",
        _ => "I",
    }
}

// ── Pure display-string helpers (testable) ────────────────────────────────────

/// Format the pointer-position intensity label.
///
/// Returns `"Pointer SUV: {:.2}"` when `pointer_suv` is `Some`,
/// `"Pointer value: {:.0}"` when `pointer_intensity != 0.0`, or `""` otherwise.
pub(crate) fn format_pointer_str(pointer_intensity: f32, pointer_suv: Option<f32>) -> String {
    match pointer_suv {
        Some(s) => format!("Pointer SUV: {:.2}", s),
        None if pointer_intensity != 0.0 => format!("Pointer value: {:.0}", pointer_intensity),
        _ => String::new(),
    }
}

/// Format the cursor-position intensity label.
///
/// Returns `"Cursor SUV: {:.2}"` when `cursor_suv` is `Some`,
/// `"Cursor value: {:.0}"` when `cursor_value` is `Some`, or `""` otherwise.
pub(crate) fn format_cursor_str(cursor_value: Option<f32>, cursor_suv: Option<f32>) -> String {
    match (cursor_suv, cursor_value) {
        (Some(s), _) => format!("Cursor SUV: {:.2}", s),
        (None, Some(v)) => format!("Cursor value: {:.0}", v),
        _ => String::new(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
