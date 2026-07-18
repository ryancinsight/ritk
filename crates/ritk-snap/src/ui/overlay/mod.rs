//! DICOM-style 4-corner information overlay and patient-orientation labels.
//!
//! # Overlay layout
//!
//! ```text
//! â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
//! â”‚ Patient Name          Series Desc       â”‚
//! â”‚ Patient ID            Modality / Date   â”‚
//! â”‚                                         â”‚
//! â”‚                                         â”‚
//! â”‚ Slice N/M             W: WWWW C: CCCC   â”‚
//! â”‚ Spacing               Zoom: ZZZ%        â”‚
//! â”‚ Dimensions            HU: VVVV          â”‚
//! â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
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

use super::anatomical_label_for_axis;
use crate::render::slice_render::WindowLevel;
use crate::LoadedVolume;

// â”€â”€ constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Font size for overlay text (points).
const OVERLAY_FONT_SIZE: f32 = 12.0;

/// Colour for all overlay text.
const OVERLAY_TEXT_COLOR: Color32 = Color32::from_rgb(255, 255, 160); // warm yellow

/// Colour for orientation labels (brighter white for visibility).
const ORIENT_LABEL_COLOR: Color32 = Color32::WHITE;

/// Margin from the viewport edge (pixels).
const MARGIN: f32 = 6.0;

// â”€â”€ OverlayRenderer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    pub cursor_suv: Option<f32> }

/// Renders DICOM-style information overlays on a viewport rectangle.
///
/// All methods are stateless; call them with the current render state on
/// every frame.
pub struct OverlayRenderer;

impl OverlayRenderer {
    // â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Draw the standard 4-corner DICOM text overlay.
    ///
    /// Corner assignments:
    ///
    /// | Corner      | Content                                          |
    /// |-------------|--------------------------------------------------|
    /// | Top-left    | Patient Name, Patient ID                         |
    /// | Top-right   | Series description, Modality, Study date         |
    /// | Bottom-left | Slice N/M, Voxel spacing, Image dimensions       |
    /// | Bottom-right| Window width/centre, Zoom %, cursor HU value    |
    ///
    /// # Parameters
    /// - `painter`      â€” egui painter for the viewport.
    /// - `rect`         â€” viewport rectangle in screen coordinates.
    /// - `volume`       â€” loaded volume supplying metadata.
    /// - `axis`         â€” current MPR axis (0=axial, 1=coronal, 2=sagittal).
    /// - `slice_index`  â€” currently displayed slice index.
    /// - `wl`           â€” current window/level settings.
    /// - `zoom`         â€” current zoom factor (1.0 = fit-to-viewport).
    /// - `cursor_value` â€” pixel value (HU) at the cursor position, or `None`.
    /// - `pointer_suv`  â€” SUVbw value under the pointer (PT only), or `None`.
    /// - `cursor_suv`   â€” SUVbw value at the linked-cursor voxel (PT only), or `None`.
    pub fn draw(painter: &Painter, rect: Rect, volume: &LoadedVolume, ctx: OverlayContext) {
        let OverlayContext {
            axis,
            slice_index,
            wl,
            zoom,
            cursor_value,
            pointer_intensity,
            pointer_suv,
            cursor_suv } = ctx;
        let [depth, rows, cols] = volume.shape;

        // â”€â”€ Top-left: patient information â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        Self::draw_text_anchored(
            painter,
            rect,
            Align2::LEFT_TOP,
            &tl_text,
            OVERLAY_TEXT_COLOR,
        );

        // â”€â”€ Top-right: series / modality / date â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        if !tr_lines.is_empty() {
            Self::draw_text_anchored(
                painter,
                rect,
                Align2::RIGHT_TOP,
                &tr_lines.join("\n"),
                OVERLAY_TEXT_COLOR,
            );
        }

        // â”€â”€ Bottom-left: slice, spacing, dimensions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        let (total_slices, dim_w, dim_h) = match axis {
            0 => (depth, cols, rows),
            1 => (rows, cols, depth),
            _ => (cols, rows, depth) };
        let axis_name = anatomical_label_for_axis(Some(volume), axis);

        let [dz, dy, dx] = volume.spacing;
        let spacing_str = format!("{:.2} Ã— {:.2} Ã— {:.2} mm", dx, dy, dz);
        let dims_str = format!("{}Ã—{}Ã—{}", cols, rows, depth);
        let slice_str = format!(
            "{}: {}/{}   {}Ã—{}",
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
        Self::draw_text_anchored(
            painter,
            rect,
            Align2::LEFT_BOTTOM,
            &bl_text,
            OVERLAY_TEXT_COLOR,
        );

        // â”€â”€ Bottom-right: W/L, zoom, cursor, pointer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        Self::draw_text_anchored(
            painter,
            rect,
            Align2::RIGHT_BOTTOM,
            &br_lines.join("\n"),
            OVERLAY_TEXT_COLOR,
        );
    }

    /// Draw patient orientation labels on the four edges of the viewport.
    ///
    /// The label placed on each edge is determined by projecting the display
    /// axes onto the LPS (Left-Posterior-Superior) anatomical frame using the
    /// direction cosine matrix.
    ///
    /// # Direction matrix convention (RITK)
    ///
    /// `direction` is a row-major 3Ã—3 matrix stored as 9 f64 values.
    /// Column 0 = depth axis (NÌ‚), column 1 = row axis, column 2 = column axis.
    ///
    /// For display purposes:
    /// - `axis = 0` (axial): horizontal = column axis (col 2),
    ///   vertical   = row axis (col 1).
    /// - `axis = 1` (coronal): horizontal = column axis (col 2),
    ///   vertical   = depth axis (col 0).
    /// - `axis = 2` (sagittal): horizontal = row axis (col 1),
    ///   vertical   = depth axis (col 0).
    ///
    /// # Parameters
    /// - `painter`   â€” egui painter for the viewport.
    /// - `rect`      â€” viewport rectangle in screen coordinates.
    /// - `axis`      â€” current MPR axis.
    /// - `direction` â€” 3Ã—3 direction cosine matrix (row-major, 9 elements).
    pub fn draw_orientation_labels(
        painter: &Painter,
        rect: Rect,
        axis: usize,
        direction: &[f64; 9],
    ) {
        let OrientationLabels {
            left: label_left,
            right: label_right,
            top: label_top,
            bottom: label_bottom } = orientation_labels(axis, direction);

        let font = FontId::proportional(14.0);
        let cx = rect.center().x;
        let cy = rect.center().y;

        // Left edge: vertically centred, left-aligned.
        painter.text(
            Pos2::new(rect.min.x + MARGIN, cy),
            Align2::LEFT_CENTER,
            label_left,
            font.clone(),
            ORIENT_LABEL_COLOR,
        );
        // Right edge: vertically centred, right-aligned.
        painter.text(
            Pos2::new(rect.max.x - MARGIN, cy),
            Align2::RIGHT_CENTER,
            label_right,
            font.clone(),
            ORIENT_LABEL_COLOR,
        );
        // Top edge: horizontally centred, top-aligned.
        painter.text(
            Pos2::new(cx, rect.min.y + MARGIN),
            Align2::CENTER_TOP,
            label_top,
            font.clone(),
            ORIENT_LABEL_COLOR,
        );
        // Bottom edge: horizontally centred, bottom-aligned.
        painter.text(
            Pos2::new(cx, rect.max.y - MARGIN),
            Align2::CENTER_BOTTOM,
            label_bottom,
            font.clone(),
            ORIENT_LABEL_COLOR,
        );
    }

    // â”€â”€ Private helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    /// Draw `text` anchored at the given corner of `rect`.
    ///
    /// `anchor` controls both which corner is used as the anchor point and
    /// how the text is aligned relative to that point.
    pub fn draw_text_anchored(
        painter: &Painter,
        rect: Rect,
        anchor: Align2,
        text: &str,
        color: Color32,
    ) {
        // Compute the anchor position including the margin offset.
        let pos = Self::anchor_pos(rect, anchor);
        painter.text(
            pos,
            anchor,
            text,
            FontId::proportional(OVERLAY_FONT_SIZE),
            color,
        );
    }

    /// Compute the screen position for the given anchor within `rect`,
    /// inset by [`MARGIN`] pixels from each edge.
    fn anchor_pos(rect: Rect, anchor: Align2) -> Pos2 {
        let x = match anchor.x() {
            egui::Align::Min => rect.min.x + MARGIN,
            egui::Align::Center => rect.center().x,
            egui::Align::Max => rect.max.x - MARGIN };
        let y = match anchor.y() {
            egui::Align::Min => rect.min.y + MARGIN,
            egui::Align::Center => rect.center().y,
            egui::Align::Max => rect.max.y - MARGIN };
        Pos2::new(x, y)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OrientationLabels {
    left: &'static str,
    right: &'static str,
    top: &'static str,
    bottom: &'static str }

fn orientation_labels(axis: usize, direction: &[f64; 9]) -> OrientationLabels {
    let col = |j: usize| -> [f64; 3] { [direction[j], direction[3 + j], direction[6 + j]] };

    let depth_axis = col(0);
    let row_axis = col(1);
    let col_axis = col(2);

    let (horiz, vert) = match axis {
        0 => (col_axis, row_axis),
        1 => (col_axis, depth_axis),
        _ => (row_axis, depth_axis) };

    OrientationLabels {
        left: lps_label(horiz, false),
        right: lps_label(horiz, true),
        top: lps_label(vert, false),
        bottom: lps_label(vert, true) }
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
        _ => "I" }
}

// â”€â”€ Pure display-string helpers (testable) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Format the pointer-position intensity label.
///
/// Returns `"Pointer SUV: {:.2}"` when `pointer_suv` is `Some`,
/// `"Pointer HU: {:.0}"` when `pointer_intensity != 0.0`, or `""` otherwise.
pub(crate) fn format_pointer_str(pointer_intensity: f32, pointer_suv: Option<f32>) -> String {
    match pointer_suv {
        Some(s) => format!("Pointer SUV: {:.2}", s),
        None if pointer_intensity != 0.0 => format!("Pointer HU: {:.0}", pointer_intensity),
        _ => String::new() }
}

/// Format the cursor-position intensity label.
///
/// Returns `"Cursor SUV: {:.2}"` when `cursor_suv` is `Some`,
/// `"Cursor HU: {:.0}"` when `cursor_value` is `Some`, or `""` otherwise.
pub(crate) fn format_cursor_str(cursor_value: Option<f32>, cursor_suv: Option<f32>) -> String {
    match (cursor_suv, cursor_value) {
        (Some(s), _) => format!("Cursor SUV: {:.2}", s),
        (None, Some(v)) => format!("Cursor HU: {:.0}", v),
        _ => String::new() }
}

// â”€â”€ Tests â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

#[cfg(test)]
mod tests;
