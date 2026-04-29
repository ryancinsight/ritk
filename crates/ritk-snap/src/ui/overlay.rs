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
//! │ Dimensions            HU: VVVV          │
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

// ── OverlayRenderer ───────────────────────────────────────────────────────────

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
    /// | Bottom-right| Window width/centre, Zoom %, cursor HU value    |
    ///
    /// # Parameters
    /// - `painter`      — egui painter for the viewport.
    /// - `rect`         — viewport rectangle in screen coordinates.
    /// - `volume`       — loaded volume supplying metadata.
    /// - `axis`         — current MPR axis (0=axial, 1=coronal, 2=sagittal).
    /// - `slice_index`  — currently displayed slice index.
    /// - `wl`           — current window/level settings.
    /// - `zoom`         — current zoom factor (1.0 = fit-to-viewport).
    /// - `cursor_value` — pixel value (HU) at the cursor position, or `None`.
    pub fn draw(
        painter: &Painter,
        rect: Rect,
        volume: &LoadedVolume,
        axis: usize,
        slice_index: usize,
        wl: WindowLevel,
        zoom: f32,
        cursor_value: Option<f32>,
    ) {
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
        Self::draw_text_anchored(
            painter,
            rect,
            Align2::LEFT_TOP,
            &tl_text,
            OVERLAY_TEXT_COLOR,
        );

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
        if !tr_lines.is_empty() {
            Self::draw_text_anchored(
                painter,
                rect,
                Align2::RIGHT_TOP,
                &tr_lines.join("\n"),
                OVERLAY_TEXT_COLOR,
            );
        }

        // ── Bottom-left: slice, spacing, dimensions ────────────────────────
        let (total_slices, dim_w, dim_h) = match axis {
            0 => (depth, cols, rows),
            1 => (rows, cols, depth),
            _ => (cols, rows, depth),
        };
        let axis_name = match axis {
            0 => "Axial",
            1 => "Coronal",
            _ => "Sagittal",
        };

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
        Self::draw_text_anchored(
            painter,
            rect,
            Align2::LEFT_BOTTOM,
            &bl_text,
            OVERLAY_TEXT_COLOR,
        );

        // ── Bottom-right: W/L, zoom, cursor HU ────────────────────────────
        let wl_str = format!("W:{:.0} C:{:.0}", wl.width, wl.center);
        let zoom_str = format!("Zoom: {:.0}%", zoom * 100.0);
        let hu_str = match cursor_value {
            Some(v) => format!("HU: {:.0}", v),
            None => String::new(),
        };
        let br_lines: Vec<&str> = [&wl_str as &str, &zoom_str as &str, &hu_str as &str]
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
    /// `direction` is a row-major 3×3 matrix stored as 9 f64 values.
    /// Column 0 = depth axis (N̂), column 1 = row axis, column 2 = column axis.
    ///
    /// For display purposes:
    /// - `axis = 0` (axial): horizontal = column axis (col 2),
    ///                        vertical   = row axis (col 1).
    /// - `axis = 1` (coronal): horizontal = column axis (col 2),
    ///                          vertical   = depth axis (col 0).
    /// - `axis = 2` (sagittal): horizontal = row axis (col 1),
    ///                           vertical   = depth axis (col 0).
    ///
    /// # Parameters
    /// - `painter`   — egui painter for the viewport.
    /// - `rect`      — viewport rectangle in screen coordinates.
    /// - `axis`      — current MPR axis.
    /// - `direction` — 3×3 direction cosine matrix (row-major, 9 elements).
    pub fn draw_orientation_labels(
        painter: &Painter,
        rect: Rect,
        axis: usize,
        direction: &[f64; 9],
    ) {
        // Extract the three column vectors from the row-major direction matrix.
        // Column j is [dir[0*3+j], dir[1*3+j], dir[2*3+j]].
        let col = |j: usize| -> [f64; 3] { [direction[j], direction[3 + j], direction[6 + j]] };

        // Column indices: 0 = depth axis (N̂), 1 = row axis, 2 = column axis.
        let depth_axis = col(0);
        let row_axis = col(1);
        let col_axis = col(2);

        // For each display axis, determine which direction vectors map to
        // the horizontal (left→right) and vertical (top→bottom) display axes.
        let (horiz, vert) = match axis {
            0 => (col_axis, row_axis),   // axial
            1 => (col_axis, depth_axis), // coronal
            _ => (row_axis, depth_axis), // sagittal
        };

        // Determine the anatomical label for a direction vector.
        // The dominant LPS component determines the label.
        // LPS: +X = Left, +Y = Posterior, +Z = Superior.
        let lps_label = |v: [f64; 3], positive: bool| -> &'static str {
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
        };

        // left edge: negative horizontal direction
        let label_left = lps_label(horiz, false);
        // right edge: positive horizontal direction
        let label_right = lps_label(horiz, true);
        // top edge: negative vertical direction
        let label_top = lps_label(vert, false);
        // bottom edge: positive vertical direction
        let label_bottom = lps_label(vert, true);

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

    // ── Private helpers ───────────────────────────────────────────────────────

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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // We cannot run egui painter methods in unit tests without a GPU context,
    // so we test the pure-computation helpers only.

    /// `anchor_pos` for LEFT_TOP must return (min.x + MARGIN, min.y + MARGIN).
    #[test]
    fn test_anchor_pos_left_top() {
        let rect = Rect::from_min_max(Pos2::new(10.0, 20.0), Pos2::new(110.0, 120.0));
        let pos = OverlayRenderer::anchor_pos(rect, Align2::LEFT_TOP);
        assert!(
            (pos.x - (10.0 + MARGIN)).abs() < 1e-4,
            "LEFT_TOP x must be rect.min.x + MARGIN"
        );
        assert!(
            (pos.y - (20.0 + MARGIN)).abs() < 1e-4,
            "LEFT_TOP y must be rect.min.y + MARGIN"
        );
    }

    /// `anchor_pos` for RIGHT_BOTTOM must return (max.x − MARGIN, max.y − MARGIN).
    #[test]
    fn test_anchor_pos_right_bottom() {
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(200.0, 100.0));
        let pos = OverlayRenderer::anchor_pos(rect, Align2::RIGHT_BOTTOM);
        assert!(
            (pos.x - (200.0 - MARGIN)).abs() < 1e-4,
            "RIGHT_BOTTOM x must be rect.max.x - MARGIN"
        );
        assert!(
            (pos.y - (100.0 - MARGIN)).abs() < 1e-4,
            "RIGHT_BOTTOM y must be rect.max.y - MARGIN"
        );
    }

    /// `anchor_pos` for CENTER_CENTER must return the rect centre exactly.
    #[test]
    fn test_anchor_pos_center_center() {
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(100.0, 80.0));
        let pos = OverlayRenderer::anchor_pos(rect, Align2::CENTER_CENTER);
        assert!(
            (pos.x - 50.0).abs() < 1e-4,
            "CENTER_CENTER x must be rect centre x = 50"
        );
        assert!(
            (pos.y - 40.0).abs() < 1e-4,
            "CENTER_CENTER y must be rect centre y = 40"
        );
    }
}
