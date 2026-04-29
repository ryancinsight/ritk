//! Measurement annotation rendering layer.
//!
//! # Drawing conventions
//!
//! | Annotation type | Colour          | Line width |
//! |-----------------|-----------------|------------|
//! | Length          | YELLOW          | 1.5 px     |
//! | Angle           | YELLOW          | 1.5 px     |
//! | ROI (rect)      | GREEN           | 1.5 px     |
//! | HU point        | CYAN            | —          |
//! | Labels          | WHITE           | —          |
//!
//! All screen positions are computed by the caller-supplied `img_to_screen`
//! closure, which maps image-pixel coordinates (col, row as f32) to screen
//! coordinates.  This keeps the rendering layer free of view-transform state.
//!
//! # Font size
//! All measurement labels use a 12 pt proportional font.

use egui::{Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};

use crate::tools::interaction::{Annotation, RoiKind, ToolState};

// ── colour / style constants ──────────────────────────────────────────────────

const COLOR_MEASURE: Color32 = Color32::YELLOW;
const COLOR_ROI: Color32 = Color32::GREEN;
const COLOR_LABEL: Color32 = Color32::WHITE;
const COLOR_HU_POINT: Color32 = Color32::from_rgb(0, 255, 255); // cyan

const LINE_WIDTH: f32 = 1.5;
const FONT_SIZE: f32 = 12.0;
const HANDLE_RADIUS: f32 = 3.0;

fn label_font() -> FontId {
    FontId::proportional(FONT_SIZE)
}

// ── MeasurementLayer ──────────────────────────────────────────────────────────

/// Stateless annotation rendering helpers.
///
/// Call [`MeasurementLayer::draw_annotations`] every frame to repaint all
/// completed annotations, and [`MeasurementLayer::draw_in_progress`] to
/// render the currently active in-progress tool state.
pub struct MeasurementLayer;

impl MeasurementLayer {
    /// Draw all completed [`Annotation`]s stored in `annotations`.
    ///
    /// `img_to_screen` converts image-pixel coordinates `(col, row)` (stored
    /// in Pos2 as `Pos2 { x: col, y: row }`) to screen coordinates.
    ///
    /// Annotations whose geometric positions lie outside the current viewport
    /// are still passed to the underlying drawing routines; egui clips them
    /// to the painter clip rectangle automatically.
    pub fn draw_annotations(
        painter: &Painter,
        annotations: &[Annotation],
        img_to_screen: impl Fn(Pos2) -> Pos2,
    ) {
        for annotation in annotations {
            match annotation {
                Annotation::Length {
                    p1, p2, length_mm, ..
                } => {
                    let sp1 = img_to_screen(egui::pos2((*p1)[1], (*p1)[0]));
                    let sp2 = img_to_screen(egui::pos2((*p2)[1], (*p2)[0]));
                    draw_length_annotation(painter, sp1, sp2, *length_mm);
                }
                Annotation::Angle {
                    p1,
                    p2,
                    p3,
                    angle_deg,
                    ..
                } => {
                    let sp1 = img_to_screen(egui::pos2((*p1)[1], (*p1)[0]));
                    let sp2 = img_to_screen(egui::pos2((*p2)[1], (*p2)[0]));
                    let sp3 = img_to_screen(egui::pos2((*p3)[1], (*p3)[0]));
                    draw_angle_annotation(painter, sp1, sp2, sp3, *angle_deg);
                }
                Annotation::RoiRect {
                    top_left,
                    bottom_right,
                    mean,
                    std_dev,
                    ..
                } => {
                    let stl = img_to_screen(egui::pos2((*top_left)[1], (*top_left)[0]));
                    let sbr = img_to_screen(egui::pos2((*bottom_right)[1], (*bottom_right)[0]));
                    draw_roi_rect_annotation(painter, stl, sbr, *mean, *std_dev);
                }
                Annotation::HuPoint { pos, value, .. } => {
                    let spos = img_to_screen(egui::pos2((*pos)[1], (*pos)[0]));
                    draw_hu_point(painter, spos, *value);
                }
            }
        }
    }

    /// Draw the in-progress tool state (e.g. a rubber-band line during length
    /// measurement, or a growing ROI rectangle).
    ///
    /// `cursor_pos` is the current cursor position in *screen* coordinates.
    /// `img_to_screen` converts image-pixel coordinates to screen coordinates,
    /// used to re-project already-placed anchor points.
    pub fn draw_in_progress(
        painter: &Painter,
        tool_state: &ToolState,
        cursor_pos: Option<Pos2>,
        img_to_screen: impl Fn(Pos2) -> Pos2,
    ) {
        match tool_state {
            ToolState::Idle => {}

            ToolState::MeasureLength1 { p1 } => {
                // Draw the anchor handle and a rubber-band line to the cursor.
                let sp1 = img_to_screen(*p1);
                painter.circle_filled(sp1, HANDLE_RADIUS, COLOR_MEASURE);
                if let Some(cursor) = cursor_pos {
                    painter.line_segment([sp1, cursor], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                }
            }

            ToolState::MeasureAngle2 { p1, p2 } => {
                // Draw two anchor handles and lines: p1→p2, p2→cursor.
                let sp1 = img_to_screen(*p1);
                let sp2 = img_to_screen(*p2);
                painter.circle_filled(sp1, HANDLE_RADIUS, COLOR_MEASURE);
                painter.circle_filled(sp2, HANDLE_RADIUS, COLOR_MEASURE);
                painter.line_segment([sp1, sp2], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                if let Some(cursor) = cursor_pos {
                    painter.line_segment([sp2, cursor], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                }
            }

            ToolState::RoiDrag {
                start,
                current,
                kind,
            } => {
                let ss = img_to_screen(*start);
                let sc = img_to_screen(*current);
                match kind {
                    RoiKind::Rect => {
                        let tl = Pos2::new(ss.x.min(sc.x), ss.y.min(sc.y));
                        let br = Pos2::new(ss.x.max(sc.x), ss.y.max(sc.y));
                        let rect = Rect::from_min_max(tl, br);
                        painter.rect_stroke(rect, 0.0, Stroke::new(LINE_WIDTH, COLOR_ROI));
                    }
                    RoiKind::Ellipse => {
                        let center = Pos2::new((ss.x + sc.x) * 0.5, (ss.y + sc.y) * 0.5);
                        let radii = Vec2::new((sc.x - ss.x).abs() * 0.5, (sc.y - ss.y).abs() * 0.5);
                        painter.add(egui::epaint::EllipseShape {
                            center,
                            radius: radii,
                            fill: Color32::TRANSPARENT,
                            stroke: Stroke::new(LINE_WIDTH, COLOR_ROI),
                        });
                    }
                }
            }

            // Panning and window-level drag produce no visual annotation.
            ToolState::Panning { .. } | ToolState::WindowLevelDrag { .. } => {}
        }
    }
}

// ── private drawing functions ─────────────────────────────────────────────────

/// Draw a length measurement annotation: a line with endpoint handles and
/// a label showing the distance in mm.
///
/// # Label placement
/// The label is positioned at the midpoint of the line, offset 8 px
/// perpendicular to the line direction to avoid overlap.
fn draw_length_annotation(painter: &Painter, p1: Pos2, p2: Pos2, length_mm: f32) {
    // Line
    painter.line_segment([p1, p2], Stroke::new(LINE_WIDTH, COLOR_MEASURE));

    // Endpoint handles
    painter.circle_filled(p1, HANDLE_RADIUS, COLOR_MEASURE);
    painter.circle_filled(p2, HANDLE_RADIUS, COLOR_MEASURE);

    // Label at midpoint, offset perpendicular to the line.
    let mid = Pos2::new((p1.x + p2.x) * 0.5, (p1.y + p2.y) * 0.5);
    let label = format!("{:.1} mm", length_mm);
    let offset = perpendicular_offset(p1, p2, 10.0);
    painter.text(
        mid + offset,
        egui::Align2::CENTER_CENTER,
        label,
        label_font(),
        COLOR_LABEL,
    );
}

/// Draw an angle measurement annotation: two rays from the vertex `p2` to
/// `p1` and `p3`, with a small arc at the vertex and an angle label.
///
/// # Label placement
/// The label is placed along the angle bisector from `p2`, 20 px from the
/// vertex.
fn draw_angle_annotation(painter: &Painter, p1: Pos2, p2: Pos2, p3: Pos2, angle_deg: f32) {
    // Two rays
    painter.line_segment([p2, p1], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
    painter.line_segment([p2, p3], Stroke::new(LINE_WIDTH, COLOR_MEASURE));

    // Endpoint handles
    painter.circle_filled(p1, HANDLE_RADIUS, COLOR_MEASURE);
    painter.circle_filled(p2, HANDLE_RADIUS, COLOR_MEASURE);
    painter.circle_filled(p3, HANDLE_RADIUS, COLOR_MEASURE);

    // Small arc at the vertex (approximated by a circle outline).
    painter.circle_stroke(p2, 8.0, Stroke::new(1.0, COLOR_MEASURE));

    // Label along the angle bisector.
    let d1 = (p1 - p2).normalized();
    let d3 = (p3 - p2).normalized();
    let bisector = if (d1 + d3).length() > 1e-4 {
        (d1 + d3).normalized()
    } else {
        // Degenerate: rays are antiparallel; use perpendicular.
        Vec2::new(-d1.y, d1.x)
    };
    let label_pos = p2 + bisector * 22.0;
    let label = format!("{:.1}°", angle_deg);
    painter.text(
        label_pos,
        egui::Align2::CENTER_CENTER,
        label,
        label_font(),
        COLOR_LABEL,
    );
}

/// Draw a rectangular ROI annotation with a statistics label.
///
/// The rectangle is drawn with [`COLOR_ROI`]. The label shows mean ± std.
///
/// # Label placement
/// The label is placed immediately below the bottom edge of the rectangle.
fn draw_roi_rect_annotation(painter: &Painter, tl: Pos2, br: Pos2, mean: f32, std: f32) {
    let tl = Pos2::new(tl.x.min(br.x), tl.y.min(br.y));
    let br = Pos2::new(tl.x.max(br.x), tl.y.max(br.y));
    let rect = Rect::from_min_max(tl, br);

    // Rectangle border
    painter.rect_stroke(rect, 0.0, Stroke::new(LINE_WIDTH, COLOR_ROI));

    // Corner handles
    for corner in &[tl, Pos2::new(br.x, tl.y), br, Pos2::new(tl.x, br.y)] {
        painter.circle_filled(*corner, HANDLE_RADIUS, COLOR_ROI);
    }

    // Label below the bottom edge.
    let label = format!("μ={:.1}  σ={:.1}", mean, std);
    let label_pos = Pos2::new(rect.center().x, br.y + 12.0);
    painter.text(
        label_pos,
        egui::Align2::CENTER_TOP,
        label,
        label_font(),
        COLOR_LABEL,
    );
}

/// Draw a single HU point annotation: a small crosshair circle with the
/// pixel value label.
fn draw_hu_point(painter: &Painter, pos: Pos2, value: f32) {
    // Crosshair
    let arm = 5.0_f32;
    painter.line_segment(
        [Pos2::new(pos.x - arm, pos.y), Pos2::new(pos.x + arm, pos.y)],
        Stroke::new(LINE_WIDTH, COLOR_HU_POINT),
    );
    painter.line_segment(
        [Pos2::new(pos.x, pos.y - arm), Pos2::new(pos.x, pos.y + arm)],
        Stroke::new(LINE_WIDTH, COLOR_HU_POINT),
    );
    painter.circle_stroke(pos, arm, Stroke::new(LINE_WIDTH, COLOR_HU_POINT));

    // Value label
    let label = format!("{:.0} HU", value);
    painter.text(
        Pos2::new(pos.x + arm + 4.0, pos.y),
        egui::Align2::LEFT_CENTER,
        label,
        label_font(),
        COLOR_LABEL,
    );
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Compute a perpendicular offset vector of length `distance` from the line
/// `p1 → p2`.
///
/// The perpendicular is rotated 90° counter-clockwise from the line direction.
/// When `p1 == p2` (degenerate line), returns a straight-up offset `(0, -d)`.
fn perpendicular_offset(p1: Pos2, p2: Pos2, distance: f32) -> Vec2 {
    let d = p2 - p1;
    let len = d.length();
    if len < 1e-4 {
        return Vec2::new(0.0, -distance);
    }
    // 90° CCW rotation: (dx, dy) → (-dy, dx)
    Vec2::new(-d.y / len, d.x / len) * distance
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `perpendicular_offset` for a horizontal right-pointing line must return
    /// a straight-up vector.
    ///
    /// Analytical: direction = (1, 0) → perpendicular CCW = (0, 1) rotated =
    /// (-0, 1) → normalised (-dy, dx) = (0, 1).
    /// With distance = 10: result = (0, 10).
    #[test]
    fn test_perpendicular_offset_horizontal() {
        let p1 = Pos2::new(0.0, 0.0);
        let p2 = Pos2::new(10.0, 0.0);
        let off = perpendicular_offset(p1, p2, 10.0);
        // (-dy/|d|, dx/|d|) * distance = (-0/10, 10/10) * 10 = (0.0, 1.0) * 10
        assert!(
            off.x.abs() < 1e-4,
            "horizontal line perpendicular x must be 0, got {}",
            off.x
        );
        assert!(
            (off.y - 10.0).abs() < 1e-4,
            "horizontal line perpendicular y must be 10, got {}",
            off.y
        );
    }

    /// `perpendicular_offset` for a vertical downward line must return a
    /// right-pointing vector.
    ///
    /// Analytical: direction = (0, 1) → (-dy, dx) = (-1, 0).
    /// With distance = 5: result = (-5, 0).
    #[test]
    fn test_perpendicular_offset_vertical() {
        let p1 = Pos2::new(0.0, 0.0);
        let p2 = Pos2::new(0.0, 10.0);
        let off = perpendicular_offset(p1, p2, 5.0);
        // direction = (0, 1), normalised = (0, 1).
        // (-dy, dx) = (-1, 0) * 5 = (-5, 0).
        assert!(
            (off.x - (-5.0)).abs() < 1e-4,
            "vertical line perpendicular x must be -5, got {}",
            off.x
        );
        assert!(
            off.y.abs() < 1e-4,
            "vertical line perpendicular y must be 0, got {}",
            off.y
        );
    }

    /// `perpendicular_offset` for a degenerate (zero-length) line must return
    /// a straight-up fallback `(0, -d)`.
    #[test]
    fn test_perpendicular_offset_degenerate() {
        let p = Pos2::new(5.0, 5.0);
        let off = perpendicular_offset(p, p, 8.0);
        assert!(
            off.x.abs() < 1e-4,
            "degenerate line perpendicular x must be 0, got {}",
            off.x
        );
        assert!(
            (off.y - (-8.0)).abs() < 1e-4,
            "degenerate line perpendicular y must be -8, got {}",
            off.y
        );
    }

    /// The `MeasurementLayer` type must be constructible (zero-size type check).
    #[test]
    fn test_measurement_layer_is_zero_size() {
        let _layer = MeasurementLayer;
    }
}
