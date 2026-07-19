//! Measurement annotation rendering layer.
//!
//! # Drawing conventions
//!
//! | Annotation type | Colour         | Line width |
//! |-----------------|----------------|------------|
//! | Length           | YELLOW         | 1.5 px     |
//! | Angle            | YELLOW         | 1.5 px     |
//! | ROI (rect)       | GREEN          | 1.5 px     |
//! | HU point         | CYAN           | â€”          |
//! | Labels           | WHITE          | â€”          |
//!
//! All screen positions are computed by the caller-supplied `img_to_screen`
//! closure, which maps image-pixel coordinates (col, row as f32) to screen
//! coordinates. This keeps the rendering layer free of view-transform state.
//!
//! # Font size
//! All measurement labels use a 12 pt proportional font.

use egui::{Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};

use crate::tools::interaction::{Annotation, RoiKind, ToolState};
use crate::ui::live_preview::{live_angle_deg, live_length_mm};

// â”€â”€ colour / style constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

// â”€â”€ MeasurementLayer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
                Annotation::RoiEllipse {
                    center,
                    radii,
                    mean,
                    std_dev,
                    ..
                } => {
                    let screen_center = img_to_screen(egui::pos2((*center)[1], (*center)[0]));
                    // Map radii to screen space using the image-to-screen transform.
                    // We derive screen radii by transforming a point offset by each
                    // radius and subtracting the centre.
                    let screen_r_pt =
                        img_to_screen(egui::pos2((*center)[1], (*center)[0] + (*radii)[0]));
                    let screen_c_pt =
                        img_to_screen(egui::pos2((*center)[1] + (*radii)[1], (*center)[0]));
                    let screen_radius_row = (screen_r_pt - screen_center).length();
                    let screen_radius_col = (screen_c_pt - screen_center).length();
                    draw_roi_ellipse_annotation(
                        painter,
                        screen_center,
                        screen_radius_row,
                        screen_radius_col,
                        *mean,
                        *std_dev,
                    );
                }
            }
        }
    }

    /// Draw the in-progress tool state (e.g. a rubber-band line during length
    /// measurement, or a growing ROI rectangle), with live distance/angle labels.
    ///
    /// # Parameters
    /// - `cursor_screen`: current cursor position in *screen* coordinates.
    /// - `cursor_img`: cursor in *image* pixel coordinates `Pos2 { x: col, y: row }`;
    ///   `None` when the cursor is outside the viewport.
    /// - `spacing`: `[row_mm_per_px, col_mm_per_px]` used to compute live mm labels.
    /// - `img_to_screen`: converts image-pixel coordinates to screen coordinates.
    pub fn draw_in_progress(
        painter: &Painter,
        tool_state: &ToolState,
        cursor_screen: Option<Pos2>,
        cursor_img: Option<Pos2>,
        spacing: [f32; 2],
        img_to_screen: impl Fn(Pos2) -> Pos2,
    ) {
        match tool_state {
            ToolState::Idle => {}
            ToolState::MeasureLength1 { p1 } => {
                // Rubber-band line from anchor to cursor with live distance label.
                let sp1 = img_to_screen(*p1);
                painter.circle_filled(sp1, HANDLE_RADIUS, COLOR_MEASURE);
                if let Some(cursor) = cursor_screen {
                    painter.line_segment([sp1, cursor], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    // Live distance label at midpoint, offset 12 px upward.
                    if let Some(cimg) = cursor_img {
                        let mm = live_length_mm([p1.y, p1.x], [cimg.y, cimg.x], spacing);
                        let label = format!("{:.1} mm", mm);
                        let mid = Pos2::new((sp1.x + cursor.x) * 0.5, (sp1.y + cursor.y) * 0.5);
                        painter.text(
                            mid + Vec2::new(0.0, -12.0),
                            egui::Align2::CENTER_CENTER,
                            label,
                            label_font(),
                            COLOR_LABEL,
                        );
                    }
                }
            }
            ToolState::MeasureAngle2 { p1, p2 } => {
                // Two anchor handles, p1â†’p2 line, rubber-band p2â†’cursor with live angle.
                let sp1 = img_to_screen(*p1);
                let sp2 = img_to_screen(*p2);
                painter.circle_filled(sp1, HANDLE_RADIUS, COLOR_MEASURE);
                painter.circle_filled(sp2, HANDLE_RADIUS, COLOR_MEASURE);
                painter.line_segment([sp1, sp2], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                if let Some(cursor) = cursor_screen {
                    painter.line_segment([sp2, cursor], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    // Live angle label at the vertex (p2), offset 12 px up-right.
                    if let Some(cimg) = cursor_img {
                        let deg = live_angle_deg([p1.y, p1.x], [p2.y, p2.x], [cimg.y, cimg.x]);
                        let label = format!("{:.1}Â°", deg);
                        painter.text(
                            sp2 + Vec2::new(8.0, -12.0),
                            egui::Align2::LEFT_CENTER,
                            label,
                            label_font(),
                            COLOR_LABEL,
                        );
                    }
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
            ToolState::Panning { .. }
            | ToolState::Zooming { .. }
            | ToolState::WindowLevelDrag { .. } => {}
        }
    }
}

// â”€â”€ private drawing functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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
    painter.circle_stroke(p2, 8.0, Stroke::new(1.0_f32, COLOR_MEASURE));
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
    let label = format!("{:.1}Â°", angle_deg);
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
/// The rectangle is drawn with [`COLOR_ROI`]. The label shows mean Â± std.
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
    let label = format!("Î¼={:.1} Ïƒ={:.1}", mean, std);
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

/// Draw an ellipse ROI annotation with a statistics label.
///
/// The ellipse is drawn with [`COLOR_ROI`]. The label shows `Î¼ Â± Ïƒ`.
///
/// # Parameters
/// - `center` â€” ellipse centre in screen coordinates.
/// - `radius_x` â€” horizontal screen-space radius (col direction).
/// - `radius_y` â€” vertical screen-space radius (row direction).
/// - `mean`, `std` â€” statistics to render in the label.
///
/// # Label placement
/// The label is placed immediately below the bottom of the ellipse.
fn draw_roi_ellipse_annotation(
    painter: &Painter,
    center: Pos2,
    radius_x: f32,
    radius_y: f32,
    mean: f32,
    std: f32,
) {
    painter.add(egui::epaint::EllipseShape {
        center,
        radius: Vec2::new(radius_x, radius_y),
        fill: Color32::TRANSPARENT,
        stroke: Stroke::new(LINE_WIDTH, COLOR_ROI),
    });
    // Corner handles at cardinal points.
    for handle in &[
        Pos2::new(center.x, center.y - radius_y), // top
        Pos2::new(center.x + radius_x, center.y), // right
        Pos2::new(center.x, center.y + radius_y), // bottom
        Pos2::new(center.x - radius_x, center.y), // left
    ] {
        painter.circle_filled(*handle, HANDLE_RADIUS, COLOR_ROI);
    }
    // Label below the bottom of the ellipse.
    let label = format!("Î¼={:.1} Ïƒ={:.1}", mean, std);
    let label_pos = Pos2::new(center.x, center.y + radius_y + 12.0);
    painter.text(
        label_pos,
        egui::Align2::CENTER_TOP,
        label,
        label_font(),
        COLOR_LABEL,
    );
}

// â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Compute a perpendicular offset vector of length `distance` from the line
/// `p1 â†’ p2`.
///
/// The perpendicular is rotated 90Â° counter-clockwise from the line direction.
/// When `p1 == p2` (degenerate line), returns a straight-up offset `(0, -d)`.
fn perpendicular_offset(p1: Pos2, p2: Pos2, distance: f32) -> Vec2 {
    let d = p2 - p1;
    let len = d.length();
    if len < 1e-4 {
        return Vec2::new(0.0, -distance);
    }
    // 90Â° CCW rotation: (dx, dy) â†’ (-dy, dx)
    Vec2::new(-d.y / len, d.x / len) * distance
}

#[cfg(test)]
mod tests;
