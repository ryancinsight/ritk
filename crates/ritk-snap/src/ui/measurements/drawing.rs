//! Drawing primitives for eframe measurement overlays.

use super::{
    label_font, COLOR_HU_POINT, COLOR_LABEL, COLOR_MEASURE, COLOR_ROI, HANDLE_RADIUS, LINE_WIDTH,
};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};

/// Draw a length measurement annotation: a line with endpoint handles and
/// a label showing the distance in mm.
///
/// # Label placement
/// The label is positioned at the midpoint of the line, offset 8 px
/// perpendicular to the line direction to avoid overlap.
pub(super) fn draw_length_annotation(painter: &Painter, p1: Pos2, p2: Pos2, length_mm: f32) {
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
pub(super) fn draw_angle_annotation(
    painter: &Painter,
    p1: Pos2,
    p2: Pos2,
    p3: Pos2,
    angle_deg: f32,
) {
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
pub(super) fn draw_roi_rect_annotation(painter: &Painter, tl: Pos2, br: Pos2, mean: f32, std: f32) {
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
    let label = format!("μ={:.1} σ={:.1}", mean, std);
    let label_pos = Pos2::new(rect.center().x, br.y + 12.0);
    painter.text(
        label_pos,
        egui::Align2::CENTER_TOP,
        label,
        label_font(),
        COLOR_LABEL,
    );
}

pub(super) fn draw_polyline(painter: &Painter, points: &[Pos2], closed: bool, color: Color32) {
    for pair in points.windows(2) {
        painter.line_segment([pair[0], pair[1]], Stroke::new(LINE_WIDTH, color));
    }
    if closed && let (Some(first), Some(last)) = (points.first(), points.last()) {
        painter.line_segment([*last, *first], Stroke::new(LINE_WIDTH, color));
    }
}

pub(super) fn draw_arrow(painter: &Painter, tail: Pos2, head: Pos2) {
    painter.line_segment([tail, head], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
    let direction = head - tail;
    let length = direction.length();
    if length > f32::EPSILON {
        let unit = direction / length;
        let side = Vec2::new(-unit.y, unit.x);
        let size = 8.0;
        painter.line_segment(
            [head, head - unit * size + side * 4.0],
            Stroke::new(LINE_WIDTH, COLOR_MEASURE),
        );
        painter.line_segment(
            [head, head - unit * size - side * 4.0],
            Stroke::new(LINE_WIDTH, COLOR_MEASURE),
        );
    }
}

/// Draw a single HU point annotation: a small crosshair circle with the
/// pixel value label.
pub(super) fn draw_hu_point(painter: &Painter, pos: Pos2, value: f32) {
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
/// The ellipse is drawn with [`COLOR_ROI`]. The label shows `μ ± σ`.
///
/// # Parameters
/// - `center` — ellipse centre in screen coordinates.
/// - `radius_x` — horizontal screen-space radius (col direction).
/// - `radius_y` — vertical screen-space radius (row direction).
/// - `mean`, `std` — statistics to render in the label.
///
/// # Label placement
/// The label is placed immediately below the bottom of the ellipse.
pub(super) fn draw_roi_ellipse_annotation(
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
    let label = format!("μ={:.1} σ={:.1}", mean, std);
    let label_pos = Pos2::new(center.x, center.y + radius_y + 12.0);
    painter.text(
        label_pos,
        egui::Align2::CENTER_TOP,
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
pub(super) fn perpendicular_offset(p1: Pos2, p2: Pos2, distance: f32) -> Vec2 {
    let d = p2 - p1;
    let len = d.length();
    if len < 1e-4 {
        return Vec2::new(0.0, -distance);
    }
    // 90° CCW rotation: (dx, dy) → (-dy, dx)
    Vec2::new(-d.y / len, d.x / len) * distance
}
