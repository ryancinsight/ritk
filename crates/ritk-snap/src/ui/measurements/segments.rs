//! Angle and deviation marks in the eframe image pane.

use super::{egui_point, label_font, COLOR_LABEL, COLOR_MEASURE, HANDLE_RADIUS, LINE_WIDTH};
use crate::tools::interaction::ImagePoint;
use egui::{Align2, Painter, Pos2, Stroke, Vec2};

pub(super) fn draw_pencil(painter: &Painter, points: &[[f32; 2]], project: impl Fn(Pos2) -> Pos2) {
    for pair in points.windows(2) {
        painter.line_segment(
            [
                project(Pos2::new(pair[0][1], pair[0][0])),
                project(Pos2::new(pair[1][1], pair[1][0])),
            ],
            Stroke::new(LINE_WIDTH, COLOR_MEASURE),
        );
    }
}

pub(super) fn draw_deviation(
    painter: &Painter,
    points: [[f32; 2]; 3],
    distance_mm: f32,
    angle_deg: f32,
    project: impl Fn(Pos2) -> Pos2,
) {
    let [a, b, baseline] = points.map(|[row, col]| project(Pos2::new(col, row)));
    let stroke = Stroke::new(LINE_WIDTH, COLOR_MEASURE);
    painter.line_segment([a, b], stroke);
    painter.line_segment([a, baseline], stroke);
    painter.text(
        b + Vec2::new(8.0, -12.0),
        Align2::LEFT_CENTER,
        format!("{distance_mm:.1} mm {angle_deg:.1}°"),
        label_font(),
        COLOR_LABEL,
    );
}

pub(super) fn draw_deviation_preview(
    painter: &Painter,
    points: [ImagePoint; 2],
    project: impl Fn(Pos2) -> Pos2,
) {
    painter.line_segment(
        points.map(|point| project(egui_point(point))),
        Stroke::new(LINE_WIDTH, COLOR_MEASURE),
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

pub(super) fn draw_arrow(painter: &Painter, tail: egui::Pos2, tip: egui::Pos2) {
    let stroke = Stroke::new(LINE_WIDTH, COLOR_MEASURE);
    painter.line_segment([tail, tip], stroke);
    if let Some(wings) = metis_ui_lang::arrow_wings(
        [f64::from(tail.x), f64::from(tail.y)],
        [f64::from(tip.x), f64::from(tip.y)],
        8.0,
        4.0,
    ) {
        for wing in wings {
            painter.line_segment(
                [
                    tip,
                    egui::Pos2::new(
                        <f32 as FloatElement>::from_f64(wing[0]),
                        <f32 as FloatElement>::from_f64(wing[1]),
                    ),
                ],
                stroke,
            );
        }
    }
}
