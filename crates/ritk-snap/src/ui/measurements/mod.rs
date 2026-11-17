//! Measurement annotation rendering layer.
//!
//! # Drawing conventions
//!
//! | Annotation type | Colour         | Line width |
//! |-----------------|----------------|------------|
//! | Length           | YELLOW         | 1.5 px     |
//! | Angle            | YELLOW         | 1.5 px     |
//! | ROI (rect)       | GREEN          | 1.5 px     |
//! | HU point         | CYAN           | —          |
//! | Labels           | WHITE          | —          |
//!
//! All screen positions are computed by the caller-supplied `img_to_screen`
//! closure, which maps image-pixel coordinates (col, row as f32) to screen
//! coordinates. This keeps the rendering layer free of view-transform state.
//!
//! # Font size
//! All measurement labels use a 12 pt proportional font.

mod drawing;
#[cfg(test)]
mod tests;
use self::drawing::{
    draw_angle_annotation, draw_arrow, draw_hu_point, draw_length_annotation, draw_polyline,
    draw_roi_ellipse_annotation, draw_roi_rect_annotation,
};

use egui::{Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};

use crate::tools::interaction::{
    Annotation, ImagePoint, PathKind, PolygonKind, RoiKind, ToolState,
};
use crate::ui::live_preview::{live_angle_deg, live_length_mm};

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

fn egui_point(point: ImagePoint) -> Pos2 {
    Pos2::new(point.x(), point.y())
}

// ── MeasurementLayer ──────────────────────────────────────────────────────────

/// Stateless annotation rendering helpers.
///
/// Call [`MeasurementLayer::draw_annotations`] every frame to repaint the
/// completed image-space annotations for the current plane, and
/// [`MeasurementLayer::draw_in_progress`] to render the currently active
/// in-progress tool state. Patient-space lengths require the oblique plane
/// transform and are omitted from this orthogonal image-space layer.
pub struct MeasurementLayer;

impl MeasurementLayer {
    /// Draw completed image-space [`Annotation`]s stored in `annotations`.
    ///
    /// `img_to_screen` converts image-pixel coordinates `(col, row)` stored in
    /// the annotations as `[row, col]` to screen coordinates. Patient-space
    /// lengths are skipped because this layer does not carry their oblique
    /// plane transform.
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
                // Patient-space annotations require the reslice plane that
                // created them. Orthogonal views must not project them as if
                // their endpoints were two-dimensional image coordinates.
                Annotation::PatientLength { .. } => {}
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
                Annotation::ClosedPolygon {
                    points,
                    area_mm2,
                    perimeter_mm,
                    ..
                } => {
                    let screen: Vec<Pos2> = points
                        .iter()
                        .map(|p| img_to_screen(egui::pos2((*p)[1], (*p)[0])))
                        .collect();
                    draw_polyline(painter, &screen, true, COLOR_ROI);
                    if let Some(anchor) = screen.first() {
                        painter.text(
                            *anchor + Vec2::new(8.0, 8.0),
                            egui::Align2::LEFT_TOP,
                            format!("{area_mm2:.1} mm²  {perimeter_mm:.1} mm"),
                            label_font(),
                            COLOR_LABEL,
                        );
                    }
                }
                Annotation::OpenPolygon {
                    points, length_mm, ..
                } => {
                    let screen: Vec<Pos2> = points
                        .iter()
                        .map(|p| img_to_screen(egui::pos2((*p)[1], (*p)[0])))
                        .collect();
                    draw_polyline(painter, &screen, false, COLOR_ROI);
                    if let Some(anchor) = screen.first() {
                        painter.text(
                            *anchor + Vec2::new(8.0, 8.0),
                            egui::Align2::LEFT_TOP,
                            format!("{length_mm:.1} mm"),
                            label_font(),
                            COLOR_LABEL,
                        );
                    }
                }
                Annotation::CobbAngle {
                    p1,
                    p2,
                    p3,
                    p4,
                    angle_deg,
                    ..
                } => {
                    let a = img_to_screen(egui::pos2(p1[1], p1[0]));
                    let b = img_to_screen(egui::pos2(p2[1], p2[0]));
                    let c = img_to_screen(egui::pos2(p3[1], p3[0]));
                    let d = img_to_screen(egui::pos2(p4[1], p4[0]));
                    painter.line_segment([a, b], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    painter.line_segment([c, d], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    painter.text(
                        b + Vec2::new(8.0, -12.0),
                        egui::Align2::LEFT_CENTER,
                        format!("{angle_deg:.1}°"),
                        label_font(),
                        COLOR_LABEL,
                    );
                }
                Annotation::Deviation {
                    start,
                    end,
                    reference,
                    distance_mm,
                    angle_deg,
                    ..
                } => {
                    let a = img_to_screen(egui::pos2(start[1], start[0]));
                    let b = img_to_screen(egui::pos2(end[1], end[0]));
                    let p = img_to_screen(egui::pos2(reference[1], reference[0]));
                    painter.line_segment([a, b], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    painter.line_segment([a, p], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    painter.line_segment([p, b], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    painter.text(
                        p + Vec2::new(8.0, 8.0),
                        egui::Align2::LEFT_TOP,
                        format!("{distance_mm:.1} mm  {angle_deg:.1}°"),
                        label_font(),
                        COLOR_LABEL,
                    );
                }
                Annotation::Arrow { tail, head, .. } => {
                    let a = img_to_screen(egui::pos2(tail[1], tail[0]));
                    let b = img_to_screen(egui::pos2(head[1], head[0]));
                    draw_arrow(painter, a, b);
                }
                Annotation::Pencil { points, .. } => {
                    let screen: Vec<Pos2> = points
                        .iter()
                        .map(|p| img_to_screen(egui::pos2((*p)[1], (*p)[0])))
                        .collect();
                    draw_polyline(painter, &screen, false, COLOR_MEASURE);
                }
            }
        }
    }

    /// Draw the in-progress tool state (e.g. a rubber-band line during length
    /// measurement, or a growing ROI rectangle), with live distance/angle labels.
    ///
    /// # Parameters
    /// - `cursor_screen`: current cursor position in *screen* coordinates.
    /// - `cursor_img`: cursor in *image* pixel coordinates as `Pos2 { x: col,
    ///   y: row }`; `None` when the cursor is outside the viewport.
    /// - `spacing`: validated `[row_mm_per_px, col_mm_per_px]`, or `None` when
    ///   the physical value cannot be represented.
    /// - `img_to_screen`: converts image-pixel coordinates to screen coordinates.
    pub fn draw_in_progress(
        painter: &Painter,
        tool_state: &ToolState,
        cursor_screen: Option<Pos2>,
        cursor_img: Option<Pos2>,
        spacing: Option<[f32; 2]>,
        img_to_screen: impl Fn(Pos2) -> Pos2,
    ) {
        match tool_state {
            ToolState::Idle => {}
            ToolState::MeasureLength1 { p1 } | ToolState::LengthDrag { start: p1, .. } => {
                // Rubber-band line from anchor to cursor with live distance label.
                let sp1 = img_to_screen(egui_point(*p1));
                painter.circle_filled(sp1, HANDLE_RADIUS, COLOR_MEASURE);
                if let Some(cursor) = cursor_screen {
                    painter.line_segment([sp1, cursor], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    // Live distance label at midpoint, offset 12 px upward.
                    if let (Some(cimg), Some(spacing)) = (cursor_img, spacing) {
                        let mm = live_length_mm([p1.y(), p1.x()], [cimg.y, cimg.x], spacing);
                        if mm.is_finite() {
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
            }
            ToolState::MeasureAngle2 { p1, p2 } => {
                // Two anchor handles, p1→p2 line, rubber-band p2→cursor with live angle.
                let sp1 = img_to_screen(egui_point(*p1));
                let sp2 = img_to_screen(egui_point(*p2));
                painter.circle_filled(sp1, HANDLE_RADIUS, COLOR_MEASURE);
                painter.circle_filled(sp2, HANDLE_RADIUS, COLOR_MEASURE);
                painter.line_segment([sp1, sp2], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                if let Some(cursor) = cursor_screen {
                    painter.line_segment([sp2, cursor], Stroke::new(LINE_WIDTH, COLOR_MEASURE));
                    // Live angle label at the vertex (p2), offset 12 px up-right.
                    if let Some(cimg) = cursor_img {
                        let deg =
                            live_angle_deg([p1.y(), p1.x()], [p2.y(), p2.x()], [cimg.y, cimg.x]);
                        let label = format!("{:.1}°", deg);
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
                let ss = img_to_screen(egui_point(*start));
                let sc = img_to_screen(egui_point(*current));
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
            ToolState::Polygon { points, kind } => {
                let screen: Vec<Pos2> = points
                    .iter()
                    .map(|p| img_to_screen(egui_point(*p)))
                    .collect();
                if let Some(cursor) = cursor_screen {
                    let mut preview = screen.clone();
                    preview.push(cursor);
                    draw_polyline(painter, &preview, *kind == PolygonKind::Closed, COLOR_ROI);
                }
                for point in screen {
                    painter.circle_filled(point, HANDLE_RADIUS, COLOR_ROI);
                }
            }
            ToolState::PathDrag { points, kind, .. } => {
                let screen: Vec<Pos2> = points
                    .iter()
                    .map(|p| img_to_screen(egui_point(*p)))
                    .collect();
                if *kind == PathKind::Arrow {
                    if let (Some(a), Some(b)) = (screen.first(), screen.last()) {
                        draw_arrow(painter, *a, *b);
                    }
                } else {
                    draw_polyline(painter, &screen, false, COLOR_MEASURE);
                }
            }
            ToolState::Cobb { points } | ToolState::CobbDrag { points } => {
                for pair in points.chunks_exact(2) {
                    painter.line_segment(
                        [
                            img_to_screen(egui_point(pair[0])),
                            img_to_screen(egui_point(pair[1])),
                        ],
                        Stroke::new(LINE_WIDTH, COLOR_MEASURE),
                    );
                }
            }
            ToolState::DeviationDrag { start, current } => {
                painter.line_segment(
                    [
                        img_to_screen(egui_point(*start)),
                        img_to_screen(egui_point(*current)),
                    ],
                    Stroke::new(LINE_WIDTH, COLOR_MEASURE),
                );
            }
            // Panning and window-level drag produce no visual annotation.
            ToolState::Panning { .. }
            | ToolState::Zooming { .. }
            | ToolState::WindowLevelDrag { .. } => {}
        }
    }
}
