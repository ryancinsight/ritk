use super::{
    egui_point, label_font, segments, MeasurementLayer, COLOR_LABEL, COLOR_MEASURE, COLOR_ROI,
    HANDLE_RADIUS, LINE_WIDTH,
};
use crate::tools::interaction::{PolygonKind, RoiKind, ToolState};
use crate::ui::live_preview::{live_angle_deg, live_length_mm};
use egui::{Color32, Painter, Pos2, Rect, Stroke, Vec2};

impl MeasurementLayer {
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
                let mut vertices = points
                    .as_slice()
                    .iter()
                    .map(|point| img_to_screen(egui_point(*point)));
                if let Some(first) = vertices.next() {
                    let stroke = Stroke::new(LINE_WIDTH, COLOR_ROI);
                    let mut last = first;
                    for vertex in vertices {
                        painter.line_segment([last, vertex], stroke);
                        last = vertex;
                    }
                    if *kind == PolygonKind::Closed && points.as_slice().len() >= 3 {
                        painter.line_segment([last, first], stroke);
                    }
                    if let Some(cursor) = cursor_screen {
                        painter.line_segment([last, cursor], stroke);
                    }
                }
            }
            // Panning and window-level drag produce no visual annotation.
            ToolState::Panning { .. }
            | ToolState::Zooming { .. }
            | ToolState::WindowLevelDrag { .. } => {}
            ToolState::DeviationDrag { start, current } => {
                segments::draw_deviation_preview(painter, [*start, *current], &img_to_screen)
            }
            ToolState::ArrowDrag { start, current } => {
                let tail = img_to_screen(egui_point(*start));
                let head = img_to_screen(egui_point(*current));
                segments::draw_arrow(painter, tail, head);
            }
            ToolState::PencilDrag { capture } => {
                let points: Vec<_> = capture
                    .as_slice()
                    .iter()
                    .map(|point| [point.y(), point.x()])
                    .collect();
                segments::draw_pencil(painter, &points, &img_to_screen);
            }
        }
    }
}
