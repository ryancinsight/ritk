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

use egui::{Color32, FontId, Painter, Pos2, Rect, Stroke, Vec2};

use crate::tools::interaction::{Annotation, ImagePoint};

mod preview;
mod segments;

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
                Annotation::Arrow { tail, head, .. } => {
                    let tail = img_to_screen(egui::pos2((*tail)[1], (*tail)[0]));
                    let head = img_to_screen(egui::pos2((*head)[1], (*head)[0]));
                    segments::draw_arrow(painter, tail, head);
                }
                Annotation::Pencil { points, .. } => {
                    segments::draw_pencil(painter, points, &img_to_screen);
                }
                // Patient-space annotations require the reslice plane that
                // created them. Orthogonal views must not project them as if
                // their endpoints were two-dimensional image coordinates.
                Annotation::PatientLength { .. } => {}
                Annotation::Deviation {
                    start,
                    end,
                    reference,
                    distance_mm,
                    angle_deg,
                    ..
                } => segments::draw_deviation(
                    painter,
                    [*start, *end, *reference],
                    *distance_mm,
                    *angle_deg,
                    &img_to_screen,
                ),
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
                    segments::draw_angle_annotation(painter, sp1, sp2, sp3, *angle_deg);
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
                    for pair in screen.windows(2) {
                        painter
                            .line_segment([pair[0], pair[1]], Stroke::new(LINE_WIDTH, COLOR_ROI));
                    }
                    if let (Some(first), Some(last)) = (screen.first(), screen.last()) {
                        painter.line_segment([*last, *first], Stroke::new(LINE_WIDTH, COLOR_ROI));
                        painter.text(
                            *first + Vec2::new(8.0, -12.0),
                            egui::Align2::LEFT_CENTER,
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
                    for pair in screen.windows(2) {
                        painter
                            .line_segment([pair[0], pair[1]], Stroke::new(LINE_WIDTH, COLOR_ROI));
                    }
                    if let Some(first) = screen.first() {
                        painter.text(
                            *first + Vec2::new(8.0, -12.0),
                            egui::Align2::LEFT_CENTER,
                            format!("{length_mm:.1} mm"),
                            label_font(),
                            COLOR_LABEL,
                        );
                    }
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
fn perpendicular_offset(p1: Pos2, p2: Pos2, distance: f32) -> Vec2 {
    let d = p2 - p1;
    let len = d.length();
    if len < 1e-4 {
        return Vec2::new(0.0, -distance);
    }
    // 90° CCW rotation: (dx, dy) → (-dy, dx)
    Vec2::new(-d.y / len, d.x / len) * distance
}

#[cfg(test)]
mod tests;
