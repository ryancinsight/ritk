//! Measurement graphics projected over native image panes.
//!
//! Completed annotations and the gesture in progress are stored in continuous
//! source-image coordinates. They pass through the same projection as the
//! pixels, so a measurement stays on its anatomy under zoom, pan and rotation,
//! and each segment is clipped to the visible image so it cannot paint chrome.

use anyhow::Result;
use metis_platform::{Color, Framebuffer};
use metis_ui_lang::DisplayList;

use super::super::frame::RenderedView;
use super::geometry::{
    screen_coordinate, source_point_to_screen, visible_image_bounds, NativeViewport,
};
use super::text::{draw_text, text_style, PANEL_DETAIL_SIZE};
use crate::tools::interaction::{Annotation, PlacedAnnotation, RoiKind, ToolState};
use crate::ui::measurements::perpendicular_offset;

pub(crate) const MEASUREMENT_COLOR: Color = Color::rgba(255, 255, 0, 255);
pub(crate) const ROI_COLOR: Color = Color::rgba(0, 255, 0, 255);
const POINT_COLOR: Color = Color::rgba(0, 255, 255, 255);
const LABEL_COLOR: Color = Color::rgba(255, 255, 255, 255);

/// Polygon resolution of a drawn ellipse.
///
/// The chord of a 64-gon deviates from its circle by `r(1 - cos(pi / 64))`,
/// which stays under one device pixel up to a radius of about 830 pixels.
const ELLIPSE_SEGMENTS: u16 = 64;
const POINT_MARK_HALF_WIDTH: f64 = 4.0;
const LABEL_OFFSET: f64 = 6.0;
/// Device distance between a length line and its label, beside its midpoint.
pub(crate) const LENGTH_LABEL_DISTANCE: f64 = 12.0;

/// One text label at device coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct MeasurementLabel {
    pub(crate) x: i32,
    pub(crate) y: i32,
    pub(crate) text: String,
}

/// Line work and labels for one image pane.
pub(crate) struct MeasurementOverlay {
    pub(crate) lines: DisplayList,
    pub(crate) labels: Vec<MeasurementLabel>,
}

impl MeasurementOverlay {
    /// Paint the overlay onto `framebuffer`.
    ///
    /// # Errors
    /// Returns an error when the label text style is outside the host range.
    pub(crate) fn render_to(&self, framebuffer: &mut Framebuffer) -> Result<()> {
        self.lines.render_to(framebuffer);
        let style = text_style(LABEL_COLOR, PANEL_DETAIL_SIZE)?;
        for label in &self.labels {
            draw_text(framebuffer, label.x, label.y, &label.text, style);
        }
        Ok(())
    }
}

/// The text shown beside a completed annotation.
pub(crate) fn measurement_label(annotation: &Annotation) -> Option<String> {
    match annotation {
        Annotation::Length { length_mm, .. } => Some(format!("{length_mm:.1} mm")),
        Annotation::PatientLength(_) => None,
        Annotation::Angle { angle_deg, .. } => Some(format!("{angle_deg:.1} deg")),
        Annotation::RoiRect {
            mean,
            std_dev,
            area_mm2,
            ..
        }
        | Annotation::RoiEllipse {
            mean,
            std_dev,
            area_mm2,
            ..
        } => Some(format!(
            "Mean {mean:.1}  SD {std_dev:.1}  {area_mm2:.1} mm2"
        )),
        Annotation::HuPoint { value, .. } => Some(format!("{value:.0}")),
    }
}

/// Build the overlay for one pane.
///
/// `gesture` is the tool state to preview; panes other than the one being
/// interacted with pass [`ToolState::Idle`].
///
/// # Errors
/// Returns an error when a projected coordinate is outside the host range or
/// the display list cannot grow.
pub(crate) fn measurement_overlay(
    view: &RenderedView,
    viewport: &NativeViewport,
    annotations: &[PlacedAnnotation],
    gesture: &ToolState,
) -> Result<MeasurementOverlay> {
    let mut overlay = MeasurementOverlay {
        lines: DisplayList::default(),
        labels: Vec::new(),
    };
    let Some(clip) = visible_image_bounds(viewport) else {
        return Ok(overlay);
    };
    let pane = Pane {
        view,
        viewport,
        clip,
    };
    for placed in annotations
        .iter()
        .filter(|placed| placed.is_visible_on(view.axis, view.slice_index))
    {
        pane.draw_annotation(&placed.annotation, &mut overlay)?;
    }
    pane.draw_gesture(gesture, &mut overlay)?;
    Ok(overlay)
}

struct Pane<'a> {
    view: &'a RenderedView,
    viewport: &'a NativeViewport,
    clip: [f64; 4],
}

impl Pane<'_> {
    /// Project a stored `[row, col]` point to device coordinates.
    fn point(&self, image: [f32; 2]) -> Result<Option<[f64; 2]>> {
        source_point_to_screen(
            self.view,
            self.viewport,
            [f64::from(image[1]), f64::from(image[0])],
        )
    }

    fn inside(&self, [x, y]: [f64; 2]) -> bool {
        let [left, top, right, bottom] = self.clip;
        x >= left && x <= right && y >= top && y <= bottom
    }

    fn segment(
        &self,
        list: &mut DisplayList,
        a: [f64; 2],
        b: [f64; 2],
        color: Color,
    ) -> Result<()> {
        if let Some((start, end)) = clip_segment(a, b, self.clip) {
            list.append_line(screen_point(start)?, screen_point(end)?, color)?;
        }
        Ok(())
    }

    fn image_segment(
        &self,
        list: &mut DisplayList,
        a: [f32; 2],
        b: [f32; 2],
        color: Color,
    ) -> Result<()> {
        if let (Some(a), Some(b)) = (self.point(a)?, self.point(b)?) {
            self.segment(list, a, b, color)?;
        }
        Ok(())
    }

    fn rectangle(
        &self,
        list: &mut DisplayList,
        first: [f32; 2],
        opposite: [f32; 2],
        color: Color,
    ) -> Result<()> {
        let corners = [
            first,
            [first[0], opposite[1]],
            opposite,
            [opposite[0], first[1]],
        ];
        for (index, corner) in corners.iter().enumerate() {
            self.image_segment(list, *corner, corners[(index + 1) % corners.len()], color)?;
        }
        Ok(())
    }

    fn ellipse(
        &self,
        list: &mut DisplayList,
        center: [f32; 2],
        radii: [f32; 2],
        color: Color,
    ) -> Result<()> {
        let vertex = |index: u16| {
            let angle = std::f32::consts::TAU * f32::from(index % ELLIPSE_SEGMENTS)
                / f32::from(ELLIPSE_SEGMENTS);
            [
                center[0] + radii[0] * angle.sin(),
                center[1] + radii[1] * angle.cos(),
            ]
        };
        for index in 0..ELLIPSE_SEGMENTS {
            self.image_segment(list, vertex(index), vertex(index + 1), color)?;
        }
        Ok(())
    }

    fn point_mark(&self, list: &mut DisplayList, image: [f32; 2], color: Color) -> Result<()> {
        if let Some([x, y]) = self.point(image)? {
            let half = POINT_MARK_HALF_WIDTH;
            self.segment(list, [x - half, y], [x + half, y], color)?;
            self.segment(list, [x, y - half], [x, y + half], color)?;
        }
        Ok(())
    }

    fn label(
        &self,
        overlay: &mut MeasurementOverlay,
        anchor: [f32; 2],
        text: String,
    ) -> Result<()> {
        match self.point(anchor)? {
            Some([x, y]) => self.label_at(overlay, [x + LABEL_OFFSET, y + LABEL_OFFSET], text),
            None => Ok(()),
        }
    }

    /// Place `text` at a device point, when the point lies on the visible image.
    fn label_at(
        &self,
        overlay: &mut MeasurementOverlay,
        point: [f64; 2],
        text: String,
    ) -> Result<()> {
        if self.inside(point) {
            overlay.labels.push(MeasurementLabel {
                x: screen_coordinate(point[0], "measurement label x")?,
                y: screen_coordinate(point[1], "measurement label y")?,
                text,
            });
        }
        Ok(())
    }

    /// Label a length beside its line's midpoint, perpendicular to the line as
    /// drawn, so the label follows the line through rotation and flips.
    fn length_label(
        &self,
        overlay: &mut MeasurementOverlay,
        p1: [f32; 2],
        p2: [f32; 2],
        text: String,
    ) -> Result<()> {
        let (Some(start), Some(end)) = (self.point(p1)?, self.point(p2)?) else {
            return Ok(());
        };
        let [dx, dy] = perpendicular_offset(start, end, LENGTH_LABEL_DISTANCE);
        let midpoint = [(start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0];
        self.label_at(overlay, [midpoint[0] + dx, midpoint[1] + dy], text)
    }

    fn draw_annotation(
        &self,
        annotation: &Annotation,
        overlay: &mut MeasurementOverlay,
    ) -> Result<()> {
        let anchor = match annotation {
            Annotation::Length { p1, p2, .. } => {
                self.image_segment(&mut overlay.lines, *p1, *p2, MEASUREMENT_COLOR)?;
                if let Some(text) = measurement_label(annotation) {
                    self.length_label(overlay, *p1, *p2, text)?;
                }
                None
            }
            Annotation::PatientLength(_) => None,
            Annotation::Angle { p1, p2, p3, .. } => {
                self.image_segment(&mut overlay.lines, *p1, *p2, MEASUREMENT_COLOR)?;
                self.image_segment(&mut overlay.lines, *p2, *p3, MEASUREMENT_COLOR)?;
                Some(*p2)
            }
            Annotation::RoiRect {
                top_left,
                bottom_right,
                ..
            } => {
                self.rectangle(&mut overlay.lines, *top_left, *bottom_right, ROI_COLOR)?;
                Some(*bottom_right)
            }
            Annotation::RoiEllipse { center, radii, .. } => {
                self.ellipse(&mut overlay.lines, *center, *radii, ROI_COLOR)?;
                Some(*center)
            }
            Annotation::HuPoint { pos, .. } => {
                self.point_mark(&mut overlay.lines, *pos, POINT_COLOR)?;
                Some(*pos)
            }
        };
        if let (Some(anchor), Some(text)) = (anchor, measurement_label(annotation)) {
            self.label(overlay, anchor, text)?;
        }
        Ok(())
    }

    fn draw_gesture(&self, gesture: &ToolState, overlay: &mut MeasurementOverlay) -> Result<()> {
        let lines = &mut overlay.lines;
        match gesture {
            ToolState::LengthDrag { start, current } => self.image_segment(
                lines,
                [start.y(), start.x()],
                [current.y(), current.x()],
                MEASUREMENT_COLOR,
            ),
            ToolState::MeasureLength1 { p1 } => {
                self.point_mark(lines, [p1.y(), p1.x()], MEASUREMENT_COLOR)
            }
            ToolState::MeasureAngle2 { p1, p2 } => {
                self.image_segment(lines, [p1.y(), p1.x()], [p2.y(), p2.x()], MEASUREMENT_COLOR)
            }
            ToolState::RoiDrag {
                start,
                current,
                kind: RoiKind::Rect,
            } => self.rectangle(
                lines,
                [start.y(), start.x()],
                [current.y(), current.x()],
                ROI_COLOR,
            ),
            ToolState::RoiDrag {
                start,
                current,
                kind: RoiKind::Ellipse,
            } => {
                let center = [
                    (start.y() + current.y()) / 2.0,
                    (start.x() + current.x()) / 2.0,
                ];
                let radii = [
                    (start.y() - current.y()).abs() / 2.0,
                    (start.x() - current.x()).abs() / 2.0,
                ];
                self.ellipse(lines, center, radii, ROI_COLOR)
            }
            ToolState::Idle
            | ToolState::Panning { .. }
            | ToolState::Zooming { .. }
            | ToolState::WindowLevelDrag { .. } => Ok(()),
        }
    }
}

fn screen_point([x, y]: [f64; 2]) -> Result<(i32, i32)> {
    Ok((
        screen_coordinate(x, "measurement x")?,
        screen_coordinate(y, "measurement y")?,
    ))
}

/// Clip a segment to `[left, top, right, bottom]` (Liang and Barsky, 1984).
fn clip_segment(a: [f64; 2], b: [f64; 2], clip: [f64; 4]) -> Option<([f64; 2], [f64; 2])> {
    let [left, top, right, bottom] = clip;
    let delta = [b[0] - a[0], b[1] - a[1]];
    let (mut enter, mut leave) = (0.0_f64, 1.0_f64);
    for (direction, distance) in [
        (-delta[0], a[0] - left),
        (delta[0], right - a[0]),
        (-delta[1], a[1] - top),
        (delta[1], bottom - a[1]),
    ] {
        if direction == 0.0 {
            if distance < 0.0 {
                return None;
            }
        } else {
            let crossing = distance / direction;
            if direction < 0.0 {
                enter = enter.max(crossing);
            } else {
                leave = leave.min(crossing);
            }
            if enter > leave {
                return None;
            }
        }
    }
    let at = |t: f64| [a[0] + t * delta[0], a[1] + t * delta[1]];
    Some((at(enter), at(leave)))
}

#[cfg(test)]
mod tests {
    use super::clip_segment;

    const CLIP: [f64; 4] = [0.0, 0.0, 10.0, 10.0];

    #[test]
    fn a_segment_inside_the_clip_is_unchanged() {
        assert_eq!(
            clip_segment([1.0, 2.0], [3.0, 4.0], CLIP),
            Some(([1.0, 2.0], [3.0, 4.0]))
        );
    }

    #[test]
    fn a_crossing_segment_is_cut_at_the_boundary() {
        assert_eq!(
            clip_segment([-5.0, 5.0], [15.0, 5.0], CLIP),
            Some(([0.0, 5.0], [10.0, 5.0]))
        );
    }

    #[test]
    fn a_segment_outside_the_clip_is_rejected() {
        assert_eq!(clip_segment([11.0, 0.0], [20.0, 9.0], CLIP), None);
    }

    #[test]
    fn a_parallel_segment_outside_the_clip_is_rejected() {
        assert_eq!(clip_segment([-1.0, 2.0], [-1.0, 8.0], CLIP), None);
    }
}
