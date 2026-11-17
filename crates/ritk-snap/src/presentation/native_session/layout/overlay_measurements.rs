//! RITK-owned measurement and region overlays for native panes.

mod preview;

use anyhow::Result;
use metis_platform::{Color, Rect};
pub(in crate::presentation::native_session) use preview::append_measurement_preview;

use super::super::frame::RenderedView;
use super::crosshair::screen_coordinate;
use super::geometry::NativeViewport;
use super::text::{text_command, text_style, PANEL_DETAIL_SIZE};
use super::OverlayCommands;
use crate::geometry::affine::AffineTransform;
use crate::geometry::PatientPointMm;
use crate::tools::interaction::{Annotation, AnnotationLocation};

pub(crate) const MEASUREMENT_COLOR: Color = Color::rgba(255, 255, 0, 255);
const ROI_COLOR: Color = Color::rgba(0, 255, 0, 255);
const POINT_COLOR: Color = Color::rgba(0, 255, 255, 255);
const LABEL_OFFSET: i32 = 8;
const LENGTH_LABEL_OFFSET: f64 = 10.0;
// The projected ellipse's chord deviation is bounded by the maximum
// projected radius times 1 - cos(π / ELLIPSE_SEGMENTS).
const ELLIPSE_SEGMENTS: u16 = 64;

/// Draws every completed annotation that belongs to the displayed plane.
pub(crate) fn append_measurement_overlay(
    overlay: &mut OverlayCommands,
    views: &[RenderedView],
    viewports: &[NativeViewport],
    annotations: &[Annotation],
    affine: Option<AffineTransform>,
) -> Result<()> {
    for (view, viewport) in views.iter().zip(viewports.iter()) {
        let Some(pane) = Pane::new(view, viewport)? else {
            continue;
        };
        for annotation in annotations {
            if !annotation.visible_on_image(AnnotationLocation {
                plane: view.axis,
                slice: view.slice_index,
            }) {
                continue;
            }
            pane.draw(annotation, overlay, affine)?;
        }
    }
    Ok(())
}

struct Pane<'a> {
    view: &'a RenderedView,
    viewport: &'a NativeViewport,
    output_width: f64,
    output_height: f64,
}

impl<'a> Pane<'a> {
    fn new(view: &'a RenderedView, viewport: &'a NativeViewport) -> Result<Option<Self>> {
        let [width, height] = view.transform.output_size(view.source_size);
        if width == 0 || height == 0 {
            return Ok(None);
        }
        Ok(Some(Self {
            view,
            viewport,
            output_width: f64::from(u32::try_from(width)?),
            output_height: f64::from(u32::try_from(height)?),
        }))
    }

    fn point(&self, image: [f64; 2]) -> Result<(i32, i32)> {
        let output = self
            .view
            .transform
            .source_to_output_coordinates(image, self.view.source_size);
        let x = self.viewport.image.x + output[0] / self.output_width * self.viewport.image.width;
        let y = self.viewport.image.y + output[1] / self.output_height * self.viewport.image.height;
        Ok((
            screen_coordinate(x, "measurement x")?,
            screen_coordinate(y, "measurement y")?,
        ))
    }

    fn line(
        &self,
        overlay: &mut OverlayCommands,
        a: [f64; 2],
        b: [f64; 2],
        color: Color,
    ) -> Result<()> {
        let (start, end) = (self.point(a)?, self.point(b)?);
        overlay.push(metis_ui_lang::DisplayCommand::DrawLine { start, end, color });
        Ok(())
    }

    fn arrow(&self, overlay: &mut OverlayCommands, tail: [f64; 2], tip: [f64; 2]) -> Result<()> {
        let (start, end) = (self.point(tail)?, self.point(tip)?);
        let [x, y] = [f64::from(end.0), f64::from(end.1)];
        let [dx, dy] = [x - f64::from(start.0), y - f64::from(start.1)];
        let length = dx.hypot(dy);
        if length == 0.0 {
            return Ok(());
        }
        let color = MEASUREMENT_COLOR;
        overlay.push(metis_ui_lang::DisplayCommand::DrawLine { start, end, color });
        const TIP_LENGTH_PIXELS: f64 = 10.0;
        let size = TIP_LENGTH_PIXELS.min(length);
        let [ux, uy] = [dx * size / length, dy * size / length];
        for sign in [-0.5, 0.5] {
            let wing = (
                screen_coordinate(x - ux + sign * uy, "arrow wing x")?,
                screen_coordinate(y - uy - sign * ux, "arrow wing y")?,
            );
            overlay.push(metis_ui_lang::DisplayCommand::DrawLine {
                start: end,
                end: wing,
                color,
            });
        }
        Ok(())
    }

    fn label(&self, overlay: &mut OverlayCommands, anchor: [f64; 2], text: String) -> Result<()> {
        let (x, y) = self.point(anchor)?;
        self.place_label(
            overlay,
            [
                x.saturating_add(LABEL_OFFSET),
                y.saturating_add(LABEL_OFFSET),
            ],
            text,
        )
    }

    fn length_label(
        &self,
        overlay: &mut OverlayCommands,
        a: [f64; 2],
        b: [f64; 2],
        text: String,
    ) -> Result<()> {
        let (start, end) = (self.point(a)?, self.point(b)?);
        let (x0, y0) = (f64::from(start.0), f64::from(start.1));
        let (x1, y1) = (f64::from(end.0), f64::from(end.1));
        let (dx, dy) = (x1 - x0, y1 - y0);
        let norm = dx.hypot(dy);
        let (ox, oy) = if norm == 0.0 {
            (0.0, -LENGTH_LABEL_OFFSET)
        } else {
            (
                -dy / norm * LENGTH_LABEL_OFFSET,
                dx / norm * LENGTH_LABEL_OFFSET,
            )
        };
        self.place_label(
            overlay,
            [
                screen_coordinate((x0 + x1) * 0.5 + ox, "length label x")?,
                screen_coordinate((y0 + y1) * 0.5 + oy, "length label y")?,
            ],
            text,
        )
    }

    fn place_label(
        &self,
        overlay: &mut OverlayCommands,
        [x, y]: [i32; 2],
        text: String,
    ) -> Result<()> {
        let style = text_style(MEASUREMENT_COLOR, PANEL_DETAIL_SIZE)?;
        let Some(bounds) = style.extent(0, 0, &text) else {
            return Ok(());
        };
        let pane = self.viewport.panel_rect()?;
        let right =
            pane.x + pane.width - bounds.width - super::super::slice_scrollbar::RIGHT_GUTTER;
        let bottom = pane.y + pane.height - bounds.height;
        if right < pane.x || bottom < pane.y {
            return Ok(());
        }
        let x = (x + bounds.x).clamp(pane.x, right);
        let y = (y + bounds.y).clamp(pane.y, bottom);
        for row in [
            y,
            y.saturating_add(bounds.height),
            y.saturating_sub(bounds.height),
        ] {
            if (pane.y..=bottom).contains(&row)
                && !overlay.overlaps_text(Rect::new(x, row, bounds.width, bounds.height))
            {
                overlay.push(text_command(text, x - bounds.x, row - bounds.y, style));
                break;
            }
        }
        Ok(())
    }

    fn draw(
        &self,
        annotation: &Annotation,
        overlay: &mut OverlayCommands,
        affine: Option<AffineTransform>,
    ) -> Result<()> {
        match annotation {
            Annotation::Length {
                plane,
                p1,
                p2,
                length_mm,
                ..
            } if *plane == self.view.axis => {
                let a = image_point(*p1);
                let b = image_point(*p2);
                self.line(overlay, a, b, MEASUREMENT_COLOR)?;
                self.length_label(overlay, a, b, format!("{length_mm:.1} mm"))?;
            }
            Annotation::PatientLength { plane, length, .. } if *plane == self.view.axis => {
                let Some(affine) = affine else {
                    return Ok(());
                };
                let Some(a) = plane_point(self.view.axis, affine, length.start_mm()) else {
                    return Ok(());
                };
                let Some(b) = plane_point(self.view.axis, affine, length.end_mm()) else {
                    return Ok(());
                };
                self.line(overlay, a, b, MEASUREMENT_COLOR)?;
                self.length_label(overlay, a, b, format!("{:.1} mm", length.length_mm()))?;
            }
            Annotation::Angle {
                p1,
                p2,
                p3,
                angle_deg,
                ..
            } => {
                let a = image_point(*p1);
                let b = image_point(*p2);
                let c = image_point(*p3);
                self.line(overlay, a, b, MEASUREMENT_COLOR)?;
                self.line(overlay, b, c, MEASUREMENT_COLOR)?;
                self.label(overlay, b, format!("{angle_deg:.1} deg"))?;
            }
            Annotation::RoiRect {
                top_left,
                bottom_right,
                mean,
                std_dev,
                area_mm2,
                ..
            } => {
                let a = image_point(*top_left);
                let b = image_point(*bottom_right);
                let corners = [a, [a[0], b[1]], b, [b[0], a[1]]];
                for pair in corners.windows(2) {
                    self.line(overlay, pair[0], pair[1], ROI_COLOR)?;
                }
                self.line(overlay, corners[3], corners[0], ROI_COLOR)?;
                self.label(
                    overlay,
                    b,
                    format!("Mean {mean:.1} SD {std_dev:.1} {area_mm2:.1} mm2"),
                )?;
            }
            Annotation::RoiEllipse {
                center,
                radii,
                mean,
                std_dev,
                area_mm2,
                ..
            } => {
                let mut previous = None;
                for i in 0..=ELLIPSE_SEGMENTS {
                    let theta = std::f64::consts::TAU * f64::from(i % ELLIPSE_SEGMENTS)
                        / f64::from(ELLIPSE_SEGMENTS);
                    let current = [
                        f64::from(center[1]) + f64::from(radii[1]) * theta.cos(),
                        f64::from(center[0]) + f64::from(radii[0]) * theta.sin(),
                    ];
                    if let Some(previous) = previous {
                        self.line(overlay, previous, current, ROI_COLOR)?;
                    }
                    previous = Some(current);
                }
                self.label(
                    overlay,
                    image_point(*center),
                    format!("Mean {mean:.1} SD {std_dev:.1} {area_mm2:.1} mm2"),
                )?;
            }
            Annotation::HuPoint { pos, value, .. } => {
                let p = image_point(*pos);
                let half = 4.0;
                self.line(
                    overlay,
                    [p[0] - half, p[1]],
                    [p[0] + half, p[1]],
                    POINT_COLOR,
                )?;
                self.line(
                    overlay,
                    [p[0], p[1] - half],
                    [p[0], p[1] + half],
                    POINT_COLOR,
                )?;
                self.label(overlay, p, format!("{value:.0}"))?;
            }
            Annotation::ClosedPolygon {
                points,
                area_mm2,
                perimeter_mm,
                ..
            } => {
                let polygon: Vec<[f64; 2]> = points.iter().map(|p| image_point(*p)).collect();
                for pair in polygon.windows(2) {
                    self.line(overlay, pair[0], pair[1], ROI_COLOR)?;
                }
                if let (Some(first), Some(last)) = (polygon.first(), polygon.last()) {
                    self.line(overlay, *last, *first, ROI_COLOR)?;
                }
                if let Some(anchor) = polygon.first() {
                    self.label(
                        overlay,
                        *anchor,
                        format!("{area_mm2:.1} mm2  {perimeter_mm:.1} mm"),
                    )?;
                }
            }
            Annotation::OpenPolygon {
                points, length_mm, ..
            } => {
                let polygon: Vec<[f64; 2]> = points.iter().map(|p| image_point(*p)).collect();
                for pair in polygon.windows(2) {
                    self.line(overlay, pair[0], pair[1], ROI_COLOR)?;
                }
                if let (Some(first), Some(last)) = (polygon.first(), polygon.last()) {
                    self.length_label(overlay, *first, *last, format!("{length_mm:.1} mm"))?;
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
                let a = image_point(*p1);
                let b = image_point(*p2);
                let c = image_point(*p3);
                let d = image_point(*p4);
                self.line(overlay, a, b, MEASUREMENT_COLOR)?;
                self.line(overlay, c, d, MEASUREMENT_COLOR)?;
                self.label(overlay, b, format!("{angle_deg:.1} deg"))?;
            }
            Annotation::Deviation {
                start,
                end,
                reference,
                distance_mm,
                angle_deg,
                ..
            } => {
                let a = image_point(*start);
                let b = image_point(*end);
                let p = image_point(*reference);
                self.line(overlay, a, b, MEASUREMENT_COLOR)?;
                self.line(overlay, a, p, MEASUREMENT_COLOR)?;
                self.line(overlay, p, b, MEASUREMENT_COLOR)?;
                self.label(
                    overlay,
                    p,
                    format!("{distance_mm:.1} mm  {angle_deg:.1} deg"),
                )?;
            }
            Annotation::Arrow { tail, head, .. } => {
                self.arrow(overlay, image_point(*tail), image_point(*head))?;
            }
            Annotation::Pencil { points, .. } => {
                for pair in points.windows(2) {
                    self.line(
                        overlay,
                        image_point(pair[0]),
                        image_point(pair[1]),
                        MEASUREMENT_COLOR,
                    )?;
                }
            }
            _ => {}
        }
        Ok(())
    }
}

fn image_point(point: [f32; 2]) -> [f64; 2] {
    [f64::from(point[1]), f64::from(point[0])]
}

fn plane_point(axis: usize, affine: AffineTransform, point: PatientPointMm) -> Option<[f64; 2]> {
    let voxel = affine.patient_to_voxel(point.coordinates());
    if voxel.iter().any(|value| !value.is_finite()) {
        return None;
    }
    match axis {
        0 => Some([voxel[2], voxel[1]]),
        1 => Some([voxel[2], voxel[0]]),
        2 => Some([voxel[1], voxel[0]]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{Annotation, AnnotationLocation};

    #[test]
    fn anchored_annotation_requires_matching_plane_and_slice() {
        let annotation = Annotation::HuPoint {
            location: Some(AnnotationLocation { plane: 1, slice: 4 }),
            pos: [0.0, 0.0],
            value: -10.0,
        };

        assert!(annotation.visible_on_image(AnnotationLocation { plane: 1, slice: 4 }));
        assert!(!annotation.visible_on_image(AnnotationLocation { plane: 0, slice: 4 }));
        assert!(!annotation.visible_on_image(AnnotationLocation { plane: 1, slice: 3 }));
    }

    #[test]
    fn legacy_point_without_image_location_does_not_claim_an_image() {
        let annotation = Annotation::HuPoint {
            location: None,
            pos: [0.0, 0.0],
            value: -10.0,
        };
        assert!(!annotation.visible_on_image(AnnotationLocation { plane: 0, slice: 0 }));
        assert!(!annotation.visible_on_image(AnnotationLocation {
            plane: 2,
            slice: 17
        }));
    }
}
