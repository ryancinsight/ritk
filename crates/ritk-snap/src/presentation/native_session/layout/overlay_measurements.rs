//! RITK-owned measurement and region overlays for native panes.

use anyhow::Result;
use metis_platform::Color;

use super::super::frame::RenderedView;
use super::crosshair::screen_coordinate;
use super::geometry::NativeViewport;
use super::text::{text_command, text_style, PANEL_DETAIL_SIZE};
use super::OverlayCommands;
use crate::geometry::affine::AffineTransform;
use crate::geometry::PatientPointMm;
use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint, ToolState};

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
        let output_size = view.transform.output_size(view.source_size);
        if output_size.contains(&0) {
            continue;
        }
        let pane = Pane {
            view,
            viewport,
            output_width: f64::from(u32::try_from(output_size[0])?),
            output_height: f64::from(u32::try_from(output_size[1])?),
        };
        for annotation in annotations {
            if !annotation_visible(annotation, view.axis, view.slice_index) {
                continue;
            }
            pane.draw(annotation, overlay, affine)?;
        }
    }
    Ok(())
}

pub(crate) fn append_measurement_preview(
    overlay: &mut OverlayCommands,
    view: &RenderedView,
    viewport: &NativeViewport,
    tool_state: &ToolState,
    location: Option<AnnotationLocation>,
) -> Result<()> {
    if location.is_some_and(|source| {
        source
            != AnnotationLocation {
                plane: view.axis,
                slice: view.slice_index,
            }
    }) {
        return Ok(());
    }
    let output_size = view.transform.output_size(view.source_size);
    if output_size.contains(&0) {
        return Ok(());
    }
    let pane = Pane {
        view,
        viewport,
        output_width: f64::from(u32::try_from(output_size[0])?),
        output_height: f64::from(u32::try_from(output_size[1])?),
    };
    match tool_state {
        ToolState::CobbDrag {
            first: Some(first),
            start,
            current,
        } => {
            pane.draw_preview_segment(overlay, *first)?;
            pane.draw_preview_segment(overlay, [*start, *current])?;
        }
        ToolState::CobbDrag { start, current, .. } => {
            pane.draw_preview_segment(overlay, [*start, *current])?;
        }
        ToolState::CobbPending { segment } => {
            pane.draw_preview_segment(overlay, *segment)?;
        }
        _ => {}
    }
    Ok(())
}

fn annotation_visible(annotation: &Annotation, plane: usize, slice: usize) -> bool {
    if let Some(AnnotationLocation {
        plane: recorded,
        slice: index,
    }) = annotation.location()
    {
        return recorded == plane && index == slice;
    }
    matches!(
        annotation,
        Annotation::Length { plane: recorded, .. }
            | Annotation::PatientLength { plane: recorded, .. } if *recorded == plane
    )
}

struct Pane<'a> {
    view: &'a RenderedView,
    viewport: &'a NativeViewport,
    output_width: f64,
    output_height: f64,
}

impl Pane<'_> {
    fn draw_preview_segment(
        &self,
        overlay: &mut OverlayCommands,
        segment: [ImagePoint; 2],
    ) -> Result<()> {
        self.line(
            overlay,
            [f64::from(segment[0].x()), f64::from(segment[0].y())],
            [f64::from(segment[1].x()), f64::from(segment[1].y())],
            MEASUREMENT_COLOR,
        )
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

    fn label(&self, overlay: &mut OverlayCommands, anchor: [f64; 2], text: String) -> Result<()> {
        let (x, y) = self.point(anchor)?;
        overlay.push(text_command(
            text,
            x.saturating_add(LABEL_OFFSET),
            y.saturating_add(LABEL_OFFSET),
            text_style(MEASUREMENT_COLOR, PANEL_DETAIL_SIZE)?,
        ));
        Ok(())
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
        overlay.push(text_command(
            text,
            screen_coordinate((x0 + x1) * 0.5 + ox, "length label x")?,
            screen_coordinate((y0 + y1) * 0.5 + oy, "length label y")?,
            text_style(MEASUREMENT_COLOR, PANEL_DETAIL_SIZE)?,
        ));
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
                let baseline = image_point(*reference);
                self.line(overlay, a, b, MEASUREMENT_COLOR)?;
                self.line(overlay, a, baseline, MEASUREMENT_COLOR)?;
                self.label(
                    overlay,
                    b,
                    format!("{distance_mm:.1} mm {angle_deg:.1} deg"),
                )?;
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
                    self.label(
                        overlay,
                        *first,
                        format!("{area_mm2:.1} mm2  {perimeter_mm:.1} mm"),
                    )?;
                }
            }
            Annotation::OpenPolygon {
                points, length_mm, ..
            } => {
                let polyline: Vec<[f64; 2]> = points.iter().map(|p| image_point(*p)).collect();
                for pair in polyline.windows(2) {
                    self.line(overlay, pair[0], pair[1], ROI_COLOR)?;
                }
                if let Some(first) = polyline.first() {
                    self.label(overlay, *first, format!("{length_mm:.1} mm"))?;
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
    use super::{annotation_visible, Annotation, AnnotationLocation};

    #[test]
    fn anchored_annotation_requires_matching_plane_and_slice() {
        let annotation = Annotation::HuPoint {
            location: Some(AnnotationLocation { plane: 1, slice: 4 }),
            pos: [0.0, 0.0],
            value: -10.0,
        };

        assert!(annotation_visible(&annotation, 1, 4));
        assert!(!annotation_visible(&annotation, 0, 4));
        assert!(!annotation_visible(&annotation, 1, 3));
    }

    #[test]
    fn legacy_point_without_image_location_does_not_claim_an_image() {
        let annotation = Annotation::HuPoint {
            location: None,
            pos: [0.0, 0.0],
            value: -10.0,
        };
        assert!(!annotation_visible(&annotation, 0, 0));
        assert!(!annotation_visible(&annotation, 2, 17));
    }
}
