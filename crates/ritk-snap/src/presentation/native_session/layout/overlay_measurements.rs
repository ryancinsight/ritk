//! RITK-owned measurement and region overlays for native panes.

mod cobb;
mod handles;
mod labels;

use super::super::frame::RenderedView;
use super::crosshair::screen_coordinate;
use super::geometry::NativeViewport;
use super::text::{text_style, PANEL_DETAIL_SIZE};
use super::OverlayCommands;
use crate::geometry::affine::AffineTransform;
use crate::geometry::PatientPointMm;
use crate::tools::interaction::{Annotation, AnnotationLocation, ImagePoint, ToolState};
use anyhow::{anyhow, Result};
use metis_platform::{Color, Rect};
use metis_ui_lang::PlacementObstacle;
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
    selected_annotation: Option<usize>,
    pending_annotation: Option<(usize, &Annotation)>,
    affine: Option<AffineTransform>,
    length_scale: impl Fn(Option<AnnotationLocation>) -> f64,
    preview: Option<(&ToolState, Option<AnnotationLocation>)>,
    frame: Rect,
    reserved: &[PlacementObstacle],
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
        let mut labels = Vec::new();
        labels
            .try_reserve(annotations.len())
            .map_err(|_| anyhow!("native measurement label allocation failed"))?;
        let pending_index = pending_annotation.map(|(pending_index, _)| pending_index);
        for (index, annotation) in annotations.iter().enumerate() {
            if !annotation_visible(annotation, view.axis, view.slice_index) {
                continue;
            }

            let annotation = pending_annotation
                .filter(|(pending_index, _)| *pending_index == index)
                .map_or(annotation, |(_, candidate)| candidate);
            pane.draw(
                annotation,
                overlay,
                &mut labels,
                affine,
                length_scale(annotation.location()),
            )?;
            if selected_annotation == Some(index) || pending_index == Some(index) {
                handles::append(&pane, annotation, overlay)?;
            }
        }
        if let Some((tool_state, location)) = preview {
            append_measurement_preview(overlay, view, viewport, tool_state, location)?;
        }
        for (text, origin) in labels {
            pane.draw_label(overlay, text, origin, frame, reserved)?;
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
    if !matches!(
        tool_state,
        ToolState::LengthDrag { .. }
            | ToolState::ArrowDrag { .. }
            | ToolState::PencilDrag { .. }
            | ToolState::CobbDrag { .. }
            | ToolState::CobbPending { .. }
    ) || location.is_some_and(|source| {
        source
            != (AnnotationLocation {
                plane: view.axis,
                slice: view.slice_index,
            })
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
    if let ToolState::LengthDrag { start, current } = tool_state {
        pane.segment(overlay, [*start, *current], MEASUREMENT_COLOR)?;
    } else if let ToolState::ArrowDrag { start, current } = tool_state {
        pane.arrow(overlay, [*start, *current], MEASUREMENT_COLOR)?;
    } else if let ToolState::PencilDrag { capture } = tool_state {
        for pair in capture.as_slice().windows(2) {
            pane.segment(overlay, [pair[0], pair[1]], MEASUREMENT_COLOR)?;
        }
    } else {
        pane.cobb_preview(overlay, tool_state)?;
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

    fn segment(
        &self,
        overlay: &mut OverlayCommands,
        points: [ImagePoint; 2],
        color: Color,
    ) -> Result<()> {
        self.line(
            overlay,
            image_point([points[0].y(), points[0].x()]),
            image_point([points[1].y(), points[1].x()]),
            color,
        )
    }

    fn arrow(
        &self,
        overlay: &mut OverlayCommands,
        points: [ImagePoint; 2],
        color: Color,
    ) -> Result<()> {
        let a = image_point([points[0].y(), points[0].x()]);
        let b = image_point([points[1].y(), points[1].x()]);
        let tail = self.point(a)?;
        let tip = self.point(b)?;
        overlay.push(metis_ui_lang::DisplayCommand::DrawLine {
            start: tail,
            end: tip,
            color,
        });
        if let Some(wings) = metis_ui_lang::arrow_wings(
            [f64::from(tail.0), f64::from(tail.1)],
            [f64::from(tip.0), f64::from(tip.1)],
            8.0,
            4.0,
        ) {
            for wing in wings {
                overlay.push(metis_ui_lang::DisplayCommand::DrawLine {
                    start: tip,
                    end: (
                        screen_coordinate(wing[0], "arrow wing x")?,
                        screen_coordinate(wing[1], "arrow wing y")?,
                    ),
                    color,
                });
            }
        }
        Ok(())
    }

    fn draw(
        &self,
        annotation: &Annotation,
        overlay: &mut OverlayCommands,
        labels: &mut Vec<(String, (i32, i32))>,
        affine: Option<AffineTransform>,
        length_scale: f64,
    ) -> Result<()> {
        match annotation {
            Annotation::Length {
                plane,
                p1,
                p2,
                length_mm,
                ..
            } if *plane == self.view.axis => {
                let segment = [ImagePoint::new(p1[1], p1[0]), ImagePoint::new(p2[1], p2[0])];
                self.segment(overlay, segment, MEASUREMENT_COLOR)?;
                self.length_label(
                    labels,
                    image_point(*p1),
                    image_point(*p2),
                    format!("{:.1} mm", f64::from(*length_mm) * length_scale),
                )?;
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
                self.length_label(labels, a, b, format!("{:.1} mm", length.length_mm()))?;
            }
            Annotation::Pencil { points, .. } => {
                points.windows(2).try_for_each(|pair| {
                    self.line(
                        overlay,
                        image_point(pair[0]),
                        image_point(pair[1]),
                        MEASUREMENT_COLOR,
                    )
                })?;
            }
            Annotation::Angle {
                p1,
                p2,
                p3,
                angle_deg,
                ..
            } => {
                let first = [ImagePoint::new(p1[1], p1[0]), ImagePoint::new(p2[1], p2[0])];
                let second = [ImagePoint::new(p2[1], p2[0]), ImagePoint::new(p3[1], p3[0])];
                self.segment(overlay, first, MEASUREMENT_COLOR)?;
                self.segment(overlay, second, MEASUREMENT_COLOR)?;
                self.label(labels, image_point(*p2), format!("{angle_deg:.1} deg"))?;
            }
            Annotation::Deviation {
                start,
                end,
                reference,
                distance_mm,
                angle_deg,
                ..
            } => {
                let first = [
                    ImagePoint::new(start[1], start[0]),
                    ImagePoint::new(end[1], end[0]),
                ];
                let baseline = [
                    ImagePoint::new(start[1], start[0]),
                    ImagePoint::new(reference[1], reference[0]),
                ];
                self.segment(overlay, first, MEASUREMENT_COLOR)?;
                self.segment(overlay, baseline, MEASUREMENT_COLOR)?;
                self.label(
                    labels,
                    image_point(*end),
                    format!("{distance_mm:.1} mm {angle_deg:.1} deg"),
                )?;
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
                    labels,
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
                    labels,
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
                self.label(labels, p, format!("{value:.0}"))?;
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
                        labels,
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
                    self.label(labels, *first, format!("{length_mm:.1} mm"))?;
                }
            }
            Annotation::Arrow { tail, head, .. } => {
                self.arrow(
                    overlay,
                    [
                        ImagePoint::new(tail[1], tail[0]),
                        ImagePoint::new(head[1], head[0]),
                    ],
                    MEASUREMENT_COLOR,
                )?;
            }
            Annotation::CobbAngle {
                points, angle_deg, ..
            } => self.cobb(overlay, labels, *points, *angle_deg)?,
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
mod tests;
