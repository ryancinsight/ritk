//! Length and angle overlays for the native viewer framebuffer.
//!
//! Annotation points use the interaction model's source-image convention:
//! `[row, column]` in pixel-edge coordinates. Projection applies the same RITK
//! [`ViewTransform`](crate::ui::ViewTransform) used to build the displayed
//! frame, then the compositor's spacing-aware fit, zoom and pan equation.
//! Stored length values are millimetres and stored angles are degrees.

use super::super::frame::RenderedView;
use super::super::layout::{NativeViewport, text_style};
use crate::app::SnapApp;
use crate::tools::interaction::{Annotation, ImagePoint, ToolState, ViewportOffset};
use anyhow::{Result, anyhow, bail};
use metis_platform::{Color, Framebuffer, Rect};
use metis_ui_lang::{DisplayCommand, DisplayList};

const MEASUREMENT_COLOR: Color = Color::rgba(255, 230, 0, 255);
const LABEL_COLOR: Color = Color::rgba(255, 255, 255, 255);
const LABEL_SIZE: u32 = 12;
const HANDLE_RADIUS: f64 = 3.0;
const LENGTH_LABEL_OFFSET: f64 = 10.0;
const ANGLE_LABEL_OFFSET: f64 = 22.0;

#[derive(Clone, Copy)]
struct PanelBounds {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl PanelBounds {
    const fn from_viewport(viewport: NativeViewport) -> Self {
        Self {
            x: viewport.panel_x,
            y: viewport.panel_y,
            width: viewport.panel_width,
            height: viewport.panel_height,
        }
    }

    fn clip(self) -> Option<ClipRect> {
        if self.width == 0 || self.height == 0 {
            return None;
        }
        Some(ClipRect {
            left: f64::from(self.x),
            top: f64::from(self.y),
            right: f64::from(self.x.saturating_add(self.width).saturating_sub(1)),
            bottom: f64::from(self.y.saturating_add(self.height).saturating_sub(1)),
        })
    }
}

#[derive(Clone, Copy)]
struct ClipRect {
    left: f64,
    top: f64,
    right: f64,
    bottom: f64,
}

impl ClipRect {
    fn contains(self, point: [f64; 2]) -> bool {
        point[0] >= self.left
            && point[0] <= self.right
            && point[1] >= self.top
            && point[1] <= self.bottom
    }
}

struct ImageToPanel {
    transform: crate::ui::ViewTransform,
    source_size: [usize; 2],
    output_size: [f64; 2],
    origin: [f64; 2],
    texel: [f64; 2],
    clip: ClipRect,
}

impl ImageToPanel {
    fn from_view(
        view: &RenderedView,
        viewport: NativeViewport,
        zoom: f32,
        pan: ViewportOffset,
    ) -> Result<Option<Self>> {
        let bounds = PanelBounds::from_viewport(viewport);
        let Some(panel_clip) = bounds.clip() else {
            return Ok(None);
        };
        let [row_spacing, column_spacing] = view.frame.display_spacing().values();
        let output_size = [
            f64::from(view.frame.width()),
            f64::from(view.frame.height()),
        ];
        let reference = row_spacing.max(column_spacing);
        let relative = [column_spacing / reference, row_spacing / reference];
        let physical = [output_size[0] * relative[0], output_size[1] * relative[1]];
        if physical
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            bail!("native measurement geometry has invalid physical extent");
        }
        let fit =
            (f64::from(bounds.width) / physical[0]).min(f64::from(bounds.height) / physical[1]);
        let zoom = f64::from(zoom);
        let texel = [relative[0] * fit * zoom, relative[1] * fit * zoom];
        let rendered = [output_size[0] * texel[0], output_size[1] * texel[1]];
        let origin = [
            f64::from(bounds.x)
                + (f64::from(bounds.width) - rendered[0]) * 0.5
                + f64::from(pan.x()),
            f64::from(bounds.y)
                + (f64::from(bounds.height) - rendered[1]) * 0.5
                + f64::from(pan.y()),
        ];
        if texel
            .iter()
            .chain(origin.iter())
            .any(|value| !value.is_finite())
            || texel.iter().any(|value| *value <= 0.0)
        {
            bail!("native measurement projection is outside the finite positive range");
        }
        let clip = ClipRect {
            left: origin[0].max(panel_clip.left),
            top: origin[1].max(panel_clip.top),
            right: (origin[0] + rendered[0] - 1.0).min(panel_clip.right),
            bottom: (origin[1] + rendered[1] - 1.0).min(panel_clip.bottom),
        };
        if clip.left > clip.right || clip.top > clip.bottom {
            return Ok(None);
        }
        Ok(Some(Self {
            transform: view.transform,
            source_size: view.source_size,
            output_size,
            origin,
            texel,
            clip,
        }))
    }

    fn project(&self, point: [f32; 2]) -> Option<[f64; 2]> {
        if point.iter().any(|coordinate| !coordinate.is_finite()) {
            return None;
        }
        let output = self.transform.source_to_output_coordinates(
            [f64::from(point[1]), f64::from(point[0])],
            self.source_size,
        );
        if output[0] < 0.0
            || output[0] > self.output_size[0]
            || output[1] < 0.0
            || output[1] > self.output_size[1]
        {
            return None;
        }
        Some([
            output[0].mul_add(self.texel[0], self.origin[0]),
            output[1].mul_add(self.texel[1], self.origin[1]),
        ])
    }
}

pub(super) fn render_measurements(
    framebuffer: &mut Framebuffer,
    app: &SnapApp,
    view: &RenderedView,
    viewport: NativeViewport,
) -> Result<()> {
    let overlay = measurement_overlay(app, view, viewport)?;
    if viewport.panel_width == 0 || viewport.panel_height == 0 {
        return Ok(());
    }
    let panel = Rect::new(
        i32::try_from(viewport.panel_x).map_err(|_| anyhow!("measurement panel x exceeds i32"))?,
        i32::try_from(viewport.panel_y).map_err(|_| anyhow!("measurement panel y exceeds i32"))?,
        i32::try_from(viewport.panel_width)
            .map_err(|_| anyhow!("measurement panel width exceeds i32"))?,
        i32::try_from(viewport.panel_height)
            .map_err(|_| anyhow!("measurement panel height exceeds i32"))?,
    );
    framebuffer.render_clipped(panel, |surface| overlay.render_to(surface));
    Ok(())
}

fn measurement_overlay(
    app: &SnapApp,
    view: &RenderedView,
    viewport: NativeViewport,
) -> Result<DisplayList> {
    if app.axis != view.axis {
        return Ok(DisplayList::default());
    }
    let Some(projection) = ImageToPanel::from_view(view, viewport, app.zoom, app.pan_offset)?
    else {
        return Ok(DisplayList::default());
    };
    overlay_for(&app.annotations, &app.tool_state, &projection)
}

fn overlay_for(
    annotations: &[Annotation],
    tool_state: &ToolState,
    projection: &ImageToPanel,
) -> Result<DisplayList> {
    let mut overlay = DisplayList::default();
    for annotation in annotations {
        match annotation {
            Annotation::Length { p1, p2, length_mm } => {
                let (Some(first), Some(second)) =
                    (projection.project(*p1), projection.project(*p2))
                else {
                    continue;
                };
                push_segment(&mut overlay, projection.clip, first, second)?;
                push_handle(&mut overlay, projection.clip, first)?;
                push_handle(&mut overlay, projection.clip, second)?;
                if length_mm.is_finite() {
                    let midpoint = midpoint(first, second);
                    let offset = perpendicular(first, second, LENGTH_LABEL_OFFSET);
                    push_label(
                        &mut overlay,
                        projection.clip,
                        [midpoint[0] + offset[0], midpoint[1] + offset[1]],
                        &format!("{length_mm:.1} mm"),
                    )?;
                }
            }
            Annotation::Angle {
                p1,
                p2,
                p3,
                angle_deg,
            } => {
                let (Some(first), Some(vertex), Some(third)) = (
                    projection.project(*p1),
                    projection.project(*p2),
                    projection.project(*p3),
                ) else {
                    continue;
                };
                push_segment(&mut overlay, projection.clip, vertex, first)?;
                push_segment(&mut overlay, projection.clip, vertex, third)?;
                for point in [first, vertex, third] {
                    push_handle(&mut overlay, projection.clip, point)?;
                }
                if angle_deg.is_finite() {
                    let direction = angle_bisector(first, vertex, third);
                    push_label(
                        &mut overlay,
                        projection.clip,
                        [
                            vertex[0] + direction[0] * ANGLE_LABEL_OFFSET,
                            vertex[1] + direction[1] * ANGLE_LABEL_OFFSET,
                        ],
                        &format!("{angle_deg:.1}°"),
                    )?;
                }
            }
            Annotation::PatientLength(_)
            | Annotation::RoiRect { .. }
            | Annotation::RoiEllipse { .. }
            | Annotation::HuPoint { .. } => {}
        }
    }

    match tool_state {
        ToolState::MeasureLength1 { p1 } => {
            if let Some(point) = project_image_point(projection, *p1) {
                push_handle(&mut overlay, projection.clip, point)?;
            }
        }
        ToolState::MeasureAngle2 { p1, p2 } => {
            if let (Some(first), Some(vertex)) = (
                project_image_point(projection, *p1),
                project_image_point(projection, *p2),
            ) {
                push_segment(&mut overlay, projection.clip, first, vertex)?;
                push_handle(&mut overlay, projection.clip, first)?;
                push_handle(&mut overlay, projection.clip, vertex)?;
            }
        }
        ToolState::Idle
        | ToolState::Panning { .. }
        | ToolState::Zooming { .. }
        | ToolState::WindowLevelDrag { .. }
        | ToolState::RoiDrag { .. } => {}
    }
    Ok(overlay)
}

fn project_image_point(projection: &ImageToPanel, point: ImagePoint) -> Option<[f64; 2]> {
    projection.project([point.y(), point.x()])
}

fn push_segment(
    overlay: &mut DisplayList,
    clip: ClipRect,
    start: [f64; 2],
    end: [f64; 2],
) -> Result<()> {
    let Some((start, end)) = clip_segment(clip, start, end) else {
        return Ok(());
    };
    overlay.append_line(screen_point(start)?, screen_point(end)?, MEASUREMENT_COLOR)?;
    Ok(())
}

fn push_handle(overlay: &mut DisplayList, clip: ClipRect, point: [f64; 2]) -> Result<()> {
    if !clip.contains(point) {
        return Ok(());
    }
    push_segment(
        overlay,
        clip,
        [point[0] - HANDLE_RADIUS, point[1]],
        [point[0] + HANDLE_RADIUS, point[1]],
    )?;
    push_segment(
        overlay,
        clip,
        [point[0], point[1] - HANDLE_RADIUS],
        [point[0], point[1] + HANDLE_RADIUS],
    )
}

fn push_label(
    overlay: &mut DisplayList,
    clip: ClipRect,
    center: [f64; 2],
    label: &str,
) -> Result<()> {
    let style = text_style(LABEL_COLOR, LABEL_SIZE)?;
    let width = style.advance(label);
    let height = style.line_height();
    let x = (center[0] - width * 0.5).clamp(clip.left, (clip.right - width).max(clip.left));
    let y = (center[1] - height * 0.5).clamp(clip.top, (clip.bottom - height).max(clip.top));
    overlay
        .commands
        .try_reserve(1)
        .map_err(|_| anyhow!("native measurement command allocation failed"))?;
    overlay.commands.push(DisplayCommand::DrawText {
        text: label.to_owned(),
        x: screen_coordinate(x, "measurement label x")?,
        y: screen_coordinate(y, "measurement label y")?,
        style,
    });
    Ok(())
}

fn midpoint(first: [f64; 2], second: [f64; 2]) -> [f64; 2] {
    [(first[0] + second[0]) * 0.5, (first[1] + second[1]) * 0.5]
}

fn perpendicular(first: [f64; 2], second: [f64; 2], distance: f64) -> [f64; 2] {
    let delta = [second[0] - first[0], second[1] - first[1]];
    let length = delta[0].hypot(delta[1]);
    if length <= f64::EPSILON {
        [0.0, -distance]
    } else {
        [-delta[1] / length * distance, delta[0] / length * distance]
    }
}

fn angle_bisector(first: [f64; 2], vertex: [f64; 2], third: [f64; 2]) -> [f64; 2] {
    let first_direction = unit([first[0] - vertex[0], first[1] - vertex[1]]);
    let third_direction = unit([third[0] - vertex[0], third[1] - vertex[1]]);
    let sum = [
        first_direction[0] + third_direction[0],
        first_direction[1] + third_direction[1],
    ];
    if sum[0].hypot(sum[1]) <= f64::EPSILON {
        [-first_direction[1], first_direction[0]]
    } else {
        unit(sum)
    }
}

fn unit(vector: [f64; 2]) -> [f64; 2] {
    let length = vector[0].hypot(vector[1]);
    if length <= f64::EPSILON {
        [0.0, 0.0]
    } else {
        [vector[0] / length, vector[1] / length]
    }
}

fn clip_segment(clip: ClipRect, start: [f64; 2], end: [f64; 2]) -> Option<([f64; 2], [f64; 2])> {
    let delta = [end[0] - start[0], end[1] - start[1]];
    let mut lower = 0.0_f64;
    let mut upper = 1.0_f64;
    for (direction, distance) in [
        (-delta[0], start[0] - clip.left),
        (delta[0], clip.right - start[0]),
        (-delta[1], start[1] - clip.top),
        (delta[1], clip.bottom - start[1]),
    ] {
        if direction.abs() <= f64::EPSILON {
            if distance < 0.0 {
                return None;
            }
            continue;
        }
        let ratio = distance / direction;
        if direction < 0.0 {
            lower = lower.max(ratio);
        } else {
            upper = upper.min(ratio);
        }
        if lower > upper {
            return None;
        }
    }
    Some((
        [
            delta[0].mul_add(lower, start[0]),
            delta[1].mul_add(lower, start[1]),
        ],
        [
            delta[0].mul_add(upper, start[0]),
            delta[1].mul_add(upper, start[1]),
        ],
    ))
}

fn screen_point(point: [f64; 2]) -> Result<(i32, i32)> {
    Ok((
        screen_coordinate(point[0], "measurement x")?,
        screen_coordinate(point[1], "measurement y")?,
    ))
}

fn screen_coordinate(value: f64, label: &str) -> Result<i32> {
    if !value.is_finite() || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
        bail!("{label} coordinate is outside the native display range");
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "finite display coordinates are checked against the i32 host contract"
    )]
    Ok(value.round() as i32)
}

#[cfg(test)]
mod tests;
