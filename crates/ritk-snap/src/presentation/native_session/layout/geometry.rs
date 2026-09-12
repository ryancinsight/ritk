//! Screen geometry and coordinate mapping for native presentation.

use crate::app::action_adapter::ViewerViewport;
use anyhow::{anyhow, bail, Result};

use super::super::frame::RenderedView;

/// Pixel separator between the three native orthogonal panels.
pub(super) const VIEW_GAP_PIXELS: u32 = 4;

/// Screen placement and RITK coordinate mapping for one composed view.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NativeViewport {
    pub(super) panel: ScreenRect,
    pub(super) panel_x: u32,
    pub(super) panel_y: u32,
    pub(super) panel_width: u32,
    pub(super) panel_height: u32,
    pub(super) image: ScreenRect,
    pub(super) mapping: ViewerViewport,
}

impl NativeViewport {
    pub(crate) const fn axis(self) -> usize {
        self.mapping.axis()
    }

    pub(crate) const fn mapping(self) -> ViewerViewport {
        self.mapping
    }

    pub(crate) fn contains(self, x: f64, y: f64) -> bool {
        self.panel.contains(x, y)
    }

    #[cfg(test)]
    pub(crate) const fn panel_width(self) -> u32 {
        self.panel_width
    }

    #[cfg(test)]
    pub(crate) fn center(self) -> (i32, i32) {
        let x = (self.image.x + self.image.width * 0.5).round();
        let y = (self.image.y + self.image.height * 0.5).round();
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test coordinates are bounded by the native surface"
        )]
        let x = x as i32;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "test coordinates are bounded by the native surface"
        )]
        let y = y as i32;
        (x, y)
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ScreenRect {
    pub(super) x: f64,
    pub(super) y: f64,
    pub(super) width: f64,
    pub(super) height: f64,
}

impl ScreenRect {
    pub(super) fn contains(self, x: f64, y: f64) -> bool {
        x.is_finite()
            && y.is_finite()
            && x >= self.x
            && y >= self.y
            && x < self.x + self.width
            && y < self.y + self.height
    }
}

pub(super) fn placement(
    view: &RenderedView,
    panel_x: u32,
    panel_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: egui::Vec2,
) -> Result<NativeViewport> {
    placement_with_bounds(
        view,
        panel_x,
        0,
        panel_width,
        surface_height,
        zoom,
        pan_offset,
    )
}

pub(super) fn placement_with_bounds(
    view: &RenderedView,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
    zoom: f32,
    pan_offset: egui::Vec2,
) -> Result<NativeViewport> {
    let image = placement_geometry(
        [view.frame.width(), view.frame.height()],
        view.display_spacing,
        panel_x,
        panel_y,
        panel_width,
        panel_height,
        zoom,
        pan_offset,
    )?;
    let mapping = ViewerViewport::new(
        view.axis,
        [image.x, image.y],
        [
            image.width / f64::from(view.frame.width()),
            image.height / f64::from(view.frame.height()),
        ],
        view.source_size,
        view.transform,
    )
    .map_err(|error| anyhow!("construct native viewer viewport: {error}"))?;
    Ok(NativeViewport {
        panel: ScreenRect {
            x: f64::from(panel_x),
            y: f64::from(panel_y),
            width: f64::from(panel_width),
            height: f64::from(panel_height),
        },
        panel_x,
        panel_y,
        panel_width,
        panel_height,
        image,
        mapping,
    })
}

pub(super) fn placement_geometry(
    frame_size: [u32; 2],
    display_spacing: [f64; 2],
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
    zoom: f32,
    pan_offset: egui::Vec2,
) -> Result<ScreenRect> {
    let frame_width = f64::from(frame_size[0]);
    let frame_height = f64::from(frame_size[1]);
    let reference = display_spacing[0].max(display_spacing[1]);
    let relative_x = display_spacing[1] / reference;
    let relative_y = display_spacing[0] / reference;
    let physical_width = frame_width * relative_x;
    let physical_height = frame_height * relative_y;
    if !physical_width.is_finite()
        || !physical_height.is_finite()
        || physical_width <= 0.0
        || physical_height <= 0.0
    {
        bail!("native view physical geometry is outside the finite positive range");
    }
    let fit =
        (f64::from(panel_width) / physical_width).min(f64::from(panel_height) / physical_height);
    let zoom = f64::from(zoom);
    let texel_x = relative_x * fit * zoom;
    let texel_y = relative_y * fit * zoom;
    let rendered_width = frame_width * texel_x;
    let rendered_height = frame_height * texel_y;
    let origin_x = f64::from(panel_x)
        + (f64::from(panel_width) - rendered_width) * 0.5
        + f64::from(pan_offset.x);
    let origin_y = f64::from(panel_y)
        + (f64::from(panel_height) - rendered_height) * 0.5
        + f64::from(pan_offset.y);
    if ![
        texel_x,
        texel_y,
        rendered_width,
        rendered_height,
        origin_x,
        origin_y,
    ]
    .iter()
    .all(|value| value.is_finite() && *value > f64::from(f32::MIN))
        || texel_x > f64::from(f32::MAX)
        || texel_y > f64::from(f32::MAX)
        || origin_x > f64::from(f32::MAX)
        || origin_y > f64::from(f32::MAX)
    {
        bail!("native viewer geometry exceeds finite host range");
    }
    Ok(ScreenRect {
        x: origin_x,
        y: origin_y,
        width: rendered_width,
        height: rendered_height,
    })
}
