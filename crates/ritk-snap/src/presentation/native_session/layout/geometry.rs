//! Screen geometry and coordinate mapping for native presentation.

use crate::app::viewer_viewport::ViewerViewport;
use crate::presentation::PresentationSpacing;
use crate::tools::interaction::ViewportOffset;
use anyhow::{anyhow, bail, Result};
#[cfg(test)]
use metis_platform::Rect;

use super::super::frame::RenderedView;

/// Pixel separator between native viewer panels.
pub(crate) const VIEW_GAP_PIXELS: u32 = 4;

/// Bounded client-area region reserved for medical-image panes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::presentation::native_session) struct ViewportArea {
    pub(in crate::presentation::native_session) x: u32,
    pub(in crate::presentation::native_session) y: u32,
    pub(in crate::presentation::native_session) width: u32,
    pub(in crate::presentation::native_session) height: u32,
}

impl ViewportArea {
    pub(in crate::presentation::native_session) const fn full(width: u32, height: u32) -> Self {
        Self {
            x: 0,
            y: 0,
            width,
            height,
        }
    }
}

/// Screen placement and RITK coordinate mapping for one composed view.
#[derive(Debug, Clone, Copy)]
pub(crate) struct NativeViewport {
    pub(super) panel: ScreenRect,
    pub(in crate::presentation::native_session) panel_x: u32,
    pub(in crate::presentation::native_session) panel_y: u32,
    pub(in crate::presentation::native_session) panel_width: u32,
    pub(in crate::presentation::native_session) panel_height: u32,
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

    pub(super) fn hidden(view: &RenderedView) -> Result<Self> {
        let mut viewport =
            placement_with_bounds(view, 0, 0, 1, 1, 1.0, ViewportOffset::new(0.0, 0.0))?;
        viewport.panel = ScreenRect {
            x: 0.0,
            y: 0.0,
            width: 0.0,
            height: 0.0,
        };
        viewport.panel_x = 0;
        viewport.panel_y = 0;
        viewport.panel_width = 0;
        viewport.panel_height = 0;
        Ok(viewport)
    }

    #[cfg(test)]
    pub(crate) const fn panel_width(self) -> u32 {
        self.panel_width
    }

    #[cfg(test)]
    pub(crate) fn panel_rect(self) -> Result<Rect> {
        Ok(Rect::new(
            i32::try_from(self.panel_x).map_err(|_| anyhow!("panel x exceeds i32"))?,
            i32::try_from(self.panel_y).map_err(|_| anyhow!("panel y exceeds i32"))?,
            i32::try_from(self.panel_width).map_err(|_| anyhow!("panel width exceeds i32"))?,
            i32::try_from(self.panel_height).map_err(|_| anyhow!("panel height exceeds i32"))?,
        ))
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

pub(super) fn placement_with_bounds(
    view: &RenderedView,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
    zoom: f32,
    pan_offset: ViewportOffset,
) -> Result<NativeViewport> {
    let image = placement_geometry(
        [view.frame.width(), view.frame.height()],
        view.frame.display_spacing(),
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
    display_spacing: PresentationSpacing,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
    zoom: f32,
    pan_offset: ViewportOffset,
) -> Result<ScreenRect> {
    let frame_width = f64::from(frame_size[0]);
    let frame_height = f64::from(frame_size[1]);
    let [row_spacing, column_spacing] = display_spacing.values();
    let reference = row_spacing.max(column_spacing);
    let relative_x = column_spacing / reference;
    let relative_y = row_spacing / reference;
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
        + f64::from(pan_offset.x());
    let origin_y = f64::from(panel_y)
        + (f64::from(panel_height) - rendered_height) * 0.5
        + f64::from(pan_offset.y());
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

/// Project a continuous source-image point to device coordinates in `viewport`.
///
/// Uses the transform that places the pane's pixels, so overlays stay on the
/// anatomy under rotation, flip, zoom and pan.
pub(super) fn source_point_to_screen(
    view: &RenderedView,
    viewport: &NativeViewport,
    source: [f64; 2],
) -> Result<Option<[f64; 2]>> {
    let output_size = view.transform.output_size(view.source_size);
    if output_size.contains(&0) {
        return Ok(None);
    }
    let [output_x, output_y] = view
        .transform
        .source_to_output_coordinates(source, view.source_size);
    let output_width = f64::from(u32::try_from(output_size[0])?);
    let output_height = f64::from(u32::try_from(output_size[1])?);
    let x = viewport.image.x + output_x / output_width * viewport.image.width;
    let y = viewport.image.y + output_y / output_height * viewport.image.height;
    Ok((x.is_finite() && y.is_finite()).then_some([x, y]))
}

/// The visible part of the image rectangle as `[left, top, right, bottom]`.
pub(super) fn visible_image_bounds(viewport: &NativeViewport) -> Option<[f64; 4]> {
    let left = viewport.image.x.max(viewport.panel.x);
    let right =
        (viewport.image.x + viewport.image.width).min(viewport.panel.x + viewport.panel.width);
    let top = viewport.image.y.max(viewport.panel.y);
    let bottom =
        (viewport.image.y + viewport.image.height).min(viewport.panel.y + viewport.panel.height);
    (left < right && top < bottom).then_some([left, top, right, bottom])
}

/// Round a finite device coordinate to the host's integer range.
pub(super) fn screen_coordinate(value: f64, label: &str) -> Result<i32> {
    if !value.is_finite() || value < f64::from(i32::MIN) || value > f64::from(i32::MAX) {
        bail!("{label} coordinate is outside the native display range")
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "finite display coordinates are checked against the i32 host contract"
    )]
    Ok(value.round() as i32)
}

#[cfg(test)]
mod tests;
