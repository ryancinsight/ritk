//! Responsive native pane composition.

use anyhow::{anyhow, bail, Result};
use metis_platform::{Color, Framebuffer};

use super::super::frame::RenderedView;
use super::super::projection::RenderedProjection;
use super::composition::{
    append_overlay_list, application_overlay, blit_frame, blit_rgba_frame, projection_overlay,
};
use super::geometry::{
    placement_geometry, placement_with_bounds, NativeViewport, ScreenRect, VIEW_GAP_PIXELS,
};
use crate::presentation::{PaneLayout, PaneRole};
use crate::tools::interaction::ViewportOffset;

/// Compose a responsive single, dual or four-pane surface.
pub(crate) fn surface_frames_responsive(
    views: &[RenderedView; 3],
    projection: Option<&RenderedProjection>,
    layout: PaneLayout,
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: ViewportOffset,
    cine_enabled: bool,
    cine_fps: f32,
    show_application_overlay: bool,
) -> Result<(Framebuffer, [NativeViewport; 3])> {
    if !zoom.is_finite() || zoom <= 0.0 {
        bail!("native viewer zoom must be finite and positive");
    }
    let panes = layout.partition(surface_width, surface_height, VIEW_GAP_PIXELS)?;
    let mut framebuffer = Framebuffer::new(surface_width, surface_height)
        .map_err(|error| anyhow!("allocate responsive native framebuffer: {error}"))?;
    framebuffer.clear(Color::BLACK);
    let mut viewports = [
        NativeViewport::hidden(&views[0])?,
        NativeViewport::hidden(&views[1])?,
        NativeViewport::hidden(&views[2])?,
    ];
    let mut projection_panel = None;
    for (slot, role) in layout.roles().iter().copied().enumerate() {
        let pane = panes[slot].ok_or_else(|| anyhow!("responsive layout omitted pane {slot}"))?;
        match role {
            PaneRole::Axis(axis) => {
                let view = views.get(axis).ok_or_else(|| {
                    anyhow!("responsive axis {axis} is outside the orthogonal views")
                })?;
                let viewport = placement_with_bounds(
                    view,
                    pane.x,
                    pane.y,
                    pane.width,
                    pane.height,
                    zoom,
                    pan_offset,
                )?;
                blit_frame(
                    &mut framebuffer,
                    view,
                    viewport,
                    surface_width,
                    surface_height,
                )?;
                viewports[axis] = viewport;
            }
            PaneRole::Projection => {
                let projection = projection
                    .ok_or_else(|| anyhow!("responsive quad layout requires a projection"))?;
                let panel = ScreenRect {
                    x: f64::from(pane.x),
                    y: f64::from(pane.y),
                    width: f64::from(pane.width),
                    height: f64::from(pane.height),
                };
                let image = placement_geometry(
                    [projection.frame.width(), projection.frame.height()],
                    projection.frame.display_spacing(),
                    pane.x,
                    pane.y,
                    pane.width,
                    pane.height,
                    zoom,
                    pan_offset,
                )?;
                blit_rgba_frame(
                    &mut framebuffer,
                    &projection.frame,
                    image,
                    panel,
                    surface_width,
                    surface_height,
                )?;
                projection_panel = Some((projection, pane));
            }
        }
    }
    if show_application_overlay {
        let mut overlay = application_overlay(views, &viewports, cine_enabled, cine_fps)?;
        if let Some((projection, pane)) = projection_panel {
            append_overlay_list(
                &mut overlay,
                projection_overlay(projection, pane.x, pane.y, pane.width, pane.height)?,
            )?;
        }
        overlay.render_to(&mut framebuffer);
    }
    Ok((framebuffer, viewports))
}
