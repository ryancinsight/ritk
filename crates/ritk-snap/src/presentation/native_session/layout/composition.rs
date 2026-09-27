//! Pixel composition for native presentation layouts.

use anyhow::{anyhow, bail, Result};
use metis_platform::{Color, Framebuffer};

use super::super::frame::RenderedView;
use super::super::projection::RenderedProjection;
use super::geometry::{
    placement, placement_geometry, placement_with_bounds, NativeViewport, ScreenRect,
    VIEW_GAP_PIXELS,
};
use super::overlay::{append_overlay_list, application_overlay, projection_overlay};
use crate::presentation::PresentationFrame;
use crate::tools::interaction::ViewportOffset;

/// Compose the three views into one bounded Métis framebuffer.
pub(crate) fn surface_frames(
    views: &[RenderedView; 3],
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: ViewportOffset,
    cine_enabled: bool,
    cine_fps: f32,
    show_application_overlay: bool,
) -> Result<(Framebuffer, [NativeViewport; 3])> {
    if surface_width == 0 || surface_height == 0 {
        bail!("native surface dimensions must be nonzero while rendering");
    }
    if !zoom.is_finite() || zoom <= 0.0 {
        bail!("native viewer zoom must be finite and positive");
    }
    let gaps = VIEW_GAP_PIXELS
        .checked_mul(2)
        .ok_or_else(|| anyhow!("native view gap arithmetic overflows"))?;
    let available_width = surface_width
        .checked_sub(gaps)
        .ok_or_else(|| anyhow!("native surface is narrower than its view separators"))?;
    if available_width < 3 {
        bail!("native surface cannot allocate three orthogonal view panels");
    }
    let base_width = available_width / 3;
    let remainder = available_width % 3;
    let panel_widths = [
        base_width + u32::from(remainder > 0),
        base_width + u32::from(remainder > 1),
        base_width,
    ];
    let mut framebuffer = Framebuffer::new(surface_width, surface_height)
        .map_err(|error| anyhow!("allocate native viewer framebuffer: {error}"))?;
    framebuffer.clear(Color::BLACK);
    let viewports = [
        placement(
            &views[0],
            0,
            panel_widths[0],
            surface_height,
            zoom,
            pan_offset,
        )?,
        placement(
            &views[1],
            panel_widths[0] + VIEW_GAP_PIXELS,
            panel_widths[1],
            surface_height,
            zoom,
            pan_offset,
        )?,
        placement(
            &views[2],
            panel_widths[0] + VIEW_GAP_PIXELS + panel_widths[1] + VIEW_GAP_PIXELS,
            panel_widths[2],
            surface_height,
            zoom,
            pan_offset,
        )?,
    ];
    for (view, viewport) in views.iter().zip(viewports) {
        blit_frame(
            &mut framebuffer,
            view,
            viewport,
            surface_width,
            surface_height,
        )?;
    }
    if show_application_overlay {
        let overlay = application_overlay(views, &viewports, cine_enabled, cine_fps)?;
        overlay.render_to(&mut framebuffer);
    }
    Ok((framebuffer, viewports))
}

/// Compose orthogonal planes and a RITK scalar projection in a bounded 2×2 layout.
pub(crate) fn surface_frames_with_projection(
    views: &[RenderedView; 3],
    projection: &RenderedProjection,
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: ViewportOffset,
    cine_enabled: bool,
    cine_fps: f32,
    show_application_overlay: bool,
) -> Result<(Framebuffer, [NativeViewport; 3])> {
    let mut composition = compose_four_panel(
        views,
        &projection.frame,
        [surface_width, surface_height],
        PaneNavigation { zoom, pan_offset },
        PaneNavigation { zoom, pan_offset },
    )?;
    if show_application_overlay {
        let mut overlay =
            application_overlay(views, &composition.viewports, cine_enabled, cine_fps)?;
        append_overlay_list(
            &mut overlay,
            projection_overlay(
                projection,
                composition.fourth_panel.x,
                composition.fourth_panel.y,
                composition.fourth_panel.width,
                composition.fourth_panel.height,
            )?,
        )?;
        overlay.render_to(&mut composition.framebuffer);
    }
    Ok((composition.framebuffer, composition.viewports))
}

#[derive(Clone, Copy)]
struct PaneNavigation {
    zoom: f32,
    pan_offset: ViewportOffset,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PanelBounds {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

impl PanelBounds {
    fn screen_rect(self) -> ScreenRect {
        ScreenRect {
            x: f64::from(self.x),
            y: f64::from(self.y),
            width: f64::from(self.width),
            height: f64::from(self.height),
        }
    }
}

struct FourPanelComposition {
    framebuffer: Framebuffer,
    viewports: [NativeViewport; 3],
    fourth_panel: PanelBounds,
}

/// Place three orthogonal frames and a host-neutral fourth frame in one grid.
fn compose_four_panel(
    views: &[RenderedView; 3],
    fourth_frame: &PresentationFrame,
    surface_size: [u32; 2],
    orthogonal: PaneNavigation,
    fourth: PaneNavigation,
) -> Result<FourPanelComposition> {
    let [surface_width, surface_height] = surface_size;
    if surface_width == 0 || surface_height == 0 {
        bail!("native surface dimensions must be nonzero while rendering");
    }
    if !orthogonal.zoom.is_finite() || orthogonal.zoom <= 0.0 {
        bail!("native viewer zoom must be finite and positive");
    }
    if !fourth.zoom.is_finite() || fourth.zoom <= 0.0 {
        bail!("native fourth-panel zoom must be finite and positive");
    }
    let available_width = surface_width
        .checked_sub(VIEW_GAP_PIXELS)
        .ok_or_else(|| anyhow!("native four-panel layout is narrower than its separator"))?;
    let available_height = surface_height
        .checked_sub(VIEW_GAP_PIXELS)
        .ok_or_else(|| anyhow!("native four-panel layout is shorter than its separator"))?;
    if available_width < 2 || available_height < 2 {
        bail!("native four-panel layout cannot allocate four panels");
    }
    let column_widths = [
        available_width / 2 + available_width % 2,
        available_width / 2,
    ];
    let row_heights = [
        available_height / 2 + available_height % 2,
        available_height / 2,
    ];
    let viewports = [
        placement_with_bounds(
            &views[0],
            0,
            0,
            column_widths[0],
            row_heights[0],
            orthogonal.zoom,
            orthogonal.pan_offset,
        )?,
        placement_with_bounds(
            &views[1],
            column_widths[0] + VIEW_GAP_PIXELS,
            0,
            column_widths[1],
            row_heights[0],
            orthogonal.zoom,
            orthogonal.pan_offset,
        )?,
        placement_with_bounds(
            &views[2],
            0,
            row_heights[0] + VIEW_GAP_PIXELS,
            column_widths[0],
            row_heights[1],
            orthogonal.zoom,
            orthogonal.pan_offset,
        )?,
    ];
    let fourth_panel = PanelBounds {
        x: column_widths[0] + VIEW_GAP_PIXELS,
        y: row_heights[0] + VIEW_GAP_PIXELS,
        width: column_widths[1],
        height: row_heights[1],
    };
    let fourth_image = placement_geometry(
        [fourth_frame.width(), fourth_frame.height()],
        fourth_frame.display_spacing(),
        fourth_panel.x,
        fourth_panel.y,
        fourth_panel.width,
        fourth_panel.height,
        fourth.zoom,
        fourth.pan_offset,
    )?;
    let mut framebuffer = Framebuffer::new(surface_width, surface_height)
        .map_err(|error| anyhow!("allocate native viewer framebuffer: {error}"))?;
    framebuffer.clear(Color::BLACK);
    for (view, viewport) in views.iter().zip(viewports) {
        blit_frame(
            &mut framebuffer,
            view,
            viewport,
            surface_width,
            surface_height,
        )?;
    }
    blit_rgba_frame(
        &mut framebuffer,
        fourth_frame,
        fourth_image,
        fourth_panel.screen_rect(),
        surface_width,
        surface_height,
    )?;
    Ok(FourPanelComposition {
        framebuffer,
        viewports,
        fourth_panel,
    })
}

pub(super) fn blit_frame(
    framebuffer: &mut Framebuffer,
    view: &RenderedView,
    viewport: NativeViewport,
    surface_width: u32,
    surface_height: u32,
) -> Result<()> {
    blit_rgba_frame(
        framebuffer,
        &view.frame,
        viewport.image,
        viewport.panel,
        surface_width,
        surface_height,
    )
}

pub(super) fn blit_rgba_frame(
    framebuffer: &mut Framebuffer,
    frame: &crate::presentation::PresentationFrame,
    image: ScreenRect,
    panel: ScreenRect,
    surface_width: u32,
    surface_height: u32,
) -> Result<()> {
    let frame_width = f64::from(frame.width());
    let frame_height = f64::from(frame.height());
    let texel_x = image.width / frame_width;
    let texel_y = image.height / frame_height;
    let frame_width_usize = usize::try_from(frame.width())?;
    let frame_height_usize = usize::try_from(frame.height())?;
    for y in 0..surface_height {
        let screen_y = f64::from(y) + 0.5;
        // The x coordinate is checked in the inner loop; this y-only guard
        // avoids source calculations for rows outside a grid panel.
        if screen_y < panel.y || screen_y >= panel.y + panel.height {
            continue;
        }
        let source_y = ((screen_y - image.y) / texel_y).floor();
        if source_y < 0.0 || source_y >= frame_height {
            continue;
        }
        #[expect(
            clippy::cast_possible_truncation,
            reason = "source coordinate is checked against the bounded frame height"
        )]
        let source_y = source_y as usize;
        for x in 0..surface_width {
            let screen_x = f64::from(x) + 0.5;
            if !panel.contains(screen_x, screen_y) {
                continue;
            }
            let source_x = ((screen_x - image.x) / texel_x).floor();
            if source_x < 0.0 || source_x >= frame_width {
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "source coordinate is checked against the bounded frame width"
            )]
            let source_x = source_x as usize;
            let offset = source_y
                .checked_mul(frame_width_usize)
                .and_then(|row| row.checked_add(source_x))
                .and_then(|pixel| pixel.checked_mul(4))
                .ok_or_else(|| anyhow!("native viewer frame offset overflows"))?;
            let pixel = frame
                .rgba()
                .get(offset..offset + 4)
                .ok_or_else(|| anyhow!("native viewer frame storage is truncated"))?;
            let x = i32::try_from(x)?;
            let y = i32::try_from(y)?;
            framebuffer.set_pixel(x, y, Color::rgba(pixel[0], pixel[1], pixel[2], pixel[3]));
        }
    }
    debug_assert_eq!(
        frame_height_usize.checked_mul(frame_width_usize),
        Some(frame.rgba().len() / 4)
    );
    Ok(())
}

#[cfg(test)]
#[path = "composition/tests.rs"]
mod tests;
