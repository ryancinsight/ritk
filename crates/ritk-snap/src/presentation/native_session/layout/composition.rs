//! Pixel composition for native presentation layouts.

use anyhow::{anyhow, bail, Result};
use metis_platform::{draw_text, fill_rect, Color, Framebuffer, Rect};

use super::super::frame::RenderedView;
use super::super::projection::RenderedProjection;
use super::geometry::{
    placement, placement_geometry, placement_with_bounds, NativeViewport, ScreenRect,
    VIEW_GAP_PIXELS,
};

/// Compose the three views into one bounded Métis framebuffer.
pub(crate) fn surface_frames(
    views: &[RenderedView; 3],
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: egui::Vec2,
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
        draw_application_overlay(&mut framebuffer, views, &viewports)?;
    }
    Ok((framebuffer, viewports))
}

/// Compose orthogonal planes and the RITK axial MIP in a bounded 2×2 layout.
pub(crate) fn surface_frames_with_mip(
    views: &[RenderedView; 3],
    projection: &RenderedProjection,
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
    pan_offset: egui::Vec2,
    show_application_overlay: bool,
) -> Result<(Framebuffer, [NativeViewport; 3])> {
    if surface_width == 0 || surface_height == 0 {
        bail!("native surface dimensions must be nonzero while rendering");
    }
    if !zoom.is_finite() || zoom <= 0.0 {
        bail!("native viewer zoom must be finite and positive");
    }
    let available_width = surface_width
        .checked_sub(VIEW_GAP_PIXELS)
        .ok_or_else(|| anyhow!("native MIP layout is narrower than its separator"))?;
    let available_height = surface_height
        .checked_sub(VIEW_GAP_PIXELS)
        .ok_or_else(|| anyhow!("native MIP layout is shorter than its separator"))?;
    if available_width < 2 || available_height < 2 {
        bail!("native MIP layout cannot allocate four panels");
    }
    let column_widths = [
        available_width / 2 + available_width % 2,
        available_width / 2,
    ];
    let row_heights = [
        available_height / 2 + available_height % 2,
        available_height / 2,
    ];
    let mut framebuffer = Framebuffer::new(surface_width, surface_height)
        .map_err(|error| anyhow!("allocate native MIP framebuffer: {error}"))?;
    framebuffer.clear(Color::BLACK);
    let viewports = [
        placement_with_bounds(
            &views[0],
            0,
            0,
            column_widths[0],
            row_heights[0],
            zoom,
            pan_offset,
        )?,
        placement_with_bounds(
            &views[1],
            column_widths[0] + VIEW_GAP_PIXELS,
            0,
            column_widths[1],
            row_heights[0],
            zoom,
            pan_offset,
        )?,
        placement_with_bounds(
            &views[2],
            0,
            row_heights[0] + VIEW_GAP_PIXELS,
            column_widths[0],
            row_heights[1],
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
    let projection_panel = ScreenRect {
        x: f64::from(column_widths[0] + VIEW_GAP_PIXELS),
        y: f64::from(row_heights[0] + VIEW_GAP_PIXELS),
        width: f64::from(column_widths[1]),
        height: f64::from(row_heights[1]),
    };
    let projection_image = placement_geometry(
        [projection.frame.width(), projection.frame.height()],
        projection.display_spacing,
        column_widths[0] + VIEW_GAP_PIXELS,
        row_heights[0] + VIEW_GAP_PIXELS,
        column_widths[1],
        row_heights[1],
        zoom,
        pan_offset,
    )?;
    blit_rgba_frame(
        &mut framebuffer,
        &projection.frame,
        projection_image,
        projection_panel,
        surface_width,
        surface_height,
    )?;
    if show_application_overlay {
        draw_application_overlay(&mut framebuffer, views, &viewports)?;
        draw_projection_overlay(
            &mut framebuffer,
            projection,
            column_widths[0] + VIEW_GAP_PIXELS,
            row_heights[0] + VIEW_GAP_PIXELS,
            column_widths[1],
            row_heights[1],
        )?;
    }
    Ok((framebuffer, viewports))
}

pub(crate) const OVERLAY_BAR_HEIGHT: i32 = 20;
const OVERLAY_MARGIN: i32 = 6;
const OVERLAY_BACKGROUND: Color = Color::rgba(0, 0, 0, 224);
pub(crate) const OVERLAY_TEXT: Color = Color::rgba(255, 255, 160, 255);

fn draw_application_overlay(
    framebuffer: &mut Framebuffer,
    views: &[RenderedView; 3],
    viewports: &[NativeViewport; 3],
) -> Result<()> {
    for (view, viewport) in views.iter().zip(viewports) {
        let panel_x =
            i32::try_from(viewport.panel_x).map_err(|_| anyhow!("native overlay x exceeds i32"))?;
        let panel_y =
            i32::try_from(viewport.panel_y).map_err(|_| anyhow!("native overlay y exceeds i32"))?;
        let panel_width = i32::try_from(viewport.panel_width)
            .map_err(|_| anyhow!("native overlay width exceeds i32"))?;
        let panel_height = i32::try_from(viewport.panel_height)
            .map_err(|_| anyhow!("native overlay height exceeds i32"))?;
        if panel_width <= OVERLAY_MARGIN * 2 {
            continue;
        }
        fill_rect(
            framebuffer,
            Rect::new(panel_x, panel_y, panel_width, OVERLAY_BAR_HEIGHT),
            OVERLAY_BACKGROUND,
        );
        fill_rect(
            framebuffer,
            Rect::new(
                panel_x,
                panel_y + panel_height - OVERLAY_BAR_HEIGHT,
                panel_width,
                OVERLAY_BAR_HEIGHT,
            ),
            OVERLAY_BACKGROUND,
        );
        let title = format!("METIS  RITK-SNAP  {}", view.plane_name);
        draw_text(
            framebuffer,
            panel_x + OVERLAY_MARGIN,
            panel_y + 2,
            &title,
            OVERLAY_TEXT,
            1,
        );
        let footer = format!(
            "Slice {}/{}  {}x{}  W:{:.0} C:{:.0}",
            view.slice_index.saturating_add(1),
            view.slice_count,
            view.frame.width(),
            view.frame.height(),
            view.window_level.width,
            view.window_level.center
        );
        draw_text(
            framebuffer,
            panel_x + OVERLAY_MARGIN,
            panel_y + panel_height - OVERLAY_BAR_HEIGHT + 2,
            &footer,
            OVERLAY_TEXT,
            1,
        );
    }
    Ok(())
}

fn draw_projection_overlay(
    framebuffer: &mut Framebuffer,
    projection: &RenderedProjection,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
) -> Result<()> {
    let panel_x =
        i32::try_from(panel_x).map_err(|_| anyhow!("native projection overlay x exceeds i32"))?;
    let panel_y =
        i32::try_from(panel_y).map_err(|_| anyhow!("native projection overlay y exceeds i32"))?;
    let panel_width = i32::try_from(panel_width)
        .map_err(|_| anyhow!("native projection overlay width exceeds i32"))?;
    let panel_height = i32::try_from(panel_height)
        .map_err(|_| anyhow!("native projection overlay height exceeds i32"))?;
    if panel_width <= OVERLAY_MARGIN * 2 || panel_height <= OVERLAY_BAR_HEIGHT * 2 {
        return Ok(());
    }
    fill_rect(
        framebuffer,
        Rect::new(panel_x, panel_y, panel_width, OVERLAY_BAR_HEIGHT),
        OVERLAY_BACKGROUND,
    );
    fill_rect(
        framebuffer,
        Rect::new(
            panel_x,
            panel_y + panel_height - OVERLAY_BAR_HEIGHT,
            panel_width,
            OVERLAY_BAR_HEIGHT,
        ),
        OVERLAY_BACKGROUND,
    );
    draw_text(
        framebuffer,
        panel_x + OVERLAY_MARGIN,
        panel_y + 2,
        "METIS  RITK-SNAP  3D MIP",
        OVERLAY_TEXT,
        1,
    );
    let footer = format!(
        "Axial MIP  {}x{}",
        projection.frame.width(),
        projection.frame.height()
    );
    draw_text(
        framebuffer,
        panel_x + OVERLAY_MARGIN,
        panel_y + panel_height - OVERLAY_BAR_HEIGHT + 2,
        &footer,
        OVERLAY_TEXT,
        1,
    );
    Ok(())
}

fn blit_frame(
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

fn blit_rgba_frame(
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
