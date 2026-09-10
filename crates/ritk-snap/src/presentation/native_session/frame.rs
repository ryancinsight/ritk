//! Slice-to-frame conversion for the native viewer session.

use crate::app::action_adapter::ViewerViewport;
use crate::app::SnapApp;
use crate::presentation::PresentationFrame;
use crate::render::{SliceRenderer, WindowLevel};
use crate::ui::{apply_to_image, ViewTransform};
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};
use anyhow::{anyhow, bail, Context, Result};
use metis_platform::{Color, Framebuffer};

pub(super) fn render_current_slice(
    app: &SnapApp,
) -> Result<(PresentationFrame, [usize; 2], ViewTransform)> {
    let volume = app
        .loaded
        .as_ref()
        .ok_or_else(|| anyhow!("native viewer has no loaded RITK volume"))?;
    let (index, _) = app.axis_slice_info(app.axis);
    let window_level = WindowLevel::new(
        f64::from(
            app.viewer_state
                .window_center
                .unwrap_or(DEFAULT_WINDOW_CENTER),
        ),
        f64::from(
            app.viewer_state
                .window_width
                .unwrap_or(DEFAULT_WINDOW_WIDTH)
                .max(1.0),
        ),
    );
    let image = SliceRenderer::render(volume, app.axis, index, window_level, app.colormap);
    let source_size = image.size;
    let transform = app.view_transform;
    let image = apply_to_image(&image, transform);
    let frame = PresentationFrame::from_color_image(&image)
        .context("convert RITK slice to a bounded presentation frame")?;
    Ok((frame, [source_size[0], source_size[1]], transform))
}

pub(super) fn surface_frame(
    frame: &PresentationFrame,
    source_size: [usize; 2],
    transform: ViewTransform,
    axis: usize,
    surface_width: u32,
    surface_height: u32,
    zoom: f32,
) -> Result<(Framebuffer, ViewerViewport)> {
    if surface_width == 0 || surface_height == 0 {
        bail!("native surface dimensions must be nonzero while rendering");
    }
    if source_size.contains(&0) || !zoom.is_finite() || zoom <= 0.0 {
        bail!("native viewer frame geometry must be finite and nonzero");
    }
    let frame_width = f64::from(frame.width());
    let frame_height = f64::from(frame.height());
    let scale = (f64::from(surface_width) / frame_width)
        .min(f64::from(surface_height) / frame_height)
        * f64::from(zoom);
    if !scale.is_finite() || scale <= 0.0 {
        bail!("native viewer scale is outside the finite positive range");
    }
    let origin_x = (f64::from(surface_width) - frame_width * scale) * 0.5;
    let origin_y = (f64::from(surface_height) - frame_height * scale) * 0.5;
    if scale > f64::from(f32::MAX)
        || scale < f64::from(f32::MIN)
        || origin_x > f64::from(f32::MAX)
        || origin_x < f64::from(f32::MIN)
        || origin_y > f64::from(f32::MAX)
        || origin_y < f64::from(f32::MIN)
    {
        bail!("native viewer geometry exceeds f32 range");
    }
    #[expect(
        clippy::cast_possible_truncation,
        reason = "scale and origin are checked against the f32 range above"
    )]
    let texel = scale as f32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "scale and origin are checked against the f32 range above"
    )]
    let origin_x = origin_x as f32;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "scale and origin are checked against the f32 range above"
    )]
    let origin_y = origin_y as f32;
    let viewport = ViewerViewport::new(
        axis,
        egui::pos2(origin_x, origin_y),
        egui::vec2(texel, texel),
        source_size,
        transform,
    )
    .map_err(|error| anyhow!("construct native viewer viewport: {error}"))?;

    let mut framebuffer = Framebuffer::new(surface_width, surface_height)
        .map_err(|error| anyhow!("allocate native viewer framebuffer: {error}"))?;
    framebuffer.clear(Color::BLACK);
    let frame_width_usize = usize::try_from(frame.width())?;
    let frame_height_usize = usize::try_from(frame.height())?;
    for y in 0..surface_height {
        for x in 0..surface_width {
            let source_x = ((f64::from(x) + 0.5 - f64::from(origin_x)) / f64::from(texel)).floor();
            let source_y = ((f64::from(y) + 0.5 - f64::from(origin_y)) / f64::from(texel)).floor();
            if source_x < 0.0
                || source_y < 0.0
                || source_x >= frame_width
                || source_y >= frame_height
            {
                continue;
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "source coordinates are checked against bounded frame dimensions"
            )]
            let source_x = source_x as usize;
            #[expect(
                clippy::cast_possible_truncation,
                reason = "source coordinates are checked against bounded frame dimensions"
            )]
            let source_y = source_y as usize;
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
    Ok((framebuffer, viewport))
}
