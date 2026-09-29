//! Rasterize bounded series thumbnails.

use crate::presentation::PresentationFrame;
use anyhow::{anyhow, Result};
use metis_platform::{Color, Framebuffer, Rect};

pub(in crate::presentation::native_session::window_controls) fn draw_thumbnail(
    framebuffer: &mut Framebuffer,
    bounds: Rect,
    frame: &PresentationFrame,
) -> Result<()> {
    if bounds.width <= 0 || bounds.height <= 0 {
        return Ok(());
    }
    let target_width =
        u32::try_from(bounds.width).map_err(|_| anyhow!("preview width is negative"))?;
    let target_height =
        u32::try_from(bounds.height).map_err(|_| anyhow!("preview height is negative"))?;
    let (draw_width, draw_height) =
        fit_dimensions(frame.width(), frame.height(), target_width, target_height)?;
    let left = u32::try_from(bounds.x)
        .map_err(|_| anyhow!("preview x is negative"))?
        .checked_add(target_width.saturating_sub(draw_width) / 2)
        .ok_or_else(|| anyhow!("preview x overflows"))?;
    let top = u32::try_from(bounds.y)
        .map_err(|_| anyhow!("preview y is negative"))?
        .checked_add(target_height.saturating_sub(draw_height) / 2)
        .ok_or_else(|| anyhow!("preview y overflows"))?;
    let source_width = usize::try_from(frame.width())
        .map_err(|_| anyhow!("preview source width exceeds usize"))?;
    let rgba = frame.rgba();
    for y in 0..draw_height {
        let source_y =
            usize::try_from(u64::from(y) * u64::from(frame.height()) / u64::from(draw_height))
                .map_err(|_| anyhow!("preview source row exceeds usize"))?;
        for x in 0..draw_width {
            let source_x =
                usize::try_from(u64::from(x) * u64::from(frame.width()) / u64::from(draw_width))
                    .map_err(|_| anyhow!("preview source column exceeds usize"))?;
            let offset = source_y
                .checked_mul(source_width)
                .and_then(|index| index.checked_add(source_x))
                .and_then(|index| index.checked_mul(4))
                .ok_or_else(|| anyhow!("preview source offset overflows"))?;
            let pixel = rgba
                .get(offset..offset.saturating_add(4))
                .ok_or_else(|| anyhow!("preview frame storage is incomplete"))?;
            let [red, green, blue, alpha] =
                <[u8; 4]>::try_from(pixel).map_err(|_| anyhow!("preview pixel is not RGBA"))?;
            let target_x = i32::try_from(left.saturating_add(x))
                .map_err(|_| anyhow!("preview target x exceeds i32"))?;
            let target_y = i32::try_from(top.saturating_add(y))
                .map_err(|_| anyhow!("preview target y exceeds i32"))?;
            framebuffer.set_pixel(target_x, target_y, Color::rgba(red, green, blue, alpha));
        }
    }
    Ok(())
}

pub(in crate::presentation::native_session::window_controls) fn fit_dimensions(
    source_width: u32,
    source_height: u32,
    target_width: u32,
    target_height: u32,
) -> Result<(u32, u32)> {
    if source_width == 0 || source_height == 0 || target_width == 0 || target_height == 0 {
        return Err(anyhow!("series preview dimensions must be nonzero"));
    }
    if u64::from(source_width) * u64::from(target_height)
        >= u64::from(source_height) * u64::from(target_width)
    {
        let height = u64::from(source_height) * u64::from(target_width) / u64::from(source_width);
        Ok((
            target_width,
            u32::try_from(height)
                .map_err(|_| anyhow!("series preview fitted height exceeds u32"))?
                .max(1),
        ))
    } else {
        let width = u64::from(source_width) * u64::from(target_height) / u64::from(source_height);
        Ok((
            u32::try_from(width)
                .map_err(|_| anyhow!("series preview fitted width exceeds u32"))?
                .max(1),
            target_height,
        ))
    }
}
