//! DICOM image values overlaid on each comparison panel.

use super::super::super::frame::RenderedView;
use super::super::text::{draw_text, TextStyle};
use anyhow::{anyhow, Result};
use arrayvec::ArrayString;
use metis_platform::Framebuffer;
use std::fmt::Write as _;

pub(super) fn draw_dicom_corner_values(
    framebuffer: &mut Framebuffer,
    view: &RenderedView,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
    style: &TextStyle,
    shadow: &TextStyle,
) -> Result<()> {
    if panel_width < 140 || panel_height < 48 {
        return Ok(());
    }
    let panel_x = i32::try_from(panel_x).map_err(|_| anyhow!("DICOM annotation x exceeds i32"))?;
    let panel_y = i32::try_from(panel_y).map_err(|_| anyhow!("DICOM annotation y exceeds i32"))?;
    let panel_width =
        i32::try_from(panel_width).map_err(|_| anyhow!("DICOM annotation width exceeds i32"))?;
    let panel_height =
        i32::try_from(panel_height).map_err(|_| anyhow!("DICOM annotation height exceeds i32"))?;

    let mut image = ArrayString::<40>::new();
    write!(
        &mut image,
        "Image {} / {}",
        view.slice_index.saturating_add(1),
        view.slice_count
    )
    .map_err(|_| anyhow!("DICOM image annotation exceeds its buffer"))?;
    let left_x = panel_x
        .checked_add(8)
        .ok_or_else(|| anyhow!("DICOM annotation left edge overflows"))?;
    let top_y = panel_y
        .checked_add(7)
        .ok_or_else(|| anyhow!("DICOM annotation top edge overflows"))?;
    draw_shadowed_text(framebuffer, left_x, top_y, image.as_str(), style, shadow)?;

    let mut window = ArrayString::<48>::new();
    write!(
        &mut window,
        "W {:.0}  C {:.0}",
        view.window_level.width, view.window_level.center
    )
    .map_err(|_| anyhow!("DICOM window annotation exceeds its buffer"))?;
    let bottom_y = panel_y
        .checked_add(panel_height)
        .and_then(|bottom| bottom.checked_sub(18))
        .ok_or_else(|| anyhow!("DICOM annotation bottom edge overflows"))?;
    draw_shadowed_text(
        framebuffer,
        left_x,
        bottom_y,
        window.as_str(),
        style,
        shadow,
    )?;

    let mut dimensions = ArrayString::<32>::new();
    write!(
        &mut dimensions,
        "{}x{}",
        view.frame.width(),
        view.frame.height()
    )
    .map_err(|_| anyhow!("DICOM dimensions annotation exceeds its buffer"))?;
    let text_width = style
        .extent(0, 0, dimensions.as_str())
        .map_or(0, |bounds| bounds.width);
    let right_x = panel_x
        .checked_add(panel_width)
        .and_then(|right| right.checked_sub(text_width))
        .and_then(|right| right.checked_sub(8))
        .ok_or_else(|| anyhow!("DICOM annotation right edge overflows"))?;
    draw_shadowed_text(
        framebuffer,
        right_x,
        bottom_y,
        dimensions.as_str(),
        style,
        shadow,
    )?;
    Ok(())
}

fn draw_shadowed_text(
    framebuffer: &mut Framebuffer,
    x: i32,
    y: i32,
    value: &str,
    style: &TextStyle,
    shadow: &TextStyle,
) -> Result<()> {
    let shadow_x = x
        .checked_add(1)
        .ok_or_else(|| anyhow!("DICOM annotation shadow x overflows"))?;
    let shadow_y = y
        .checked_add(1)
        .ok_or_else(|| anyhow!("DICOM annotation shadow y overflows"))?;
    draw_text(framebuffer, shadow_x, shadow_y, value, *shadow);
    draw_text(framebuffer, x, y, value, *style);
    Ok(())
}
