//! Typography shared by the native viewer chrome and its display list.

use anyhow::{anyhow, Result};
use metis_platform::typeface::TextSize;
use metis_platform::{Color, Framebuffer};
use metis_ui_lang::DisplayCommand;

pub(in crate::presentation::native_session) use metis_platform::typeface::TextStyle;

pub(in crate::presentation::native_session) const PANEL_TITLE_SIZE: u32 = 16;
pub(in crate::presentation::native_session) const PANEL_DETAIL_SIZE: u32 = 13;

pub(in crate::presentation::native_session) fn text_style(
    color: Color,
    size_pixels: u32,
) -> Result<TextStyle> {
    let size = TextSize::new(f64::from(size_pixels))
        .ok_or_else(|| anyhow!("text size {size_pixels} is outside the Métis display range"))?;
    Ok(TextStyle::new(color, size))
}

pub(in crate::presentation::native_session) fn draw_text(
    framebuffer: &mut Framebuffer,
    x: i32,
    y: i32,
    text: &str,
    style: TextStyle,
) {
    metis_platform::typeface::draw_text(framebuffer, x, y, text, style);
}

pub(in crate::presentation::native_session) fn text_command(
    text: impl Into<String>,
    x: i32,
    y: i32,
    style: TextStyle,
) -> DisplayCommand {
    DisplayCommand::DrawText {
        text: text.into(),
        x,
        y,
        style,
    }
}
