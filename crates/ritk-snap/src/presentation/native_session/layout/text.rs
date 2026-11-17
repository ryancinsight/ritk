//! Typography shared by the native viewer chrome and its display list.

use anyhow::{anyhow, Result};
use metis_platform::typeface::TextSize;
use metis_platform::{Color, Framebuffer};
use metis_ui_lang::DisplayCommand;
use std::borrow::Cow;

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
        text: std::sync::Arc::from(text.into().as_str()),
        x,
        y,
        style,
    }
}

pub(super) fn fit_text(text: &str, width: i32, style: TextStyle) -> Cow<'_, str> {
    const ELLIPSIS: &str = "...";
    let fits = |value: &str| {
        style
            .extent(0, 0, value)
            .is_some_and(|rect| rect.width <= width)
    };
    if fits(text) {
        return Cow::Borrowed(text);
    }
    if !fits(ELLIPSIS) || text.is_empty() {
        return Cow::Borrowed("");
    }
    let mut shortened = String::from(ELLIPSIS);
    for character in text.chars() {
        let end = shortened.len() - ELLIPSIS.len();
        shortened.insert(end, character);
        if !fits(&shortened) {
            shortened.remove(end);
            break;
        }
    }
    Cow::Owned(shortened)
}
