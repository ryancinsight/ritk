use anyhow::{anyhow, Result};
use metis_platform::typeface::{TextSize, TextStyle};
use metis_platform::Color;

pub(in crate::presentation::native_session) const PANEL_TITLE_SIZE: u32 = 14;
pub(in crate::presentation::native_session) const PANEL_DETAIL_SIZE: u32 = 12;

pub(in crate::presentation::native_session) fn text_style(
    color: Color,
    size_pixels: u32,
) -> Result<TextStyle> {
    let size = TextSize::new(f64::from(size_pixels))
        .ok_or_else(|| anyhow!("text size {size_pixels} is outside the Métis display range"))?;
    Ok(TextStyle::new(color, size))
}
