use anyhow::{Result, anyhow};
use metis_platform::Color;
use metis_platform::typeface::{TextSize, TextStyle};

pub(in crate::presentation::native_session) const PANEL_TITLE_SIZE: u32 = 14;
pub(in crate::presentation::native_session) const PANEL_DETAIL_SIZE: u32 = 12;
pub(in crate::presentation::native_session) const MEASUREMENT_LABEL_SIZE: u32 = 12;
pub(in crate::presentation::native_session) const SELECTOR_TITLE_SIZE: u32 = 16;
pub(in crate::presentation::native_session) const SELECTOR_ROW_SIZE: u32 = 14;
pub(in crate::presentation::native_session) const SELECTOR_FOOTER_SIZE: u32 = 12;

pub(in crate::presentation::native_session) fn text_style(
    color: Color,
    size_pixels: u32,
) -> Result<TextStyle> {
    let size = TextSize::new(f64::from(size_pixels))
        .ok_or_else(|| anyhow!("text size {size_pixels} is outside the Métis display range"))?;
    Ok(TextStyle::new(color, size))
}
