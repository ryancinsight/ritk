//! Rectangular regions reserved by native viewer chrome.

use super::super::super::layout::ViewportArea;
use anyhow::{anyhow, Result};
use metis_platform::Rect;

const MENU_HEIGHT: u32 = 30;
const TOOLBAR_HEIGHT: u32 = 48;
const STATUS_HEIGHT: u32 = 26;
const SERIES_PREVIEW_WIDTH: u32 = 276;
const MIN_SERIES_PREVIEW_WINDOW_WIDTH: u32 = 640;
const MIN_SERIES_PREVIEW_WINDOW_HEIGHT: u32 = 480;
const MIN_VIEWPORT_WIDTH: u32 = 360;

pub(in crate::presentation::native_session::window_controls) struct ChromeGeometry {
    pub(in crate::presentation::native_session::window_controls) menu_bar: Rect,
    pub(in crate::presentation::native_session::window_controls) toolbar: Rect,
    pub(in crate::presentation::native_session::window_controls) series_preview: Rect,
    pub(in crate::presentation::native_session::window_controls) status_bar: Rect,
    pub(in crate::presentation::native_session::window_controls) viewport_area: ViewportArea,
    pub(in crate::presentation::native_session::window_controls) menu_height: u32,
    pub(in crate::presentation::native_session::window_controls) toolbar_height: u32,
}

impl ChromeGeometry {
    pub(in crate::presentation::native_session::window_controls) fn new(
        width: u32,
        height: u32,
        show_series_preview: bool,
    ) -> Result<Self> {
        let menu_height = height.min(MENU_HEIGHT);
        let status_height = height.saturating_sub(menu_height).min(STATUS_HEIGHT);
        let toolbar_height = height
            .saturating_sub(menu_height)
            .saturating_sub(status_height)
            .min(TOOLBAR_HEIGHT);
        let content_y = menu_height
            .checked_add(toolbar_height)
            .ok_or_else(|| anyhow!("native content y overflows"))?;
        let content_height = height
            .saturating_sub(menu_height)
            .saturating_sub(toolbar_height)
            .saturating_sub(status_height);
        let preview_width = if show_series_preview
            && width >= MIN_SERIES_PREVIEW_WINDOW_WIDTH
            && height >= MIN_SERIES_PREVIEW_WINDOW_HEIGHT
        {
            SERIES_PREVIEW_WIDTH.min(width.saturating_sub(MIN_VIEWPORT_WIDTH))
        } else {
            0
        };
        let width_i32 =
            i32::try_from(width).map_err(|_| anyhow!("native chrome width exceeds i32"))?;
        let menu_height_i32 =
            i32::try_from(menu_height).map_err(|_| anyhow!("native menu height exceeds i32"))?;
        let toolbar_height_i32 = i32::try_from(toolbar_height)
            .map_err(|_| anyhow!("native toolbar height exceeds i32"))?;
        let status_height_i32 = i32::try_from(status_height)
            .map_err(|_| anyhow!("native status height exceeds i32"))?;
        let preview_width_i32 = i32::try_from(preview_width)
            .map_err(|_| anyhow!("native series preview width exceeds i32"))?;
        let content_y_i32 =
            i32::try_from(content_y).map_err(|_| anyhow!("native content y exceeds i32"))?;
        let content_height_i32 = i32::try_from(content_height)
            .map_err(|_| anyhow!("native content height exceeds i32"))?;
        let status_y = i32::try_from(height.saturating_sub(status_height))
            .map_err(|_| anyhow!("native status y exceeds i32"))?;
        Ok(Self {
            menu_bar: Rect::new(0, 0, width_i32, menu_height_i32),
            toolbar: Rect::new(0, menu_height_i32, width_i32, toolbar_height_i32),
            series_preview: Rect::new(0, content_y_i32, preview_width_i32, content_height_i32),
            status_bar: Rect::new(0, status_y, width_i32, status_height_i32),
            viewport_area: ViewportArea {
                x: preview_width,
                y: content_y,
                width: width.saturating_sub(preview_width),
                height: content_height,
            },
            menu_height,
            toolbar_height,
        })
    }
}
