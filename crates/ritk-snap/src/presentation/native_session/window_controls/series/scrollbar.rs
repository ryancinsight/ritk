//! Vertical scroll position for the series preview rail.

use super::super::super::series_browser::SeriesBrowser;
use super::navigator::{card_bounds, visible_count, SCROLLBAR_WIDTH};
use super::rect_contains;
use anyhow::Result;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};
use metis_ui_lang::{ScrollbarAxis, ScrollbarGeometry};

const TRACK_COLOR: Color = Color::rgb(47, 55, 64);
const THUMB_COLOR: Color = Color::rgb(111, 128, 143);
const TRACK_INSET: i32 = 3;

pub(super) fn render(
    framebuffer: &mut Framebuffer,
    area: Rect,
    browser: &SeriesBrowser,
) -> Result<()> {
    let Some(scrollbar) = geometry(area, browser.len(), browser.first_visible())? else {
        return Ok(());
    };
    fill_rect(
        framebuffer,
        scrollbar.track(),
        CornerRadius::SQUARE,
        TRACK_COLOR,
    );
    fill_rect(
        framebuffer,
        scrollbar.thumb(),
        CornerRadius::SQUARE,
        THUMB_COLOR,
    );
    Ok(())
}

pub(in crate::presentation::native_session::window_controls) fn page_direction(
    area: Rect,
    browser: &SeriesBrowser,
    x: f64,
    y: f64,
) -> Result<Option<i32>> {
    let Some(scrollbar) = geometry(area, browser.len(), browser.first_visible())? else {
        return Ok(None);
    };
    if !rect_contains(scrollbar.track(), x, y) {
        return Ok(None);
    }
    let thumb = scrollbar.thumb();
    let direction = if rect_contains(thumb, x, y) {
        0
    } else if y < f64::from(thumb.y) {
        -1
    } else {
        1
    };
    Ok(Some(direction))
}

fn geometry(area: Rect, total: usize, first_visible: usize) -> Result<Option<ScrollbarGeometry>> {
    let visible = visible_count(area);
    let bounds = card_bounds(area);
    let track_x = area
        .x
        .saturating_add(area.width)
        .saturating_sub(SCROLLBAR_WIDTH.saturating_add(TRACK_INSET));
    let track = Rect::new(track_x, bounds.y, SCROLLBAR_WIDTH, bounds.height);
    Ok(ScrollbarGeometry::new(
        ScrollbarAxis::Vertical,
        track,
        u64::try_from(total)?,
        u32::try_from(visible)?,
        u64::try_from(first_visible)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thumb_tracks_the_first_and_last_visible_series() {
        let area = Rect::new(0, 88, 276, 686);
        let visible = visible_count(area);
        let first = geometry(area, 20, 0)
            .expect("first scrollbar geometry")
            .expect("overflow shows a scrollbar");
        let last = geometry(area, 20, 20 - visible)
            .expect("last scrollbar geometry")
            .expect("overflow shows a scrollbar");

        assert_eq!(visible, 5);
        assert_eq!(first.track(), Rect::new(268, 144, 5, 622));
        assert_eq!(first.thumb(), Rect::new(268, 144, 5, 155));
        assert_eq!(last.thumb(), Rect::new(268, 611, 5, 155));
    }

    #[test]
    fn scrollbar_is_absent_when_all_series_fit() {
        let area = Rect::new(0, 88, 276, 686);
        let visible = visible_count(area);
        assert_eq!(
            geometry(area, visible, 0).expect("non-overflow layout"),
            None
        );
        assert!(geometry(area, visible + 1, 0)
            .expect("overflow layout")
            .is_some());
    }
}
