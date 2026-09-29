//! Vertical scroll position for the series preview rail.

use super::super::super::series_browser::SeriesBrowser;
use super::navigator::{card_bounds, visible_count, SCROLLBAR_WIDTH};
use super::rect_contains;
use anyhow::{anyhow, Result};
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};

const TRACK_COLOR: Color = Color::rgb(47, 55, 64);
const THUMB_COLOR: Color = Color::rgb(111, 128, 143);
const MIN_THUMB_HEIGHT: i64 = 16;
const TRACK_INSET: i32 = 3;

pub(super) fn render(
    framebuffer: &mut Framebuffer,
    area: Rect,
    browser: &SeriesBrowser,
) -> Result<()> {
    let Some((track, thumb)) = geometry(area, browser.len(), browser.first_visible())? else {
        return Ok(());
    };
    fill_rect(framebuffer, track, CornerRadius::SQUARE, TRACK_COLOR);
    fill_rect(framebuffer, thumb, CornerRadius::SQUARE, THUMB_COLOR);
    Ok(())
}

pub(in crate::presentation::native_session::window_controls) fn page_direction(
    area: Rect,
    browser: &SeriesBrowser,
    x: f64,
    y: f64,
) -> Result<Option<i32>> {
    let Some((track, thumb)) = geometry(area, browser.len(), browser.first_visible())? else {
        return Ok(None);
    };
    if !rect_contains(track, x, y) {
        return Ok(None);
    }
    let direction = if rect_contains(thumb, x, y) {
        0
    } else if y < f64::from(thumb.y) {
        -1
    } else {
        1
    };
    Ok(Some(direction))
}

fn geometry(area: Rect, total: usize, first_visible: usize) -> Result<Option<(Rect, Rect)>> {
    let visible = visible_count(area);
    let bounds = card_bounds(area);
    if total <= visible || visible == 0 || bounds.height <= 0 {
        return Ok(None);
    }

    let travel = i64::from(bounds.height);
    let total_i64 = i64::try_from(total).map_err(|_| anyhow!("series count exceeds i64"))?;
    let visible_i64 =
        i64::try_from(visible).map_err(|_| anyhow!("visible series count exceeds i64"))?;
    let thumb_height = travel
        .checked_mul(visible_i64)
        .ok_or_else(|| anyhow!("series scrollbar thumb height overflows"))?
        / total_i64;
    let thumb_height = thumb_height.clamp(MIN_THUMB_HEIGHT.min(travel), travel);
    let max_first = total.saturating_sub(visible);
    let first_visible = first_visible.min(max_first);
    let max_first_i64 =
        i64::try_from(max_first).map_err(|_| anyhow!("scrollable series count exceeds i64"))?;
    let first_visible_i64 =
        i64::try_from(first_visible).map_err(|_| anyhow!("series scroll position exceeds i64"))?;
    let thumb_travel = travel.saturating_sub(thumb_height);
    let thumb_offset = thumb_travel
        .checked_mul(first_visible_i64)
        .ok_or_else(|| anyhow!("series scrollbar thumb position overflows"))?
        / max_first_i64;
    let thumb_y = bounds.y.saturating_add(
        i32::try_from(thumb_offset)
            .map_err(|_| anyhow!("series scrollbar thumb position exceeds i32"))?,
    );
    let track_x = area
        .x
        .saturating_add(area.width)
        .saturating_sub(SCROLLBAR_WIDTH.saturating_add(TRACK_INSET));
    let track = Rect::new(track_x, bounds.y, SCROLLBAR_WIDTH, bounds.height);
    let thumb = Rect::new(
        track.x,
        thumb_y,
        track.width,
        i32::try_from(thumb_height)
            .map_err(|_| anyhow!("series scrollbar thumb height exceeds i32"))?,
    );
    Ok(Some((track, thumb)))
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

        assert_eq!(visible, 4);
        assert_eq!(first.0, Rect::new(268, 276, 5, 490));
        assert_eq!(first.1, Rect::new(268, 276, 5, 98));
        assert_eq!(last.1, Rect::new(268, 668, 5, 98));
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
