//! Study-grouped series cards in the native viewer's preview rail.

use super::super::super::series_browser::SeriesBrowser;
use super::super::series::rect_contains;
use crate::presentation::PresentationFrame;
use anyhow::{anyhow, Result};
use metis_platform::Rect;

pub(super) const CARD_HEIGHT: i32 = 104;
pub(super) const CARD_GAP: i32 = 8;
pub(super) const HEADER_HEIGHT: i32 = 48;
const METADATA_HEIGHT: i32 = 132;
const CARD_BOTTOM: i32 = 8;
const CARD_LEFT: i32 = 8;
const CARD_RIGHT: i32 = 10;
pub(in crate::presentation::native_session::window_controls) const SCROLLBAR_WIDTH: i32 = 5;

pub(in crate::presentation::native_session::window_controls) fn card_bounds(area: Rect) -> Rect {
    let y = area
        .y
        .saturating_add(HEADER_HEIGHT)
        .saturating_add(METADATA_HEIGHT)
        .saturating_add(8);
    let bottom = area
        .y
        .saturating_add(area.height)
        .saturating_sub(CARD_BOTTOM);
    Rect::new(
        area.x.saturating_add(CARD_LEFT),
        y,
        area.width
            .saturating_sub(CARD_LEFT)
            .saturating_sub(CARD_RIGHT)
            .saturating_sub(SCROLLBAR_WIDTH.saturating_add(4))
            .max(0),
        bottom.saturating_sub(y).max(0),
    )
}

pub(in crate::presentation::native_session::window_controls) fn visible_count(area: Rect) -> usize {
    let bounds = card_bounds(area);
    if bounds.width < 180 || bounds.height < CARD_HEIGHT {
        return 0;
    }
    let available = i64::from(bounds.height).saturating_add(i64::from(CARD_GAP));
    let stride = i64::from(CARD_HEIGHT.saturating_add(CARD_GAP));
    usize::try_from(available / stride).expect("invariant: visible card count fits in usize")
}

pub(super) fn visible_start(browser: &SeriesBrowser, count: usize) -> usize {
    browser
        .first_visible()
        .min(browser.len().saturating_sub(count))
}

pub(super) fn card_at(bounds: Rect, visible_index: usize) -> Result<Rect> {
    let visible_i32 =
        i32::try_from(visible_index).map_err(|_| anyhow!("visible series index exceeds i32"))?;
    let stride = CARD_HEIGHT
        .checked_add(CARD_GAP)
        .ok_or_else(|| anyhow!("series card stride overflows"))?;
    let y = bounds
        .y
        .checked_add(
            visible_i32
                .checked_mul(stride)
                .ok_or_else(|| anyhow!("series card position overflows"))?,
        )
        .ok_or_else(|| anyhow!("series card y overflows"))?;
    Ok(Rect::new(bounds.x, y, bounds.width, CARD_HEIGHT))
}

pub(super) fn prepare_thumbnails(
    browser: &mut SeriesBrowser,
    area: Rect,
    displayed_series: &[Option<usize>],
) -> Result<()> {
    let count = visible_count(area);
    let start = visible_start(browser, count);
    for index in start..browser.len().min(start.saturating_add(count)) {
        if !displayed_series.contains(&Some(index)) && !browser.ensure_thumbnail(index) {
            return Err(anyhow!("visible series index is absent from catalog"));
        }
    }
    Ok(())
}

pub(in crate::presentation::native_session::window_controls) fn index_at(
    browser: &SeriesBrowser,
    area: Rect,
    x: f64,
    y: f64,
) -> Option<usize> {
    let bounds = card_bounds(area);
    if !rect_contains(bounds, x, y) || bounds.width < 180 || bounds.height < CARD_HEIGHT {
        return None;
    }
    let start = visible_start(browser, visible_count(area));
    for offset_index in 0..visible_count(area) {
        let card = card_at(bounds, offset_index).ok()?;
        if rect_contains(card, x, y) {
            let index = start.checked_add(offset_index)?;
            return (index < browser.len()).then_some(index);
        }
    }
    None
}

#[cfg(test)]
pub(in crate::presentation::native_session::window_controls) fn card_center(
    browser: &SeriesBrowser,
    area: Rect,
    index: usize,
) -> Option<(i32, i32)> {
    let count = visible_count(area);
    let start = visible_start(browser, count);
    let offset = index.checked_sub(start)?;
    if offset >= count {
        return None;
    }
    let card = card_at(card_bounds(area), offset).ok()?;
    Some((
        card.x.checked_add(card.width / 2)?,
        card.y.checked_add(card.height / 2)?,
    ))
}

pub(in crate::presentation::native_session::window_controls) fn displayed_preview<'a>(
    index: usize,
    displayed_series: &[Option<usize>],
    series_previews: &[Option<&'a PresentationFrame>],
) -> Option<&'a PresentationFrame> {
    displayed_series
        .iter()
        .position(|series_index| *series_index == Some(index))
        .and_then(|panel| series_previews.get(panel).copied().flatten())
}

#[cfg(test)]
mod tests {
    use arrayvec::ArrayString;

    #[test]
    fn assignment_badge_keeps_sparse_high_panel_identities() {
        let mut displayed = [None; 20];
        displayed[3] = Some(4);
        displayed[19] = Some(4);

        assert_eq!(
            super::super::cards::assignment_badge(&displayed, 4).expect("bounded assignments"),
            Some(ArrayString::from("P4, P20").expect("static badge fits"))
        );
        displayed[3] = None;
        assert_eq!(
            super::super::cards::assignment_badge(&displayed, 4).expect("bounded assignments"),
            Some(ArrayString::from("P20").expect("static badge fits"))
        );
    }

    #[test]
    fn repeated_panel_assignments_fit_a_bounded_badge() {
        let displayed = [Some(4); 20];
        assert_eq!(
            super::super::cards::assignment_badge(&displayed, 4).expect("bounded assignment badge"),
            Some(ArrayString::from("P1, P2, P3, +17").expect("static badge fits"))
        );
        assert_eq!(
            super::super::cards::assignment_badge(&displayed, 3).expect("unassigned series"),
            None
        );
    }
}
