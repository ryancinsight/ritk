//! RadiAnt-style series rail beside the native image workspace.

use super::super::layout::text_style;
use super::super::series_browser::{SeriesBrowser, SeriesChoice};
use crate::app::SnapApp;
use crate::presentation::PresentationFrame;
use anyhow::{anyhow, Result};
use arrayvec::ArrayString;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::typeface::draw_text;
use metis_platform::{Color, Framebuffer, Rect};
use std::fmt::Write as _;

const PANEL_BACKGROUND: Color = Color::rgb(31, 37, 45);
const PANEL_EDGE: Color = Color::rgb(71, 81, 92);
const CARD_BACKGROUND: Color = Color::rgb(18, 22, 27);
const CARD_INACTIVE: Color = Color::rgb(38, 45, 54);
const CARD_ACTIVE: Color = Color::rgb(38, 87, 112);
const TEXT_COLOR: Color = Color::rgb(225, 232, 239);
const MUTED_COLOR: Color = Color::rgb(153, 169, 183);
const CARD_HEIGHT: i32 = 88;
const CARD_GAP: i32 = 7;
const CARD_TOP: i32 = 60;
const CARD_BOTTOM: i32 = 8;

pub(super) fn render(
    framebuffer: &mut Framebuffer,
    area: Rect,
    browser: Option<&SeriesBrowser>,
    app: &SnapApp,
    series_previews: &[Option<&PresentationFrame>],
    active_panel: usize,
    displayed_series: &[Option<usize>],
) -> Result<()> {
    if area.width <= 0 || area.height <= 0 {
        return Ok(());
    }
    fill_rect(framebuffer, area, CornerRadius::SQUARE, PANEL_BACKGROUND);
    let right = area
        .x
        .checked_add(area.width)
        .and_then(|edge| edge.checked_sub(1))
        .ok_or_else(|| anyhow!("series preview divider x overflows"))?;
    fill_rect(
        framebuffer,
        Rect::new(right, area.y, 1, area.height),
        CornerRadius::SQUARE,
        PANEL_EDGE,
    );

    let heading_style = text_style(TEXT_COLOR, 11)?;
    let detail_style = text_style(MUTED_COLOR, 10)?;
    draw_text(
        framebuffer,
        offset(area.x, 10)?,
        offset(area.y, 8)?,
        "STUDIES AND SERIES",
        heading_style,
    );
    if let Some(browser) = browser {
        let mut summary = ArrayString::<64>::new();
        write!(
            &mut summary,
            "{} studies  |  {} series",
            browser.study_count(),
            browser.len()
        )
        .map_err(|_| anyhow!("series preview summary exceeds its display buffer"))?;
        draw_text(
            framebuffer,
            offset(area.x, 10)?,
            offset(area.y, 27)?,
            summary.as_str(),
            detail_style,
        );
    } else if app.loaded.is_some() {
        draw_text(
            framebuffer,
            offset(area.x, 10)?,
            offset(area.y, 27)?,
            "1 study  |  1 series",
            detail_style,
        );
    } else {
        draw_text(
            framebuffer,
            offset(area.x, 10)?,
            offset(area.y, 27)?,
            "Open a study to view its series",
            detail_style,
        );
    }
    let mut target = ArrayString::<24>::new();
    write!(&mut target, "LOAD TARGET: P{}", active_panel + 1)
        .map_err(|_| anyhow!("series target label exceeds its display buffer"))?;
    let target_width = detail_style
        .extent(0, 0, target.as_str())
        .map_or(0, |bounds| bounds.width);
    let target_x = area
        .x
        .saturating_add(area.width)
        .saturating_sub(target_width)
        .saturating_sub(12);
    draw_fit(
        framebuffer,
        target_x,
        offset(area.y, 43)?,
        target.as_str(),
        detail_style,
        area.x
            .saturating_add(area.width)
            .saturating_sub(target_x)
            .saturating_sub(12),
    );

    if let Some(browser) = browser {
        render_cards(
            framebuffer,
            area,
            browser,
            series_previews,
            displayed_series,
            active_panel,
        )?;
    }
    Ok(())
}

fn render_cards(
    framebuffer: &mut Framebuffer,
    area: Rect,
    browser: &SeriesBrowser,
    series_previews: &[Option<&PresentationFrame>],
    displayed_series: &[Option<usize>],
    active_panel: usize,
) -> Result<()> {
    let bounds = card_bounds(area);
    let count = visible_count(area);
    let start = visible_start(browser, count);
    let title_style = text_style(TEXT_COLOR, 11)?;
    let detail_style = text_style(MUTED_COLOR, 10)?;
    for visible_index in 0..count {
        let index = start.saturating_add(visible_index);
        let Some(choice) = browser.choice(index) else {
            break;
        };
        let visible_i32 = i32::try_from(visible_index)
            .map_err(|_| anyhow!("visible series index exceeds i32"))?;
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
        let card = Rect::new(bounds.x, y, bounds.width, CARD_HEIGHT);
        fill_rect(
            framebuffer,
            card,
            CornerRadius::SQUARE,
            if displayed_series.get(active_panel).copied().flatten() == Some(index) {
                CARD_ACTIVE
            } else {
                CARD_INACTIVE
            },
        );

        let image = Rect::new(
            offset(card.x, 6)?,
            offset(card.y, 6)?,
            64.min(card.width.saturating_sub(12)),
            card.height.saturating_sub(12),
        );
        fill_rect(framebuffer, image, CornerRadius::SQUARE, CARD_BACKGROUND);
        if let Some(preview) = displayed_preview(index, displayed_series, series_previews) {
            draw_thumbnail(framebuffer, image, preview)?;
        } else {
            draw_fit(
                framebuffer,
                offset(image.x, 8)?,
                offset(image.y, u32::try_from((image.height - 12).max(0) / 2)?)?,
                choice.modality.as_ref(),
                text_style(TEXT_COLOR, 15)?,
                image.width.saturating_sub(16),
            );
        }

        let text_x = offset(image.x, u32::try_from(image.width.saturating_add(10))?)?;
        let text_width = card
            .x
            .saturating_add(card.width)
            .saturating_sub(text_x)
            .saturating_sub(8);
        let group = group_label(choice)?;
        let badge_reserve = if displayed_series.contains(&Some(index)) {
            38
        } else {
            0
        };
        draw_fit(
            framebuffer,
            text_x,
            offset(card.y, 6)?,
            group.as_str(),
            detail_style,
            text_width.saturating_sub(badge_reserve),
        );
        draw_fit(
            framebuffer,
            text_x,
            offset(card.y, 24)?,
            choice.description.as_ref(),
            title_style,
            text_width,
        );
        draw_fit(
            framebuffer,
            text_x,
            offset(card.y, 42)?,
            choice.modality.as_ref(),
            detail_style,
            text_width,
        );
        let mut details = ArrayString::<40>::new();
        write!(
            &mut details,
            "{} images  |  Study {}",
            choice.instance_count, choice.study_number
        )
        .map_err(|_| anyhow!("series details exceed their display buffer"))?;
        draw_fit(
            framebuffer,
            text_x,
            offset(card.y, 60)?,
            details.as_str(),
            detail_style,
            text_width,
        );

        if displayed_series.contains(&Some(index)) {
            let mut badge = ArrayString::<64>::new();
            for (panel, series_index) in displayed_series.iter().enumerate() {
                if *series_index == Some(index) {
                    if !badge.is_empty() {
                        badge.try_push_str(", ").map_err(|_| {
                            anyhow!("series panel badge exceeds its display buffer")
                        })?;
                    }
                    write!(&mut badge, "P{}", panel + 1)
                        .map_err(|_| anyhow!("series panel badge exceeds its display buffer"))?;
                }
            }
            let badge_style = text_style(TEXT_COLOR, 10)?;
            let badge_width = badge_style
                .extent(0, 0, badge.as_str())
                .map_or(0, |extent| extent.width);
            let badge_x = card
                .x
                .saturating_add(card.width)
                .saturating_sub(badge_width)
                .saturating_sub(8);
            draw_fit(
                framebuffer,
                badge_x,
                offset(card.y, 6)?,
                badge.as_str(),
                badge_style,
                badge_width,
            );
        }
    }
    Ok(())
}

fn displayed_preview<'a>(
    index: usize,
    displayed_series: &[Option<usize>],
    series_previews: &[Option<&'a PresentationFrame>],
) -> Option<&'a PresentationFrame> {
    displayed_series
        .iter()
        .position(|series_index| *series_index == Some(index))
        .and_then(|panel| series_previews.get(panel).copied().flatten())
}

pub(super) fn index_at(browser: &SeriesBrowser, area: Rect, x: f64, y: f64) -> Option<usize> {
    let bounds = card_bounds(area);
    if !rect_contains(bounds, x, y) {
        return None;
    }
    if bounds.width <= 0 || bounds.height < CARD_HEIGHT {
        return None;
    }
    let start = visible_start(browser, visible_count(area));
    for offset in 0..visible_count(area) {
        let visible_i32 = i32::try_from(offset).ok()?;
        let stride = CARD_HEIGHT.checked_add(CARD_GAP)?;
        let card_y = bounds.y.checked_add(visible_i32.checked_mul(stride)?)?;
        if rect_contains(Rect::new(bounds.x, card_y, bounds.width, CARD_HEIGHT), x, y) {
            let index = start.checked_add(offset)?;
            return (index < browser.len()).then_some(index);
        }
    }
    None
}

pub(super) fn visible_count(area: Rect) -> usize {
    let bounds = card_bounds(area);
    if bounds.width == 0 || bounds.height < CARD_HEIGHT {
        return 0;
    }
    let available = i64::from(bounds.height).saturating_add(i64::from(CARD_GAP));
    let stride = i64::from(CARD_HEIGHT.saturating_add(CARD_GAP));
    usize::try_from(available / stride).expect("invariant: visible card count fits in usize")
}

fn visible_start(browser: &SeriesBrowser, count: usize) -> usize {
    browser
        .first_visible()
        .min(browser.len().saturating_sub(count))
}

fn card_bounds(area: Rect) -> Rect {
    let inset = 8_i32;
    let y = area.y.saturating_add(CARD_TOP);
    let bottom = area
        .y
        .saturating_add(area.height)
        .saturating_sub(CARD_BOTTOM);
    Rect::new(
        area.x.saturating_add(inset),
        y,
        area.width.saturating_sub(inset.saturating_mul(2)).max(0),
        bottom.saturating_sub(y).max(0),
    )
}

fn group_label(choice: &SeriesChoice) -> Result<ArrayString<64>> {
    let mut label = ArrayString::new();
    write!(
        &mut label,
        "PATIENT {}  /  STUDY {}",
        choice.patient_number, choice.study_number
    )
    .map_err(|_| anyhow!("study group label exceeds its display buffer"))?;
    Ok(label)
}

fn draw_thumbnail(
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

fn fit_dimensions(
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

fn draw_fit(
    framebuffer: &mut Framebuffer,
    x: i32,
    y: i32,
    text: &str,
    style: metis_platform::typeface::TextStyle,
    max_width: i32,
) {
    let mut end = 0;
    for (start, character) in text.char_indices() {
        let candidate_end = start + character.len_utf8();
        let candidate = &text[..candidate_end];
        let width = style
            .extent(0, 0, candidate)
            .map_or(i32::MAX, |rect| rect.width);
        if width > max_width {
            break;
        }
        end = candidate_end;
    }
    draw_text(framebuffer, x, y, &text[..end], style);
}

fn offset(origin: i32, distance: u32) -> Result<i32> {
    let distance =
        i32::try_from(distance).map_err(|_| anyhow!("series text offset exceeds i32"))?;
    origin
        .checked_add(distance)
        .ok_or_else(|| anyhow!("series text position overflows"))
}

fn rect_contains(rect: Rect, x: f64, y: f64) -> bool {
    if rect.width <= 0 || rect.height <= 0 || !x.is_finite() || !y.is_finite() {
        return false;
    }
    let left = f64::from(rect.x);
    let top = f64::from(rect.y);
    x >= left && y >= top && x < left + f64::from(rect.width) && y < top + f64::from(rect.height)
}
