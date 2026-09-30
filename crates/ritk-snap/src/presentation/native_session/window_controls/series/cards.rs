//! Rendering for grouped patient, study, and series previews.

use super::super::super::layout::text_style;
use super::super::super::series_browser::{SeriesBrowser, SeriesChoice};
use super::super::series::{draw_fit, offset};
use super::super::SnapApp;
use super::metadata::render_study_metadata;
use super::navigator::{
    card_at, card_bounds, displayed_preview, visible_count, visible_start, HEADER_HEIGHT,
};
use super::scrollbar;
use super::thumbnails::draw_thumbnail;
use super::{MUTED, TEXT};
use crate::presentation::native_session::layout::TextStyle;
use crate::presentation::PresentationFrame;
use anyhow::{anyhow, Result};
use arrayvec::ArrayString;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};
use std::fmt::Write as _;

const PANEL_BACKGROUND: Color = Color::rgb(31, 37, 45);
const PANEL_EDGE: Color = Color::rgb(71, 81, 92);
const STUDY_EDGE: Color = Color::rgb(77, 129, 160);
const CARD_BACKGROUND: Color = Color::rgb(18, 22, 27);
const CARD_INACTIVE: Color = Color::rgb(38, 45, 54);
const CARD_ASSIGNED: Color = Color::rgb(47, 56, 66);
const CARD_ACTIVE: Color = Color::rgb(38, 87, 112);
const COUNT_BADGE_BACKGROUND: Color = Color::rgb(23, 58, 77);
const COUNT_BADGE_TEXT: Color = Color::rgb(242, 247, 251);
struct CardPresentation<'a> {
    browser: &'a SeriesBrowser,
    series_previews: &'a [Option<&'a PresentationFrame>],
    active_panel: usize,
    displayed_series: &'a [Option<usize>],
    title_style: TextStyle,
    detail_style: TextStyle,
}

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
    fill_rect(
        framebuffer,
        Rect::new(area.x, area.y, area.width, 1),
        CornerRadius::SQUARE,
        PANEL_EDGE,
    );

    let heading_style = text_style(TEXT, 13)?;
    let detail_style = text_style(MUTED, 11)?;
    draw_fit(
        framebuffer,
        offset(area.x, 10)?,
        offset(area.y, 8)?,
        "Series preview",
        heading_style,
        area.width.saturating_sub(130),
    );
    let mut summary = ArrayString::<64>::new();
    if let Some(browser) = browser {
        write!(
            &mut summary,
            "{} studies  |  {} series",
            browser.study_count(),
            browser.len()
        )
        .map_err(|_| anyhow!("series preview summary exceeds its display buffer"))?;
    } else if app.loaded.is_some() {
        summary
            .try_push_str("1 study  |  1 series")
            .map_err(|_| anyhow!("series preview summary exceeds its display buffer"))?;
    } else {
        summary
            .try_push_str("Open a study to view its series")
            .map_err(|_| anyhow!("series preview prompt exceeds its display buffer"))?;
    }
    draw_fit(
        framebuffer,
        offset(area.x, 10)?,
        offset(area.y, 25)?,
        summary.as_str(),
        detail_style,
        area.width.saturating_sub(20),
    );
    let mut target = ArrayString::<24>::new();
    write!(&mut target, "Load to P{}", active_panel + 1)
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
        offset(area.y, 8)?,
        target.as_str(),
        detail_style,
        target_width,
    );
    fill_rect(
        framebuffer,
        Rect::new(area.x, area.y.saturating_add(HEADER_HEIGHT), area.width, 1),
        CornerRadius::SQUARE,
        PANEL_EDGE,
    );
    fill_rect(
        framebuffer,
        Rect::new(
            area.x.saturating_add(area.width).saturating_sub(1),
            area.y,
            1,
            area.height,
        ),
        CornerRadius::SQUARE,
        PANEL_EDGE,
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
    let start = visible_start(browser, visible_count(area));
    let presentation = CardPresentation {
        browser,
        series_previews,
        active_panel,
        displayed_series,
        title_style: text_style(TEXT, 12)?,
        detail_style: text_style(MUTED, 10)?,
    };
    let count = visible_count(area);
    for offset_index in 0..count {
        let index = start.saturating_add(offset_index);
        let Some(choice) = browser.choice(index) else {
            break;
        };
        let card = card_at(bounds, offset_index)?;
        if card.x.saturating_add(card.width) > bounds.x.saturating_add(bounds.width) {
            break;
        }
        let previous = index.checked_sub(1).and_then(|prior| browser.choice(prior));
        let next = browser.choice(index.saturating_add(1));
        presentation.draw_card(framebuffer, card, index, choice, previous, next)?;
    }
    scrollbar::render(framebuffer, area, browser)
}

impl CardPresentation<'_> {
    fn draw_card(
        &self,
        framebuffer: &mut Framebuffer,
        card: Rect,
        index: usize,
        choice: &SeriesChoice,
        previous: Option<&SeriesChoice>,
        next: Option<&SeriesChoice>,
    ) -> Result<()> {
        let active = self
            .displayed_series
            .get(self.active_panel)
            .copied()
            .flatten()
            == Some(index);
        let assigned = self.displayed_series.contains(&Some(index));
        let background = if active {
            CARD_ACTIVE
        } else if assigned {
            CARD_ASSIGNED
        } else {
            CARD_INACTIVE
        };
        fill_rect(framebuffer, card, CornerRadius::SQUARE, background);
        if active {
            fill_rect(
                framebuffer,
                Rect::new(card.x, card.y, 3, card.height),
                CornerRadius::SQUARE,
                STUDY_EDGE,
            );
        }
        let starts_study = !same_study(previous, Some(choice));
        if starts_study {
            render_study_metadata(
                framebuffer,
                card,
                choice,
                self.browser.study_count(),
                self.detail_style,
            )?;
            fill_rect(
                framebuffer,
                Rect::new(card.x, card.y, card.width, 2),
                CornerRadius::SQUARE,
                STUDY_EDGE,
            );
            fill_rect(
                framebuffer,
                Rect::new(card.x, card.y, 2, card.height),
                CornerRadius::SQUARE,
                STUDY_EDGE,
            );
        }
        if !same_study(Some(choice), next) {
            let right = card.x.saturating_add(card.width).saturating_sub(2);
            fill_rect(
                framebuffer,
                Rect::new(right, card.y, 2, card.height),
                CornerRadius::SQUARE,
                STUDY_EDGE,
            );
            let bottom = card.y.saturating_add(card.height).saturating_sub(2);
            fill_rect(
                framebuffer,
                Rect::new(card.x, bottom, card.width, 2),
                CornerRadius::SQUARE,
                STUDY_EDGE,
            );
        }

        let thumbnail = Rect::new(
            offset(card.x, 6)?,
            offset(card.y, if starts_study { 52 } else { 8 })?,
            68,
            card.height
                .saturating_sub(if starts_study { 60 } else { 16 }),
        );
        fill_rect(
            framebuffer,
            thumbnail,
            CornerRadius::SQUARE,
            CARD_BACKGROUND,
        );
        if let Some(preview) = displayed_preview(index, self.displayed_series, self.series_previews)
            .or_else(|| self.browser.cached_thumbnail(index))
        {
            draw_thumbnail(framebuffer, thumbnail, preview)?;
        } else {
            draw_fit(
                framebuffer,
                offset(thumbnail.x, 4)?,
                offset(thumbnail.y, 28)?,
                "No preview",
                text_style(MUTED, 10)?,
                thumbnail.width.saturating_sub(8),
            );
        }
        draw_image_count(framebuffer, thumbnail, choice.image_count)?;

        let text_x = thumbnail
            .x
            .checked_add(thumbnail.width)
            .and_then(|x| x.checked_add(8))
            .ok_or_else(|| anyhow!("series card text x overflows"))?;
        let text_right = card
            .x
            .checked_add(card.width)
            .and_then(|x| x.checked_sub(7))
            .ok_or_else(|| anyhow!("series card right edge overflows"))?;
        let text_width = text_right.saturating_sub(text_x);
        let badge = assignment_badge(self.displayed_series, index)?;
        let badge_width = badge.as_ref().map_or(0, |value| {
            self.detail_style
                .extent(0, 0, value.as_str())
                .map_or(0, |rect| rect.width)
        });
        let first_line_width = text_width.saturating_sub(badge_width.saturating_add(4));
        let label = series_position_label(choice)?;
        draw_fit(
            framebuffer,
            text_x,
            offset(card.y, if starts_study { 53 } else { 6 })?,
            label.as_str(),
            self.detail_style,
            first_line_width,
        );
        if let Some(badge) = badge {
            draw_fit(
                framebuffer,
                text_right.saturating_sub(badge_width),
                offset(card.y, if starts_study { 53 } else { 6 })?,
                badge.as_str(),
                self.detail_style,
                badge_width,
            );
        }
        let (title_y, modality_y) = if starts_study { (69, 84) } else { (29, 61) };
        draw_fit(
            framebuffer,
            text_x,
            offset(card.y, title_y)?,
            choice.description.as_ref(),
            self.title_style,
            text_width,
        );
        draw_fit(
            framebuffer,
            text_x,
            offset(card.y, modality_y)?,
            choice.modality.as_ref(),
            self.detail_style,
            text_width,
        );
        Ok(())
    }
}

pub(super) fn draw_image_count(
    framebuffer: &mut Framebuffer,
    thumbnail: Rect,
    image_count: usize,
) -> Result<()> {
    let mut label = ArrayString::<20>::new();
    write!(&mut label, "{image_count}")
        .map_err(|_| anyhow!("series image count exceeds its display buffer"))?;
    let badge_style = text_style(COUNT_BADGE_TEXT, 10)?;
    let text_width = badge_style
        .extent(0, 0, label.as_str())
        .map_or(0, |bounds| bounds.width);
    let badge_width = text_width
        .saturating_add(8)
        .clamp(18, thumbnail.width.saturating_sub(4));
    let badge = Rect::new(
        thumbnail
            .x
            .saturating_add(thumbnail.width)
            .saturating_sub(badge_width)
            .saturating_sub(2),
        thumbnail
            .y
            .saturating_add(thumbnail.height)
            .saturating_sub(18),
        badge_width,
        16,
    );
    fill_rect(
        framebuffer,
        badge,
        CornerRadius::SQUARE,
        COUNT_BADGE_BACKGROUND,
    );
    draw_fit(
        framebuffer,
        badge.x.saturating_add(4),
        badge.y.saturating_add(3),
        label.as_str(),
        badge_style,
        badge.width.saturating_sub(8),
    );
    Ok(())
}

pub(super) fn series_position_label(choice: &SeriesChoice) -> Result<ArrayString<32>> {
    let mut label = ArrayString::new();
    write!(
        &mut label,
        "Series {}/{}",
        choice.study_series_number, choice.study_series_count
    )
    .map_err(|_| anyhow!("series position label exceeds its display buffer"))?;
    Ok(label)
}

pub(super) fn assignment_badge(
    displayed_series: &[Option<usize>],
    series_index: usize,
) -> Result<Option<ArrayString<32>>> {
    let panel_count = displayed_series
        .iter()
        .filter(|assigned| **assigned == Some(series_index))
        .count();
    if panel_count == 0 {
        return Ok(None);
    }
    let mut badge = ArrayString::new();
    let mut emitted = 0;
    for (panel, assigned) in displayed_series.iter().enumerate() {
        if *assigned != Some(series_index) {
            continue;
        }
        if emitted == 3 {
            break;
        }
        if !badge.is_empty() {
            badge
                .try_push_str(", ")
                .map_err(|_| anyhow!("panel assignment badge exceeds its display buffer"))?;
        }
        write!(&mut badge, "P{}", panel + 1)
            .map_err(|_| anyhow!("panel assignment badge exceeds its display buffer"))?;
        emitted += 1;
    }
    let omitted = panel_count.saturating_sub(emitted);
    if omitted > 0 {
        write!(&mut badge, ", +{omitted}")
            .map_err(|_| anyhow!("panel assignment badge exceeds its display buffer"))?;
    }
    Ok(Some(badge))
}

fn same_study(previous: Option<&SeriesChoice>, next: Option<&SeriesChoice>) -> bool {
    previous.zip(next).is_some_and(|(left, right)| {
        left.patient_number == right.patient_number && left.study_number == right.study_number
    })
}
