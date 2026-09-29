//! RadiAnt-style study and series navigator.

pub(super) mod cards;
mod labels;
mod metadata;
pub(super) mod navigator;
pub(super) mod scrollbar;
mod thumbnails;

#[cfg(test)]
mod tests;

use super::super::layout::{draw_text, TextStyle};
use super::super::series_browser::SeriesBrowser;
use crate::app::SnapApp;
use crate::presentation::PresentationFrame;
use anyhow::{anyhow, Result};
use metis_platform::{Color, Framebuffer, Rect};

pub(super) const TEXT: Color = Color::rgb(225, 232, 239);
pub(super) const STUDY_TEXT: Color = Color::rgb(181, 216, 237);
pub(super) const MUTED: Color = Color::rgb(153, 169, 183);

pub(super) use labels::{
    display_patient_name, format_dicom_date, format_dicom_time, patient_label, study_label,
};

pub(super) fn render(
    framebuffer: &mut Framebuffer,
    area: Rect,
    browser: Option<&SeriesBrowser>,
    app: &SnapApp,
    series_previews: &[Option<&PresentationFrame>],
    active_panel: usize,
    displayed_series: &[Option<usize>],
) -> Result<()> {
    cards::render(
        framebuffer,
        area,
        browser,
        app,
        series_previews,
        active_panel,
        displayed_series,
    )
}

pub(super) fn prepare_thumbnails(
    browser: &mut SeriesBrowser,
    area: Rect,
    displayed_series: &[Option<usize>],
) -> Result<()> {
    navigator::prepare_thumbnails(browser, area, displayed_series)
}

pub(super) fn draw_fit(
    framebuffer: &mut Framebuffer,
    x: i32,
    y: i32,
    text: &str,
    style: TextStyle,
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

pub(super) fn offset(origin: i32, distance: u32) -> Result<i32> {
    let distance =
        i32::try_from(distance).map_err(|_| anyhow!("series text offset exceeds i32"))?;
    origin
        .checked_add(distance)
        .ok_or_else(|| anyhow!("series text position overflows"))
}

pub(super) fn rect_contains(rect: Rect, x: f64, y: f64) -> bool {
    if rect.width <= 0 || rect.height <= 0 || !x.is_finite() || !y.is_finite() {
        return false;
    }
    let left = f64::from(rect.x);
    let top = f64::from(rect.y);
    x >= left && y >= top && x < left + f64::from(rect.width) && y < top + f64::from(rect.height)
}
