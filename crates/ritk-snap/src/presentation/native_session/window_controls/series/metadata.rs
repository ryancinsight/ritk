//! Patient and study information attached to each study's first series card.

use super::super::super::series_browser::SeriesChoice;
use super::super::series::draw_fit;
use super::{format_dicom_date, format_dicom_time, patient_label};
use crate::presentation::native_session::layout::TextStyle;
use anyhow::{anyhow, Result};
use arrayvec::ArrayString;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};
use std::fmt::Write as _;

const STUDY_HEADER_BACKGROUND: Color = Color::rgb(27, 45, 59);
const STUDY_DETAILS_BACKGROUND: Color = Color::rgb(38, 54, 67);
const STUDY_HEADER_EDGE: Color = Color::rgb(91, 151, 184);
const STUDY_LABEL_CAPACITY: usize = 80;

pub(super) fn render_study_metadata(
    framebuffer: &mut Framebuffer,
    card: Rect,
    choice: &SeriesChoice,
    study_count: usize,
    detail_style: TextStyle,
) -> Result<()> {
    let patient = patient_label(choice)?;
    let identity = study_identity_label(choice, study_count)?;
    let series_summary = study_series_summary_label(choice)?;
    let patient_box = Rect::new(
        card.x.saturating_add(2),
        card.y.saturating_add(2),
        card.width.saturating_sub(4),
        18,
    );
    let study_box = Rect::new(
        patient_box.x,
        patient_box.y.saturating_add(20),
        patient_box.width,
        30,
    );
    fill_rect(
        framebuffer,
        patient_box,
        CornerRadius::SQUARE,
        STUDY_HEADER_BACKGROUND,
    );
    fill_rect(
        framebuffer,
        study_box,
        CornerRadius::SQUARE,
        STUDY_DETAILS_BACKGROUND,
    );
    for rect in [patient_box, study_box] {
        fill_rect(
            framebuffer,
            Rect::new(rect.x, rect.y, 2, rect.height),
            CornerRadius::SQUARE,
            STUDY_HEADER_EDGE,
        );
    }

    let text_x = patient_box.x.saturating_add(8);
    let text_width = patient_box.width.saturating_sub(16);
    draw_fit(
        framebuffer,
        text_x,
        patient_box.y.saturating_add(3),
        patient.as_str(),
        detail_style,
        text_width,
    );
    draw_fit(
        framebuffer,
        text_x,
        study_box.y.saturating_add(2),
        identity.as_str(),
        detail_style,
        text_width,
    );
    let summary_width = detail_style
        .extent(0, 0, series_summary.as_str())
        .map_or(0, |rect| rect.width)
        .min(text_width);
    let description_width = text_width.saturating_sub(summary_width.saturating_add(8));
    let study_description = choice.acquisition.study_description().trim();
    draw_fit(
        framebuffer,
        text_x,
        study_box.y.saturating_add(15),
        if study_description.is_empty() {
            "No study description"
        } else {
            study_description
        },
        detail_style,
        description_width,
    );
    draw_fit(
        framebuffer,
        study_box
            .x
            .saturating_add(study_box.width)
            .saturating_sub(summary_width)
            .saturating_sub(8),
        study_box.y.saturating_add(15),
        series_summary.as_str(),
        detail_style,
        summary_width,
    );
    Ok(())
}

fn study_identity_label(
    choice: &SeriesChoice,
    study_count: usize,
) -> Result<ArrayString<STUDY_LABEL_CAPACITY>> {
    let date = choice
        .acquisition
        .study_date()
        .and_then(format_dicom_date)
        .unwrap_or_else(|| ArrayString::from("Date N/A").expect("static label fits"));
    let time = choice.acquisition.study_time().and_then(format_dicom_time);
    let mut label = ArrayString::new();
    write!(
        &mut label,
        "Study {}/{}  ·  {}",
        choice.study_number, study_count, date
    )
    .map_err(|_| anyhow!("study identity label exceeds its display buffer"))?;
    if let Some(time) = time {
        write!(&mut label, " {time}")
            .map_err(|_| anyhow!("study identity label exceeds its display buffer"))?;
    }
    Ok(label)
}

fn study_series_summary_label(choice: &SeriesChoice) -> Result<ArrayString<STUDY_LABEL_CAPACITY>> {
    let mut label = ArrayString::new();
    write!(
        &mut label,
        "{}  ·  {} series",
        choice.modality, choice.study_series_count
    )
    .map_err(|_| anyhow!("study series summary exceeds its display buffer"))?;
    Ok(label)
}
