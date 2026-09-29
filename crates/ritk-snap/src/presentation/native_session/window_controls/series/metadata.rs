//! Patient and study information in the series preview rail.

use super::super::super::layout::text_style;
use super::super::super::series_browser::SeriesChoice;
use super::{
    display_patient_name, draw_fit, format_dicom_date, format_dicom_time, MUTED, STUDY_TEXT, TEXT,
};
use anyhow::{anyhow, Result};
use arrayvec::ArrayString;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};
use std::fmt::Write as _;

const STUDY_INFO_BACKGROUND: Color = Color::rgb(27, 45, 59);
const STUDY_INFO_EDGE: Color = Color::rgb(91, 151, 184);
const STUDY_DETAILS_BUFFER_CAPACITY: usize = 16 + 2 + decimal_width(usize::MAX) + 12 + 64 * 4;

const fn decimal_width(mut value: usize) -> usize {
    let mut digits = 1;
    while value >= 10 {
        value /= 10;
        digits += 1;
    }
    digits
}

pub(super) fn render_study_metadata(
    framebuffer: &mut Framebuffer,
    area: Rect,
    choice: Option<&SeriesChoice>,
    study_count: usize,
) -> Result<()> {
    let Some(choice) = choice else {
        return Ok(());
    };
    let left = area.x.saturating_add(8);
    let width = area.width.saturating_sub(20);
    let patient_y = area.y.saturating_add(54);
    let study_y = patient_y.saturating_add(54);
    for (y, height) in [(patient_y, 48), (study_y, 72)] {
        let rect = Rect::new(left, y, width, height);
        fill_rect(
            framebuffer,
            rect,
            CornerRadius::SQUARE,
            STUDY_INFO_BACKGROUND,
        );
        fill_rect(
            framebuffer,
            Rect::new(rect.x, rect.y, 2, rect.height),
            CornerRadius::SQUARE,
            STUDY_INFO_EDGE,
        );
    }

    let label_style = text_style(MUTED, 10)?;
    let value_style = text_style(TEXT, 12)?;
    let detail_style = text_style(STUDY_TEXT, 10)?;
    let text_x = left.saturating_add(8);
    let text_width = width.saturating_sub(16);
    draw_fit(
        framebuffer,
        text_x,
        patient_y.saturating_add(6),
        "Patient",
        label_style,
        text_width,
    );
    let patient_name = display_patient_name(choice.acquisition.patient_name());
    draw_fit(
        framebuffer,
        text_x,
        patient_y.saturating_add(21),
        patient_name.as_str(),
        value_style,
        text_width,
    );
    let birth_date_value = choice
        .acquisition
        .patient_birth_date()
        .and_then(format_dicom_date)
        .unwrap_or_else(|| ArrayString::from("DOB N/A").expect("static label fits"));
    let mut birth_date = ArrayString::<32>::new();
    write!(&mut birth_date, "DOB  {}", birth_date_value)
        .map_err(|_| anyhow!("patient birth date exceeds its display buffer"))?;
    draw_fit(
        framebuffer,
        text_x,
        patient_y.saturating_add(36),
        birth_date.as_str(),
        detail_style,
        text_width,
    );

    let mut study_title = ArrayString::<32>::new();
    write!(
        &mut study_title,
        "Study {}/{}",
        choice.study_number, study_count
    )
    .map_err(|_| anyhow!("study position exceeds its display buffer"))?;
    draw_fit(
        framebuffer,
        text_x,
        study_y.saturating_add(6),
        study_title.as_str(),
        label_style,
        text_width,
    );
    let date = choice
        .acquisition
        .study_date()
        .and_then(format_dicom_date)
        .unwrap_or_else(|| ArrayString::from("Date N/A").expect("static label fits"));
    let time = choice.acquisition.study_time().and_then(format_dicom_time);
    let mut date_time = ArrayString::<32>::new();
    write!(&mut date_time, "{}", date)
        .map_err(|_| anyhow!("study date exceeds its display buffer"))?;
    if let Some(time) = time {
        write!(&mut date_time, "  {}", time)
            .map_err(|_| anyhow!("study time exceeds its display buffer"))?;
    }
    draw_fit(
        framebuffer,
        text_x,
        study_y.saturating_add(25),
        date_time.as_str(),
        value_style,
        text_width,
    );
    let study_details = format_study_details(
        choice.modality.as_ref(),
        choice.study_series_count,
        choice.acquisition.study_description().trim(),
    )?;
    draw_fit(
        framebuffer,
        text_x,
        study_y.saturating_add(45),
        study_details.as_str(),
        detail_style,
        text_width,
    );
    Ok(())
}

fn format_study_details(
    modality: &str,
    study_series_count: usize,
    study_description: &str,
) -> Result<ArrayString<STUDY_DETAILS_BUFFER_CAPACITY>> {
    let mut details = ArrayString::new();
    if study_description.is_empty() {
        write!(&mut details, "{}: {} series", modality, study_series_count)
    } else {
        write!(
            &mut details,
            "{}: {} series  |  {}",
            modality, study_series_count, study_description
        )
    }
    .map_err(|_| anyhow!("study details exceed their DICOM metadata buffer"))?;
    Ok(details)
}

#[cfg(test)]
mod tests {
    use super::{format_study_details, STUDY_DETAILS_BUFFER_CAPACITY};

    #[test]
    fn study_details_fit_maximum_dicom_metadata_and_series_count() {
        let description = "😀".repeat(64);
        let details = format_study_details("1234567890123456", usize::MAX, &description)
            .expect("bounded DICOM metadata fits the study label");

        assert_eq!(
            details.as_str(),
            format!("1234567890123456: {} series  |  {description}", usize::MAX)
        );
        assert_eq!(details.len(), STUDY_DETAILS_BUFFER_CAPACITY);
    }
}
