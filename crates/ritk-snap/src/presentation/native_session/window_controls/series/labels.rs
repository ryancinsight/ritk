//! Validated patient and study labels for the series navigator.

use super::super::super::series_browser::SeriesChoice;
use anyhow::{anyhow, Result};
use arrayvec::ArrayString;
use std::fmt::Write as _;

pub(in crate::presentation::native_session::window_controls) fn patient_label(
    choice: &SeriesChoice,
) -> Result<ArrayString<96>> {
    let name = display_patient_name(&choice.patient_name);
    let mut label = ArrayString::new();
    if name.as_str() == "Unknown patient" {
        write!(&mut label, "Patient {}", choice.patient_number)
            .map_err(|_| anyhow!("patient label exceeds its display buffer"))?;
    } else {
        label
            .try_push_str(name.as_str())
            .map_err(|_| anyhow!("patient label exceeds its display buffer"))?;
    }
    Ok(label)
}

pub(in crate::presentation::native_session::window_controls) fn study_label(
    choice: &SeriesChoice,
) -> Result<ArrayString<320>> {
    let mut label = ArrayString::new();
    if let Some(date) = choice.study_date.as_deref().and_then(format_dicom_date) {
        write!(&mut label, "{date}")
            .map_err(|_| anyhow!("study date exceeds its display buffer"))?;
        if let Some(time) = choice.series_time.as_deref().and_then(format_dicom_time) {
            write!(&mut label, " {time}")
                .map_err(|_| anyhow!("study time exceeds its display buffer"))?;
        }
    }
    if label.is_empty() {
        write!(&mut label, "Study {}", choice.study_number)
            .map_err(|_| anyhow!("study label exceeds its display buffer"))?;
    }
    Ok(label)
}

pub(in crate::presentation::native_session::window_controls) fn format_dicom_date(
    value: &str,
) -> Option<ArrayString<10>> {
    if value.len() != 8 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    let year = value.get(..4)?;
    let month = value.get(4..6)?;
    let day = value.get(6..)?;
    let mut display = ArrayString::new();
    write!(&mut display, "{year}-{month}-{day}").ok()?;
    Some(display)
}

pub(in crate::presentation::native_session::window_controls) fn format_dicom_time(
    value: &str,
) -> Option<ArrayString<16>> {
    let mut display = ArrayString::new();
    for (index, character) in value.char_indices() {
        if index == 2 || index == 4 {
            display.try_push(':').ok()?;
        }
        display.try_push(character).ok()?;
    }
    Some(display)
}

pub(in crate::presentation::native_session::window_controls) fn display_patient_name(
    name: &str,
) -> ArrayString<64> {
    let mut display = ArrayString::new();
    for character in name.trim().chars() {
        let normalized = if character == '^' { ' ' } else { character };
        if (normalized.is_control() || display.try_push(normalized).is_err()) && display.is_full() {
            break;
        }
    }
    if display.is_empty() {
        ArrayString::from("Unknown patient").expect("static label fits")
    } else {
        display
    }
}

#[cfg(test)]
mod tests {
    use super::format_dicom_date;

    #[test]
    fn dicom_date_formatting_preserves_the_validated_dicom_date() {
        assert_eq!(
            format_dicom_date("20240229")
                .expect("scanner-provided Gregorian date has eight digits")
                .as_str(),
            "2024-02-29"
        );
    }

    #[test]
    fn dicom_date_formatting_rejects_values_outside_its_storage_shape() {
        assert_eq!(format_dicom_date("2024011"), None);
        assert_eq!(format_dicom_date("20240A01"), None);
    }
}
