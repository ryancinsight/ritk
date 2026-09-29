//! Directory scanning and series discovery.

use crate::format::dicom::reader::dicomdir::discover_files;
use anyhow::{Context, Result};
use arrayvec::ArrayString;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use ritk_dicom::{parse_file_with, DicomRsBackend};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::format::dicom::identity::{image_series_uid, uid_is_valid};
use crate::format::dicom::reader::types::{literal_arraystring, truncate_arraystring};

use super::types::DicomSeriesInfo;

/// Raw per-file data extracted during the parallel scan phase.
struct ScannedEntry {
    series_instance_uid: ArrayString<64>,
    series_description: String,
    modality: ArrayString<16>,
    patient_id: String,
    patient_name: String,
    patient_birth_date: Option<ArrayString<8>>,
    study_instance_uid: Option<ArrayString<64>>,
    study_date: Option<ArrayString<8>>,
    study_time: Option<ArrayString<14>>,
    study_description: String,
    file_path: PathBuf,
}

pub(crate) fn sort_discovered_series(series_list: &mut [DicomSeriesInfo]) {
    series_list.sort_by(|a, b| {
        a.patient_id
            .cmp(&b.patient_id)
            .then_with(|| a.study_instance_uid.cmp(&b.study_instance_uid))
            .then_with(|| a.study_date.cmp(&b.study_date))
            .then_with(|| a.modality.cmp(&b.modality))
            .then_with(|| a.series_description.cmp(&b.series_description))
            .then_with(|| a.series_instance_uid.cmp(&b.series_instance_uid))
            .then_with(|| a.file_paths.first().cmp(&b.file_paths.first()))
    });
}

/// Scan a directory for DICOM series, grouping them by SeriesInstanceUID.
///
/// This function scans the directory in parallel to parse DICOM headers.
pub fn scan_dicom_directory<P: AsRef<Path>>(path: P) -> Result<Vec<DicomSeriesInfo>> {
    let path = path.as_ref();

    let entries = discover_files(path)?;

    if entries.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Parallel-collect per-file data (no Mutex during parallel phase).
    let raw: Vec<ScannedEntry> = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(
        entries.len(),
        |i| -> anyhow::Result<Option<ScannedEntry>> {
            let file_path = &entries[i];
            let obj = parse_file_with::<DicomRsBackend, _>(file_path)
                .context("failed to parse discovered DICOM member")?;
            let Some(uid) = image_series_uid(&obj)? else {
                return Ok(None);
            };

            let description = get_string(&obj, tags::SERIES_DESCRIPTION).unwrap_or_default();
            let modality = get_string(&obj, tags::MODALITY)
                .map(|s| truncate_arraystring::<16>(s.trim()))
                .unwrap_or_else(|| literal_arraystring("OT"));
            let patient_id = get_string(&obj, tags::PATIENT_ID).unwrap_or_default();
            let patient_name = get_string(&obj, tags::PATIENT_NAME).unwrap_or_default();
            let patient_birth_date = get_string(&obj, tags::PATIENT_BIRTH_DATE)
                .and_then(|value| valid_dicom_date(&value));
            let study_instance_uid = get_string(&obj, tags::STUDY_INSTANCE_UID)
                .and_then(|value| valid_study_uid(&value));
            let study_date =
                get_string(&obj, tags::STUDY_DATE).and_then(|value| valid_dicom_date(&value));
            let study_time =
                get_string(&obj, tags::STUDY_TIME).and_then(|value| valid_study_time(&value));
            let study_description = bounded_text(
                get_string(&obj, tags::STUDY_DESCRIPTION).unwrap_or_default(),
                64,
            );

            Ok(Some(ScannedEntry {
                series_instance_uid: uid,
                series_description: description,
                modality,
                patient_id,
                patient_name,
                patient_birth_date,
                study_instance_uid,
                study_date,
                study_time,
                study_description,
                file_path: file_path.clone(),
            }))
        },
    )
    .into_iter()
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .flatten()
    .collect();

    // 2. Sequential merge — no Mutex required.
    let mut map: HashMap<ArrayString<64>, DicomSeriesInfo> = HashMap::new();
    for entry in raw {
        map.entry(entry.series_instance_uid)
            .or_insert_with(|| DicomSeriesInfo {
                series_instance_uid: entry.series_instance_uid,
                series_description: entry.series_description,
                modality: entry.modality,
                patient_id: entry.patient_id,
                patient_name: entry.patient_name,
                patient_birth_date: entry.patient_birth_date,
                study_instance_uid: entry.study_instance_uid,
                study_date: entry.study_date,
                study_time: entry.study_time,
                study_description: entry.study_description,
                file_paths: Vec::new(),
            })
            .file_paths
            .push(entry.file_path);
    }

    let mut series_list: Vec<DicomSeriesInfo> = map.into_values().collect();

    // Sort file paths within each series for determinism.
    for series in &mut series_list {
        series.file_paths.sort();
    }
    sort_discovered_series(&mut series_list);

    Ok(series_list)
}

fn valid_study_uid(value: &str) -> Option<ArrayString<64>> {
    let value = value.trim_end_matches('\0').trim();
    uid_is_valid(value)
        .then(|| ArrayString::from(value).expect("invariant: valid DICOM UID fits 64 bytes"))
}

fn valid_dicom_date(value: &str) -> Option<ArrayString<8>> {
    let value = value.trim_end_matches('\0').trim();
    if value.len() != 8 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }

    let year = value.get(..4)?.parse::<u32>().ok()?;
    let month = value.get(4..6)?.parse::<u32>().ok()?;
    let day = value.get(6..8)?.parse::<u32>().ok()?;
    let leap_year =
        year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let last_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap_year => 29,
        2 => 28,
        _ => return None,
    };
    if year == 0 || day == 0 || day > last_day {
        return None;
    }

    Some(ArrayString::from(value).expect("invariant: validated DICOM date fits its buffer"))
}

fn valid_study_time(value: &str) -> Option<ArrayString<14>> {
    let value = value.trim_end_matches('\0').trim_end();
    if value.len() > 14 || value.is_empty() {
        return None;
    }
    let (clock, fraction) = match value.split_once('.') {
        Some((clock, fraction)) if (1..=6).contains(&fraction.len()) => (clock, Some(fraction)),
        Some(_) => return None,
        None => (value, None),
    };
    if !clock.bytes().all(|byte| byte.is_ascii_digit())
        || fraction.is_some_and(|digits| !digits.bytes().all(|byte| byte.is_ascii_digit()))
        || !matches!(clock.len(), 2 | 4 | 6)
        || fraction.is_some() && clock.len() != 6
    {
        return None;
    }
    let hour = clock.get(..2)?.parse::<u32>().ok()?;
    let minute = if clock.len() >= 4 {
        Some(clock.get(2..4)?.parse::<u32>().ok()?)
    } else {
        None
    };
    let second = if clock.len() == 6 {
        Some(clock.get(4..6)?.parse::<u32>().ok()?)
    } else {
        None
    };
    if hour > 23 || minute.is_some_and(|value| value > 59) || second.is_some_and(|value| value > 60)
    {
        return None;
    }
    Some(ArrayString::from(value).expect("invariant: validated DICOM time fits its buffer"))
}

fn bounded_text(value: String, maximum_characters: usize) -> String {
    value
        .trim_end_matches('\0')
        .chars()
        .filter(|character| !character.is_control())
        .take(maximum_characters)
        .collect()
}

fn get_string(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<String> {
    obj.element(tag).ok()?.to_str().ok().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;
    use crate::format::dicom::reader::types::literal_arraystring;
    use std::path::PathBuf;

    #[test]
    fn test_scan_empty_dir() {
        let temp = tempfile::tempdir().unwrap();
        let series = scan_dicom_directory(temp.path()).unwrap();
        assert!(series.is_empty());
    }

    #[test]
    fn discovered_series_sort_is_deterministic() {
        let mut v = vec![
            DicomSeriesInfo {
                series_instance_uid: literal_arraystring::<64>("2"),
                series_description: "B".to_owned(),
                modality: literal_arraystring::<16>("MR"),
                patient_id: "P2".to_owned(),
                patient_name: "Patient Two".to_owned(),
                study_instance_uid: Some(ArrayString::from("2.25.2").expect("study uid")),
                study_date: Some(ArrayString::from("20260102").expect("study date")),
                patient_birth_date: None,
                study_time: None,
                study_description: String::new(),
                file_paths: vec![PathBuf::from("z/2.dcm")],
            },
            DicomSeriesInfo {
                series_instance_uid: literal_arraystring::<64>("1"),
                series_description: "A".to_owned(),
                modality: literal_arraystring::<16>("CT"),
                patient_id: "P1".to_owned(),
                patient_name: "Patient One".to_owned(),
                study_instance_uid: Some(ArrayString::from("2.25.1").expect("study uid")),
                study_date: Some(ArrayString::from("20260101").expect("study date")),
                patient_birth_date: None,
                study_time: None,
                study_description: String::new(),
                file_paths: vec![PathBuf::from("a/1.dcm")],
            },
            DicomSeriesInfo {
                series_instance_uid: literal_arraystring::<64>("3"),
                series_description: "A".to_owned(),
                modality: literal_arraystring::<16>("CT"),
                patient_id: "P1".to_owned(),
                patient_name: "Patient One".to_owned(),
                study_instance_uid: Some(ArrayString::from("2.25.1").expect("study uid")),
                study_date: Some(ArrayString::from("20260101").expect("study date")),
                patient_birth_date: None,
                study_time: None,
                study_description: String::new(),
                file_paths: vec![PathBuf::from("b/1.dcm")],
            },
        ];

        sort_discovered_series(&mut v);

        let uids: Vec<&str> = v.iter().map(|s| s.series_instance_uid.as_str()).collect();
        assert_eq!(uids, vec!["1", "3", "2"]);
    }

    #[test]
    fn study_metadata_validation_preserves_only_grouping_safe_values() {
        assert_eq!(
            valid_study_uid("2.25.123\0"),
            Some(ArrayString::from("2.25.123").expect("uid"))
        );
        assert_eq!(valid_study_uid("2.25.01"), None);
        assert_eq!(valid_study_uid(&"1".repeat(65)), None);
        assert_eq!(
            valid_dicom_date("20260927"),
            Some(ArrayString::from("20260927").expect("date"))
        );
        assert_eq!(
            valid_dicom_date("20240229"),
            Some(ArrayString::from("20240229").expect("leap date"))
        );
        assert_eq!(valid_dicom_date("2026927"), None);
        assert_eq!(valid_dicom_date("2026-09-27"), None);
        assert_eq!(valid_dicom_date("20261301"), None);
        assert_eq!(valid_dicom_date("20260229"), None);
        assert_eq!(valid_dicom_date("19900191"), None);
    }

    #[test]
    fn study_time_validation_accepts_partial_and_fractional_times() {
        assert_eq!(valid_study_time("0000 ").as_deref(), Some("0000"));
        assert_eq!(valid_study_time("1010").as_deref(), Some("1010"));
        assert_eq!(
            valid_study_time("070907.0705").as_deref(),
            Some("070907.0705")
        );
        assert_eq!(valid_study_time("235960").as_deref(), Some("235960"));
        assert_eq!(valid_study_time("021"), None);
        assert_eq!(valid_study_time("2400"), None);
        assert_eq!(valid_study_time("1260"), None);
        assert_eq!(valid_study_time("126061"), None);
        assert_eq!(valid_study_time("1234.1"), None);
        assert_eq!(valid_study_time("123456.1234567"), None);
    }
}
