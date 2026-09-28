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
    study_instance_uid: Option<ArrayString<64>>,
    study_date: Option<ArrayString<8>>,
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
            let study_instance_uid = get_string(&obj, tags::STUDY_INSTANCE_UID)
                .and_then(|value| valid_study_uid(&value));
            let study_date =
                get_string(&obj, tags::STUDY_DATE).and_then(|value| valid_study_date(&value));

            Ok(Some(ScannedEntry {
                series_instance_uid: uid,
                series_description: description,
                modality,
                patient_id,
                patient_name,
                study_instance_uid,
                study_date,
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
                study_instance_uid: entry.study_instance_uid,
                study_date: entry.study_date,
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

fn valid_study_date(value: &str) -> Option<ArrayString<8>> {
    let value = value.trim_end_matches('\0').trim();
    (value.len() == 8 && value.bytes().all(|byte| byte.is_ascii_digit())).then(|| {
        ArrayString::from(value).expect("invariant: eight-digit DICOM date fits its buffer")
    })
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
            valid_study_date("20260927"),
            Some(ArrayString::from("20260927").expect("date"))
        );
        assert_eq!(valid_study_date("2026927"), None);
        assert_eq!(valid_study_date("2026-09-27"), None);
    }
}
