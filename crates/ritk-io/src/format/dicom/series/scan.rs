//! Directory scanning and series discovery.

use anyhow::{Context, Result};
use arrayvec::ArrayString;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use moirai::prelude::ParallelSlice;
use ritk_dicom::{parse_file_with, DicomRsBackend};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::format::dicom::reader::types::{literal_arraystring, truncate_arraystring};

use super::types::DicomSeriesInfo;

/// Raw per-file data extracted during the parallel scan phase.
type ScannedEntry = (ArrayString<64>, String, ArrayString<16>, String, PathBuf);

pub(crate) fn sort_discovered_series(series_list: &mut [DicomSeriesInfo]) {
    series_list.sort_by(|a, b| {
        a.patient_id
            .cmp(&b.patient_id)
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

    // Collect all file paths first.
    let entries: Vec<PathBuf> = fs::read_dir(path)
        .context("Failed to read directory")?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();

    if entries.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Parallel-collect per-file data (no Mutex during parallel phase).
    let raw: Vec<ScannedEntry> = entries
        .par()
        .map_collect(|file_path| -> anyhow::Result<Option<ScannedEntry>> {
            let Ok(obj) = parse_file_with::<DicomRsBackend, _>(file_path) else {
                return Ok(None);
            };

            let uid_raw = match get_string(&obj, tags::SERIES_INSTANCE_UID) {
                Some(u) => u,
                None => return Ok(None),
            };
            let uid = match ArrayString::<64>::from(uid_raw.trim()) {
                Ok(v) => v,
                Err(_) => {
                    tracing::warn!(
                        "SeriesInstanceUID exceeds 64 chars, truncating: {}",
                        &uid_raw.trim()[..64]
                    );
                    truncate_arraystring::<64>(uid_raw.trim())
                }
            };

            let description = get_string(&obj, tags::SERIES_DESCRIPTION).unwrap_or_default();
            let modality = get_string(&obj, tags::MODALITY)
                .map(|s| {
                    let trimmed = s.trim().to_owned();
                    match ArrayString::<16>::from(trimmed.as_str()) {
                        Ok(v) => v,
                        Err(_) => {
                            tracing::warn!(
                                "Modality exceeds 16 chars, truncating: {}",
                                &trimmed[..16]
                            );
                            truncate_arraystring::<16>(trimmed.as_str())
                        }
                    }
                })
                .unwrap_or_else(|| literal_arraystring("OT"));
            let patient_id = get_string(&obj, tags::PATIENT_ID).unwrap_or_default();

            Ok(Some((
                uid,
                description,
                modality,
                patient_id,
                file_path.clone(),
            )))
        })
        .into_iter()
        .filter_map(|r| r.ok().flatten())
        .collect();

    // 2. Sequential merge — no Mutex required.
    let mut map: HashMap<ArrayString<64>, DicomSeriesInfo> = HashMap::new();
    for (uid, description, modality, patient_id, file_path) in raw {
        map.entry(uid)
            .or_insert_with(|| DicomSeriesInfo {
                series_instance_uid: uid,
                series_description: description,
                modality,
                patient_id,
                file_paths: Vec::new(),
            })
            .file_paths
            .push(file_path);
    }

    let mut series_list: Vec<DicomSeriesInfo> = map.into_values().collect();

    // Sort file paths within each series for determinism.
    for series in &mut series_list {
        series.file_paths.sort();
    }
    sort_discovered_series(&mut series_list);

    Ok(series_list)
}

fn get_string(obj: &FileDicomObject<InMemDicomObject>, tag: dicom::core::Tag) -> Option<String> {
    obj.element(tag).ok()?.to_str().ok().map(|s| s.to_string())
}

#[cfg(test)]
mod tests {
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
                file_paths: vec![PathBuf::from("z/2.dcm")],
            },
            DicomSeriesInfo {
                series_instance_uid: literal_arraystring::<64>("1"),
                series_description: "A".to_owned(),
                modality: literal_arraystring::<16>("CT"),
                patient_id: "P1".to_owned(),
                file_paths: vec![PathBuf::from("a/1.dcm")],
            },
            DicomSeriesInfo {
                series_instance_uid: literal_arraystring::<64>("3"),
                series_description: "A".to_owned(),
                modality: literal_arraystring::<16>("CT"),
                patient_id: "P1".to_owned(),
                file_paths: vec![PathBuf::from("b/1.dcm")],
            },
        ];

        sort_discovered_series(&mut v);

        let uids: Vec<&str> = v.iter().map(|s| s.series_instance_uid.as_str()).collect();
        assert_eq!(uids, vec!["1", "3", "2"]);
    }
}
