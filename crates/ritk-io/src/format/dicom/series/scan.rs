//! Directory scanning and series discovery.

use crate::format::dicom::reader::dicomdir::discover_files_with_budget;
use anyhow::{Context, Result};
use arrayvec::ArrayString;
use dicom::dictionary_std::tags;
use dicom::object::{FileDicomObject, InMemDicomObject};
use ritk_dicom::{parse_bytes_with_budget, read_file_with_budget, DicomRsBackend};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::format::dicom::identity::image_series_uid;
use crate::format::dicom::reader::types::{literal_arraystring, truncate_arraystring};
use crate::format::dicom::reader::DicomReadBudget;

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
    scan_dicom_directory_with_budget(path, &DicomReadBudget::DEFAULT)
}

pub(super) fn scan_dicom_directory_with_budget<P: AsRef<Path>>(
    path: P,
    budget: &DicomReadBudget,
) -> Result<Vec<DicomSeriesInfo>> {
    let path = path.as_ref();
    let parser_budget = budget.parser();
    let entries = discover_files_with_budget(path, &parser_budget)?;

    if entries.is_empty() {
        return Ok(Vec::new());
    }
    budget
        .checked_instances(entries.len())
        .context("DICOM catalog candidate count exceeds budget")?;
    let declared_bytes = entries.iter().try_fold(0_usize, |total, entry| {
        let declared = std::fs::metadata(entry)
            .with_context(|| format!("failed to inspect discovered DICOM member {entry:?}"))?
            .len();
        let member_bytes = parser_budget
            .checked_bytes(declared, "DICOM catalog member")
            .context("discovered DICOM member exceeds parse budget")?;
        total
            .checked_add(member_bytes)
            .context("DICOM catalog encoded byte total overflow")
    })?;
    budget
        .checked_retained_bytes(declared_bytes)
        .context("DICOM catalog encoded bytes exceed budget")?;

    let mut encoded = Vec::new();
    let mut encoded_bytes = 0_usize;
    for entry in entries {
        let bytes = read_file_with_budget(&entry, &parser_budget)
            .context("failed to read discovered DICOM member")?;
        encoded_bytes = encoded_bytes
            .checked_add(bytes.len())
            .context("DICOM catalog encoded byte total overflow")?;
        budget
            .checked_retained_bytes(encoded_bytes)
            .context("DICOM catalog encoded bytes exceed budget")?;
        encoded
            .try_reserve(1)
            .context("failed to reserve DICOM catalog storage")?;
        encoded.push((entry, bytes));
    }

    // 1. Parallel-collect per-file data (no Mutex during parallel phase).
    let raw: Vec<ScannedEntry> = moirai::map_collect_index_with::<moirai::Adaptive, _, _>(
        encoded.len(),
        |i| -> anyhow::Result<Option<ScannedEntry>> {
            let (file_path, bytes) = &encoded[i];
            let obj = parse_bytes_with_budget::<DicomRsBackend>(bytes, &parser_budget)
                .context("failed to parse discovered DICOM member")?;
            let Some(uid) = image_series_uid(&obj)? else {
                return Ok(None);
            };

            let description = get_string(&obj, tags::SERIES_DESCRIPTION).unwrap_or_default();
            let modality = get_string(&obj, tags::MODALITY)
                .map(|s| truncate_arraystring::<16>(s.trim()))
                .unwrap_or_else(|| literal_arraystring("OT"));
            let patient_id = get_string(&obj, tags::PATIENT_ID).unwrap_or_default();

            Ok(Some((
                uid,
                description,
                modality,
                patient_id,
                file_path.clone(),
            )))
        },
    )
    .into_iter()
    .collect::<Result<Vec<_>>>()?
    .into_iter()
    .flatten()
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
    #![expect(clippy::unwrap_used, reason = "ratchet RITK-UNWRAP-1")]
    use super::*;
    use crate::format::dicom::reader::types::literal_arraystring;
    use ritk_dicom::ParseBudget;
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

    #[test]
    fn catalog_bounds_reject_malformed_candidates_before_parsing() {
        let root = tempfile::tempdir().unwrap();
        let count = root.path().join("count");
        std::fs::create_dir(&count).unwrap();
        std::fs::write(count.join("first.dcm"), [0_u8; 16]).unwrap();
        std::fs::write(count.join("second.dcm"), [0_u8; 16]).unwrap();
        let count_budget =
            DicomReadBudget::try_new_with_max_instances(ParseBudget::new(64, 64, 8), 64, 64, 1)
                .unwrap();
        let count_error = scan_dicom_directory_with_budget(&count, &count_budget).unwrap_err();
        assert!(format!("{count_error:#}").contains("candidate count exceeds budget"));

        let bytes = root.path().join("bytes");
        std::fs::create_dir(&bytes).unwrap();
        std::fs::write(bytes.join("oversized.dcm"), [0_u8; 256]).unwrap();
        let byte_budget =
            DicomReadBudget::try_new_with_max_instances(ParseBudget::new(512, 64, 8), 255, 512, 1)
                .unwrap();
        let byte_error = scan_dicom_directory_with_budget(&bytes, &byte_budget).unwrap_err();
        assert!(format!("{byte_error:#}").contains("catalog encoded bytes exceed budget"));

        let exact_budget =
            DicomReadBudget::try_new_with_max_instances(ParseBudget::new(512, 64, 8), 256, 512, 1)
                .unwrap();
        let exact_error = scan_dicom_directory_with_budget(&bytes, &exact_budget).unwrap_err();
        assert!(format!("{exact_error:#}").contains("failed to parse discovered DICOM member"));
    }
}
