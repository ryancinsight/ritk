//! Directory scanning for DICOM series and deterministic sorting.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::Result;
use ritk_io::scan_dicom_directory;
use tracing::{info, warn};
use walkdir::WalkDir;

use crate::dicom::input_path::classify_dicom_input_path;
use crate::dicom::series_tree::{SeriesEntry, SeriesTree};

/// Walk `folder` and its immediate subdirectories, attempting to scan each
/// directory for DICOM files.
///
/// # Algorithm
/// 1. Try `scan_dicom_directory(folder)` first; add the result when successful.
/// 2. Walk all subdirectories up to depth 5. For each subdirectory, try
///    `scan_dicom_directory`; skip silently on failure.
/// 3. Deduplicate by folder path so multi-level discovery never double-counts.
/// 4. Build and return a [`SeriesTree`] from the collected [`SeriesEntry`] list.
///
/// This heuristic covers both flat DICOM folders and patient/study/series
/// hierarchies without requiring a DICOMDIR index file.
///
/// # Errors
/// Returns an error only when `folder` itself cannot be read as a directory.
pub fn scan_folder_for_series<P: AsRef<Path>>(folder: P) -> Result<SeriesTree<'static>> {
    let requested = folder.as_ref();
    let folder = classify_dicom_input_path(requested)
        .dicom_root()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| requested.to_path_buf());
    info!(path = %folder.display(), "scanning folder for DICOM series");

    let mut entries: Vec<SeriesEntry> = Vec::new();
    let mut seen_folders: HashSet<PathBuf> = HashSet::new();

    // Try to scan `dir` for DICOM content and append to `entries` if not
    // already seen.
    let try_add = |dir: &Path, entries: &mut Vec<SeriesEntry>, seen: &mut HashSet<PathBuf>| {
        let canonical = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());
        if seen.contains(&canonical) {
            return;
        }
        seen.insert(canonical);
        match scan_dicom_directory(dir) {
            Ok(series_list) => {
                entries.extend(
                    series_list
                        .into_iter()
                        .map(SeriesEntry::from_dicom_series_info),
                );
            }
            Err(e) => {
                warn!(path = %dir.display(), error = %e, "skipping directory (not a DICOM series)");
            }
        }
    };

    // Scan the root folder itself.
    try_add(&folder, &mut entries, &mut seen_folders);

    // Walk subdirectories up to depth 5 with deterministic lexical ordering.
    let mut subdirs: Vec<PathBuf> = WalkDir::new(&folder)
        .min_depth(1)
        .max_depth(5)
        .follow_links(false)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_dir())
        .map(|e| e.path().to_path_buf())
        .collect();
    subdirs.sort();

    for dir in subdirs {
        try_add(&dir, &mut entries, &mut seen_folders);
    }

    sort_series_entries_deterministically(&mut entries);

    info!(
        root = %folder.display(),
        series_found = entries.len(),
        "scan complete"
    );
    Ok(SeriesTree::from_entries(entries))
}

/// Sort series entries by a deterministic multi-key order.
///
/// Key precedence: `patient_id` → `study_uid` → `study_date` → `modality`
/// → `series_description` → `series_uid` → `folder` path string.
pub(super) fn sort_series_entries_deterministically(entries: &mut [SeriesEntry]) {
    entries.sort_by(|a, b| {
        a.patient_id
            .cmp(&b.patient_id)
            .then_with(|| {
                a.study_uid
                    .as_deref()
                    .unwrap_or("")
                    .cmp(b.study_uid.as_deref().unwrap_or(""))
            })
            .then_with(|| {
                a.study_date
                    .as_deref()
                    .unwrap_or("")
                    .cmp(b.study_date.as_deref().unwrap_or(""))
            })
            .then_with(|| a.modality.cmp(&b.modality))
            .then_with(|| a.series_description.cmp(&b.series_description))
            .then_with(|| a.series_uid.cmp(&b.series_uid))
            .then_with(|| a.folder.cmp(&b.folder))
    });
}
