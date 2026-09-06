//! DICOM discovery preserves authoritative file-set indexes.
use crate::dicom::input_path::classify_dicom_input_path;
use crate::dicom::series_tree::{SeriesEntry, SeriesEntryView, SeriesTree};
use anyhow::{Context, Result};
use ritk_io::scan_dicom_directory;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// Discover acquisitions beneath a directory, stopping at each DICOMDIR boundary.
///
/// Traversal is deterministic and bounded to five directory levels. An index
/// owns its file set: unreferenced descendants are never scanned independently.
///
/// # Errors
/// Propagates unreadable directories, invalid DICOM instances, and malformed or
/// missing DICOMDIR references; failed discovery never yields a partial tree.
pub fn scan_folder_for_series<P: AsRef<Path>>(folder: P) -> Result<SeriesTree<'static>> {
    let requested = folder.as_ref();
    let mut entries: Vec<SeriesEntry> = Vec::new();
    if is_index(requested) && !requested.is_dir() {
        entries.extend(
            scan_dicom_directory(requested)?
                .into_iter()
                .map(SeriesEntry::from_dicom_series_info),
        );
    } else {
        let root = classify_dicom_input_path(requested)
            .dicom_root()
            .unwrap_or(requested)
            .to_path_buf();
        let mut directories = WalkDir::new(root)
            .max_depth(5)
            .follow_links(false)
            .sort_by_file_name()
            .into_iter();
        while let Some(entry) = directories.next() {
            let entry = entry.context("failed to traverse DICOM discovery directory")?;
            if !entry.file_type().is_dir() {
                continue;
            }
            let children: Vec<PathBuf> = std::fs::read_dir(entry.path())
                .context("failed to inspect DICOM discovery directory")?
                .map(|child| child.map(|child| child.path()))
                .collect::<std::io::Result<_>>()?;
            let indexed = children.iter().any(|child| is_index(child));
            entries.extend(
                scan_dicom_directory(entry.path())?
                    .into_iter()
                    .map(SeriesEntry::from_dicom_series_info),
            );
            if indexed {
                directories.skip_current_dir();
            }
        }
    }
    sort_series_entries_deterministically(&mut entries);
    Ok(SeriesTree::from_entries(entries))
}

fn is_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("DICOMDIR"))
}

/// Sort series entries by a deterministic multi-key order.
///
/// Key precedence: `patient_id` → `study_uid` → `study_date` → `modality`
/// → `series_description` → `series_uid` → `folder` path string.
pub(super) fn sort_series_entries_deterministically(entries: &mut [SeriesEntry]) {
    entries.sort_by(|a, b| {
        a.acquisition
            .patient_id
            .cmp(&b.acquisition.patient_id)
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
            .then_with(|| a.modality().cmp(b.modality()))
            .then_with(|| a.series_description().cmp(b.series_description()))
            .then_with(|| a.series_uid().cmp(b.series_uid()))
            .then_with(|| a.folder().cmp(b.folder()))
    });
}
