//! DICOM discovery preserves authoritative file-set indexes.
use crate::dicom::input_path::classify_dicom_input_path;
use crate::dicom::series_tree::{DiscoveredSeries, SeriesTree};
use anyhow::{Context, Result};
use ritk_io::{scan_dicom_files, DicomReadMetadata, DicomSeriesInfo, DicomStudyCatalog};
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
    let mut entries = Vec::new();
    if is_index(requested) && !requested.is_dir() {
        entries.extend(entries_with_metadata(
            DicomStudyCatalog::scan(requested)?.into_series(),
        )?);
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
            entries.extend(entries_with_metadata(
                DicomStudyCatalog::scan(entry.path())?.into_series(),
            )?);
            if indexed {
                directories.skip_current_dir();
            }
        }
    }
    Ok(SeriesTree::from_discovery(entries))
}

fn entries_with_metadata(series: Vec<DicomSeriesInfo>) -> Result<Vec<DiscoveredSeries>> {
    series
        .into_iter()
        .map(|info| {
            let metadata = first_instance_metadata(&info)?;
            Ok(DiscoveredSeries::new(info, &metadata))
        })
        .collect()
}

fn first_instance_metadata(info: &DicomSeriesInfo) -> Result<DicomReadMetadata> {
    let first = info
        .file_paths
        .first()
        .context("discovered DICOM series has no instances")?;
    scan_dicom_files(std::slice::from_ref(first))
        .with_context(|| format!("read DICOM display metadata from '{}'", first.display()))
        .map(|series| series.metadata)
}

fn is_index(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("DICOMDIR"))
}
