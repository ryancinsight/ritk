use anyhow::{bail, Context, Result};
use dicom::core::Tag;
use std::path::{Path, PathBuf};

use ritk_dicom::{parse_file_with, DicomRsBackend};

/// Attempt to read file paths from a DICOMDIR file in the given directory.
/// Returns Ok(Vec<PathBuf>) with resolved absolute paths, or Err if DICOMDIR
/// is absent or cannot be parsed. Paths are verified to exist as files.
pub(super) fn try_read_dicomdir(dir: &Path) -> Result<Vec<PathBuf>> {
    let dicomdir_path = dir.join("DICOMDIR");

    if !dicomdir_path.is_file() {
        bail!("no DICOMDIR found");
    }

    let obj = parse_file_with::<DicomRsBackend, _>(&dicomdir_path)
        .with_context(|| format!("failed to open DICOMDIR {:?}", dicomdir_path))?;

    // DirectoryRecordSequence (0004,1220) contains all records as a flat SQ.
    let drs = obj
        .element(Tag(0x0004, 0x1220))
        .with_context(|| "DICOMDIR missing DirectoryRecordSequence (0004,1220)")?;

    let mut paths = if let Some(items) = drs.value().items() {
        Vec::with_capacity(items.len())
    } else {
        Vec::new()
    };

    if let Some(items) = drs.value().items() {
        for item in items {
            // Only process IMAGE-type directory records (excludes PATIENT, STUDY, SERIES, etc.)
            let record_type = item
                .element(Tag(0x0004, 0x1430))
                .ok()
                .and_then(|e| e.to_str().ok().map(|s| s.trim().to_uppercase()));

            if record_type.as_deref() != Some("IMAGE") {
                continue;
            }

            // ReferencedFileID (0004,1500): multi-value CS, components separated by '\'.
            if let Ok(file_id_elem) = item.element(Tag(0x0004, 0x1500)) {
                if let Ok(s) = file_id_elem.to_str() {
                    let s = s.trim();
                    if s.is_empty() {
                        continue;
                    }
                    // Components separated by '\' in DICOM CS multi-value convention.
                    let mut file_path = dir.to_path_buf();
                    for component in s.split('\\') {
                        let comp = component.trim();
                        if !comp.is_empty() {
                            file_path.push(comp);
                        }
                    }
                    if file_path.is_file() {
                        paths.push(file_path);
                    }
                }
            }
        }
    }

    if paths.is_empty() {
        bail!("DICOMDIR contained no valid ReferencedFileID entries");
    }

    Ok(paths)
}
