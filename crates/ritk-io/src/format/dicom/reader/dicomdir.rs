//! Authoritative DICOM file-set discovery shared by browsing and loading.

use anyhow::{bail, Context, Result};
use dicom::core::Tag;
use std::path::{Component, Path, PathBuf};

use ritk_dicom::{parse_file_with, DicomRsBackend};

use super::detection::is_likely_dicom_file;

pub(super) fn is_dicomdir(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.eq_ignore_ascii_case("DICOMDIR"))
}

/// Resolve a directory or an explicitly selected index to its exact file set.
/// An existing index is authoritative: malformed or missing references fail.
pub(in crate::format::dicom) fn discover_files(path: &Path) -> Result<Vec<PathBuf>> {
    if is_dicomdir(path) && !path.is_dir() {
        return read_dicomdir(path);
    }
    let entries = std::fs::read_dir(path)
        .context("failed to read DICOM directory")?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    let indexes: Vec<_> = entries.iter().filter(|entry| is_dicomdir(entry)).collect();
    match indexes.as_slice() {
        [] => {
            let mut paths: Vec<_> = entries
                .into_iter()
                .filter(|entry| entry.is_file() && is_likely_dicom_file(entry))
                .collect();
            paths.sort();
            Ok(paths)
        }
        [index] => read_dicomdir(index),
        _ => bail!("multiple DICOMDIR indexes in one directory"),
    }
}

fn read_dicomdir(index: &Path) -> Result<Vec<PathBuf>> {
    let root = index
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
        .canonicalize()
        .context("failed to resolve DICOMDIR root")?;
    let obj = parse_file_with::<DicomRsBackend, _>(index).context("failed to open DICOMDIR")?;
    let sequence = obj
        .element(Tag(0x0004, 0x1220))
        .context("DICOMDIR missing DirectoryRecordSequence (0004,1220)")?;
    let items = sequence
        .value()
        .items()
        .context("DICOMDIR DirectoryRecordSequence is not a sequence")?;
    let mut paths = Vec::with_capacity(items.len());
    for item in items {
        let record_type = item
            .element(Tag(0x0004, 0x1430))
            .context("DICOMDIR record missing DirectoryRecordType")?
            .to_str()
            .context("invalid DICOMDIR DirectoryRecordType")?;
        if !record_type.trim().eq_ignore_ascii_case("IMAGE") {
            continue;
        }
        let reference = item
            .element(Tag(0x0004, 0x1500))
            .context("DICOMDIR image record missing ReferencedFileID")?
            .to_str()
            .context("invalid DICOMDIR ReferencedFileID")?;
        let mut relative = PathBuf::new();
        for component in reference.trim().split('\\') {
            // DICOM File IDs cannot contain host path syntax. Check both
            // separators and drive syntax independently of the current OS.
            let mut components = Path::new(component).components();
            if component.is_empty()
                || component.contains(['/', ':'])
                || !matches!(components.next(), Some(Component::Normal(_)))
                || components.next().is_some()
            {
                bail!("invalid DICOMDIR ReferencedFileID path component");
            }
            relative.push(component);
        }
        let resolved = root
            .join(relative)
            .canonicalize()
            .context("DICOMDIR referenced file is missing or inaccessible")?;
        if !resolved.starts_with(&root) {
            bail!("DICOMDIR referenced file escapes the file-set root");
        }
        if !resolved.is_file() {
            bail!("DICOMDIR reference is not a file");
        }
        paths.push(resolved);
    }
    if paths.is_empty() {
        bail!("DICOMDIR contained no image ReferencedFileID entries");
    }
    paths.sort();
    paths.dedup();
    Ok(paths)
}
