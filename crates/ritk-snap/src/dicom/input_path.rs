//! DICOM input path normalization for viewer workflows.
//!
//! `ritk-io` accepts a directory containing DICOM slices or a `DICOMDIR`
//! index. Viewer entry points can receive either that directory or the
//! `DICOMDIR` file itself; this module maps both forms to the canonical
//! directory path before scanning or loading.

use std::io::Read;
use std::path::{Path, PathBuf};

/// Classified DICOM viewer input path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DicomInputPath {
    /// A directory passed directly to `ritk-io`.
    Directory(PathBuf),
    /// A selected `DICOMDIR` file; the contained directory is passed to `ritk-io`.
    DicomDirFile {
        /// Path selected by the caller.
        file: PathBuf,
        /// Parent directory containing the DICOMDIR and referenced files.
        root: PathBuf },
    /// A selected DICOM slice file; the parent directory is loaded as a series root.
    SingleDicomFile {
        /// Path selected by the caller.
        file: PathBuf,
        /// Parent directory containing the series.
        root: PathBuf },
    /// A file that is not a DICOMDIR index.
    OtherFile(PathBuf) }

impl DicomInputPath {
    /// Path to pass into `ritk-io` when this input is DICOM-loadable.
    pub fn dicom_root(&self) -> Option<&Path> {
        match self {
            Self::Directory(path) => Some(path.as_path()),
            Self::DicomDirFile { root, .. } => Some(root.as_path()),
            Self::SingleDicomFile { root, .. } => Some(root.as_path()),
            Self::OtherFile(_) => None }
    }
}

/// Classify a path for DICOM viewer import.
pub fn classify_dicom_input_path(path: impl AsRef<Path>) -> DicomInputPath {
    let path = path.as_ref();
    if path.is_dir() {
        return DicomInputPath::Directory(path.to_path_buf());
    }

    if is_dicomdir_file(path) {
        let root = path.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
        return DicomInputPath::DicomDirFile {
            file: path.to_path_buf(),
            root };
    }

    if is_single_dicom_file(path) {
        let root = path.parent().unwrap_or_else(|| Path::new("")).to_path_buf();
        return DicomInputPath::SingleDicomFile {
            file: path.to_path_buf(),
            root };
    }

    DicomInputPath::OtherFile(path.to_path_buf())
}

fn is_dicomdir_file(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.eq_ignore_ascii_case("DICOMDIR"))
        .unwrap_or(false)
}

fn is_single_dicom_file(path: &Path) -> bool {
    if !path.is_file() {
        return false;
    }

    let has_dicom_ext = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("dcm") || ext.eq_ignore_ascii_case("dicom"))
        .unwrap_or(false);

    has_dicom_ext || has_dicom_preamble(path)
}

fn has_dicom_preamble(path: &Path) -> bool {
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return false };
    let mut header = [0u8; 132];
    if file.read_exact(&mut header).is_err() {
        return false;
    }
    &header[128..132] == b"DICM"
}

#[cfg(test)]
#[path = "input_path_tests.rs"]
mod tests;
