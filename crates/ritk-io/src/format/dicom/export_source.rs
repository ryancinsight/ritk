//! Reading the source files of a study export and undoing a failed one.

use ritk_dicom::dataset::DicomFile;
use std::path::{Path, PathBuf};

/// Media Storage Directory Storage, the SOP class a DICOMDIR declares; the
/// value the file-set index fixture writes.
const MEDIA_STORAGE_DIRECTORY: &str = "1.2.840.10008.1.3.10";

/// What a file under the source folder turned out to be.
pub(crate) enum Source {
    /// A DICOM Part 10 instance.
    Instance(Box<DicomFile>),
    /// A DICOMDIR, whose records name the source files.
    Index,
    /// A file that does not open as a DICOM Part 10 file.
    NotDicom,
}

/// Open `path` and classify it.
pub(crate) fn read_source(path: &Path) -> Source {
    let Ok(object) = ritk_dicom::dataset::DicomFile::open(path) else {
        return Source::NotDicom;
    };
    let sop_class = object.meta().media_storage_sop_class_uid();
    if sop_class.trim_end_matches(['\0', ' ']) == MEDIA_STORAGE_DIRECTORY {
        Source::Index
    } else {
        Source::Instance(Box::new(object))
    }
}

/// Remove the files a failed export wrote; returns those still on disk.
pub(crate) fn remove_written(written: &[PathBuf]) -> Vec<PathBuf> {
    written
        .iter()
        .filter(|path| path.exists() && std::fs::remove_file(path).is_err())
        .cloned()
        .collect()
}
