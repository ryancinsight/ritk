//! Reading the source files of a study export and undoing a failed one.

use ritk_dicom::dataset::DicomFile;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Length of the preamble that opens a Part 10 file (PS3.10 section 7.1).
const PREAMBLE_LENGTH: usize = 128;
/// The prefix that follows the preamble.
const PREFIX: &[u8] = b"DICM";
/// Offset of the first byte after the preamble and the prefix.
const PREFIX_END: u64 = 132;

/// Media Storage Directory Storage, the SOP class a DICOMDIR declares; the
/// value the file-set index fixture writes.
const MEDIA_STORAGE_DIRECTORY: &str = "1.2.840.10008.1.3.10";

/// What a file under the source folder turned out to be.
pub(crate) enum Source {
    /// A DICOM Part 10 instance.
    Instance(Box<DicomFile>),
    /// A DICOMDIR, whose records name the source files.
    Index,
    /// A file that cannot be read at all, or that carries the `DICM` prefix
    /// (at byte 0 or 128) and does not open as a DICOM Part 10 file.
    Unreadable(anyhow::Error),
    /// A file read and found without the prefix.
    NotDicom,
}

/// Open `path` and classify it.
pub(crate) fn read_source(path: &Path) -> Source {
    let object = match ritk_dicom::dataset::DicomFile::open(path) {
        Ok(object) => object,
        Err(error) => {
            return match has_part10_prefix(path) {
                Ok(false) => Source::NotDicom,
                Ok(true) => Source::Unreadable(error),
                Err(cause) => Source::Unreadable(anyhow::Error::new(cause)),
            };
        }
    };
    let sop_class = object.meta().media_storage_sop_class_uid();
    if sop_class.trim_end_matches(['\0', ' ']) == MEDIA_STORAGE_DIRECTORY {
        Source::Index
    } else {
        Source::Instance(Box::new(object))
    }
}

/// Whether `path` holds `DICM` after a 128-byte preamble or, as in a file
/// written without the preamble, as its first four bytes.
fn has_part10_prefix(path: &Path) -> std::io::Result<bool> {
    let mut header = Vec::new();
    std::fs::File::open(path)?
        .take(PREFIX_END)
        .read_to_end(&mut header)?;
    Ok(header.get(PREAMBLE_LENGTH..) == Some(PREFIX) || header.starts_with(PREFIX))
}

/// Name an exported instance from a one-based ordinal and a fixed extension.
pub(crate) fn numbered_instance_path(output: &Path, ordinal: usize) -> PathBuf {
    output.join(format!("{ordinal:06}.dcm"))
}

/// Remove the files a failed export wrote; returns those still on disk.
pub(crate) fn remove_written(written: &[PathBuf]) -> Vec<PathBuf> {
    written
        .iter()
        .filter(|path| path.exists() && std::fs::remove_file(path).is_err())
        .cloned()
        .collect()
}
