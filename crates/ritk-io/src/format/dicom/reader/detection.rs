//! DICOM Part 10 file detection.

use crate::ImageFormat;
use std::path::Path;

/// Return true when the path is likely a DICOM Part 10 file.
///
/// Primary test: file extension is recognised as DICOM by [`ImageFormat::from_path`].
/// Secondary test: files are probed for the DICM magic bytes
/// at byte offset 128 (DICOM PS3.10 §7.1).
///
/// The Part 10 preamble also admits a selected instance with an arbitrary extension.
pub(super) fn is_likely_dicom_file(path: &Path) -> bool {
    if ImageFormat::from_path(path) == Some(ImageFormat::Dicom) {
        return true;
    }
    use std::io::{Read, Seek, SeekFrom};
    if let Ok(mut f) = std::fs::File::open(path) {
        let mut magic = [0u8; 4];
        if f.seek(SeekFrom::Start(128)).is_ok() && f.read_exact(&mut magic).is_ok() {
            return &magic == b"DICM";
        }
    }
    false
}
