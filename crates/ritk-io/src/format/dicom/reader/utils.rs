//! Miscellaneous utilities for the DICOM reader.

use crate::ImageFormat;
use std::path::Path;

/// Return true when the path is likely a DICOM Part 10 file.
///
/// Primary test: file extension is recognised as DICOM by [`ImageFormat::from_path`].
/// Secondary test: extensionless files are probed for the DICM magic bytes
/// at byte offset 128 (DICOM PS3.10 §7.1).
///
/// `.hdr`/`.img` (Analyze 7.5) and `.raw` are explicitly excluded.
pub(super) fn is_likely_dicom_file(path: &Path) -> bool {
    if path.extension().is_some() {
        return ImageFormat::from_path(path) == Some(ImageFormat::Dicom);
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
