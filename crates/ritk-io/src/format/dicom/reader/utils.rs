//! Miscellaneous utilities for the DICOM reader.

use std::ffi::OsStr;
use std::path::Path;

/// Return true when the path is likely a DICOM Part 10 file.
///
/// Primary test: file extension is `.dcm`, `.dicom`, or `.ima`.
/// Secondary test: extensionless files are probed for the DICM magic bytes
/// at byte offset 128 (DICOM PS3.10 §7.1).
///
/// `.hdr`/`.img` (Analyze 7.5) and `.raw` are explicitly excluded.
pub(super) fn is_likely_dicom_file(path: &Path) -> bool {
    if let Some(ext) = path.extension().and_then(OsStr::to_str) {
        let ext_lc = ext.to_ascii_lowercase();
        return matches!(ext_lc.as_str(), "dcm" | "dicom" | "ima");
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
