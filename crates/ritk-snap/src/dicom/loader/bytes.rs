//! DICOM byte-payload detection helpers.

/// Classify a DICOM name hint or Part 10 byte payload.
///
/// A DICOM registry path, a basename of DICOMDIR, or a Part 10 preamble
/// at offset 128 identifies a DICOM payload.
pub(crate) fn is_likely_dicom_bytes(name_hint: &str, bytes: &[u8]) -> bool {
    let path = std::path::Path::new(name_hint);
    let is_dicomdir = path
        .file_name()
        .and_then(std::ffi::OsStr::to_str)
        .is_some_and(|name| name.eq_ignore_ascii_case("DICOMDIR"));
    if ritk_io::ImageFormat::from_path(path) == Some(ritk_io::ImageFormat::Dicom) || is_dicomdir {
        return true;
    }
    bytes.len() >= 132 && &bytes[128..132] == b"DICM"
}
