//! DICOM byte-payload detection helpers.

/// Heuristic check: does `name_hint` or `bytes` look like a DICOM payload?
///
/// Returns `true` when the filename extension is a known DICOM extension
/// (`.dcm`, `.dicom`, `.ima`, or literal `DICOMDIR`), or when the byte
/// payload starts with the DICM preamble at offset 128.
pub(super) fn is_likely_dicom_bytes(name_hint: &str, bytes: &[u8]) -> bool {
    let n = name_hint.to_ascii_lowercase();
    if n.ends_with(".dcm") || n.ends_with(".dicom") || n.ends_with(".ima") || n == "dicomdir" {
        return true;
    }
    bytes.len() >= 132 && &bytes[128..132] == b"DICM"
}
