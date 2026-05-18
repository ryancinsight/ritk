//! Temp-directory helpers and DICOM byte-payload detection.

use std::path::PathBuf;

use anyhow::{Context, Result};

/// Create a unique temporary subdirectory under the system temp dir.
///
/// The directory name incorporates `prefix`, the current PID, and a
/// nanosecond-precision timestamp to avoid collisions across concurrent
/// processes and rapid successive calls.
pub(super) fn create_unique_temp_subdir(prefix: &str) -> Result<PathBuf> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock must be after UNIX_EPOCH")
        .as_nanos();
    let pid = std::process::id();
    let path = std::env::temp_dir().join(format!("{prefix}_{pid}_{now}"));
    std::fs::create_dir_all(&path)
        .with_context(|| format!("failed to create temp directory '{}'", path.display()))?;
    Ok(path)
}

/// Sanitize a user-supplied filename for safe filesystem writes.
///
/// - Replaces non-alphanumeric characters (except `.`, `_`, `-`) with `_`.
/// - Truncates to 120 characters.
/// - Falls back to `slice_{index:04}.dcm` when the result is empty.
/// - Appends `.dcm` when no extension is present.
pub(super) fn sanitize_temp_filename(name: &str, index: usize) -> String {
    let trimmed = name.trim();
    if trimmed.is_empty() {
        return format!("slice_{index:04}.dcm");
    }
    let mut cleaned: String = trimmed
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.len() > 120 {
        cleaned.truncate(120);
    }
    if cleaned.is_empty() {
        cleaned = format!("slice_{index:04}.dcm");
    }
    if !cleaned.contains('.') {
        cleaned.push_str(".dcm");
    }
    cleaned
}

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
