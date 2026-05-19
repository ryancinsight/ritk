//! Python-exposed DICOM de-identification / anonymization (PS 3.15 Annex E).

use crate::errors::{RitkPyError, RitkResult};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_io::{anonymize_dicom_directory, AnonymizationProfile, AnonymizeOptions};

/// Anonymize all DICOM files in `input_dir`, writing results to `output_dir`.
///
/// Applies PS 3.15 Annex E patient de-identification.
///
/// Args:
///     input_dir:  Source directory containing DICOM files (str).
///     output_dir: Destination directory; created if absent (str).
///     profile:    Anonymization profile name (str, default "basic").
///                 Choices: `"basic"` | `"basic_replace_uids"` | `"aggressive"`.
///     clean_pixel_data: If True, zero-pad PixelData elements (bool, default False).
///     clean_private_tags: If True, remove all private DICOM elements (odd-group
///                         tags). Required for full PS 3.15 Annex E compliance
///                         (default False).
///
/// Returns:
///     dict with keys:
///       - `"file_count"` (int): total files processed.
///       - `"success_count"` (int): files successfully anonymized.
///       - `"error_count"` (int): files that failed.
///       - `"errors"` (list[list[str]]): `[[path, error_msg], ...]` for failures.
///
/// Raises:
///     IOError: if `input_dir` cannot be scanned or `output_dir` cannot be created.
#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, profile="basic", clean_pixel_data=false, clean_private_tags=false))]
pub fn anonymize_dicom_dir(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
    profile: &str,
    clean_pixel_data: bool,
    clean_private_tags: bool,
) -> RitkResult<Py<PyDict>> {
    let anon_profile = match profile {
        "basic" => AnonymizationProfile::Basic,
        "basic_replace_uids" => AnonymizationProfile::BasicReplaceUids,
        "aggressive" => AnonymizationProfile::Aggressive,
        other => {
            return Err(RitkPyError::io(format!(
                "Unknown anonymization profile '{other}'. \
                 Choices: basic, basic_replace_uids, aggressive"
            )));
        }
    };
    let options = AnonymizeOptions {
        profile: anon_profile,
        clean_pixel_data,
        clean_private_tags,
    };
    let input_owned = input_dir.to_string();
    let output_owned = output_dir.to_string();

    let stats = py
        .allow_threads(move || {
            anonymize_dicom_directory(&input_owned, &output_owned, &options)
                .map_err(|e| RitkPyError::io(format!("Anonymization error: {e}")))
        })?;

    let dict = PyDict::new_bound(py);
    dict.set_item("file_count", stats.file_count)?;
    dict.set_item("success_count", stats.success_count)?;
    dict.set_item("error_count", stats.error_count)?;
    let errors_list = PyList::empty_bound(py);
    for (path, msg) in &stats.errors {
        let pair = PyList::empty_bound(py);
        pair.append(path.display().to_string())?;
        pair.append(msg.clone())?;
        errors_list.append(pair)?;
    }
    dict.set_item("errors", errors_list)?;
    Ok(dict.into())
}
