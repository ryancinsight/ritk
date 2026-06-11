//! Python-exposed DICOM de-identification / anonymization (PS 3.15 Annex E).

use crate::errors::{RitkPyError, RitkResult};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use ritk_io::{anonymize_dicom_directory, AnonymizationProfile, AnonymizeOptions, CleaningPolicy};

/// DICOM cleaning policy for anonymization, replacing two boolean parameters.
///
/// Eliminates boolean blindness from the `clean_pixel_data: bool` +
/// `clean_private_tags: bool` pair: each variant is self-documenting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PyCleaningPolicy {
    /// No cleaning (default).
    #[default]
    None,
    /// Zero-pad PixelData elements.
    CleanPixelData,
    /// Remove all private DICOM tags.
    CleanPrivateTags,
    /// Both pixel data zero-padding and private tag removal.
    CleanAll,
}

impl<'py> FromPyObject<'py> for PyCleaningPolicy {
    fn extract_bound(ob: &pyo3::Bound<'py, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "clean_pixel_data" | "pixel" => Ok(Self::CleanPixelData),
            "clean_private_tags" | "private" => Ok(Self::CleanPrivateTags),
            "clean_all" | "all" => Ok(Self::CleanAll),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown cleaning policy '{}'. Choices: none, clean_pixel_data, clean_private_tags, clean_all",
                other
            ))),
        }
    }
}

/// Anonymize all DICOM files in `input_dir`, writing results to `output_dir`.
///
/// Applies PS 3.15 Annex E patient de-identification.
///
/// Args:
/// input_dir: Source directory containing DICOM files (str).
/// output_dir: Destination directory; created if absent (str).
/// profile: Anonymization profile name (str, default "basic").
/// Choices: `"basic"` | `"basic_replace_uids"` | `"aggressive"` | `"enhanced"`.
/// patient_name: Replacement for PatientName (str, default "ANONYMOUS").
/// patient_id: Replacement for PatientID (str, default "ANON001").
/// uid_salt: Salt for deterministic UID remapping (str, default "ritk-anon-salt").
/// cleaning: DICOM cleaning policy (str, default "none").
/// Choices: `"none"` | `"clean_pixel_data"` (or `"pixel"`) |
/// `"clean_private_tags"` (or `"private"`) | `"clean_all"` (or `"all"`).
/// `"clean_all"` is required for full PS 3.15 Annex E compliance.
///
/// Returns:
/// dict with keys:
/// - `"file_count"` (int): total files processed.
/// - `"success_count"` (int): files successfully anonymized.
/// - `"error_count"` (int): files that failed.
/// - `"errors"` (list[list[str]]): `[[path, error_msg], ...]` for failures.
///
/// Raises:
/// IOError: if `input_dir` cannot be scanned or `output_dir` cannot be created.
#[pyfunction]
#[pyo3(signature = (input_dir, output_dir, profile="basic", patient_name="ANONYMOUS", patient_id="ANON001", uid_salt="ritk-anon-salt", cleaning=PyCleaningPolicy::None))]
pub fn anonymize_dicom_dir(
    py: Python<'_>,
    input_dir: &str,
    output_dir: &str,
    profile: &str,
    patient_name: &str,
    patient_id: &str,
    uid_salt: &str,
    cleaning: PyCleaningPolicy,
) -> RitkResult<Py<PyDict>> {
    let anon_profile = match profile {
        "basic" => AnonymizationProfile::Basic,
        "basic_replace_uids" => AnonymizationProfile::BasicReplaceUids,
        "aggressive" => AnonymizationProfile::Aggressive,
        "enhanced" => AnonymizationProfile::Enhanced,
        other => {
            return Err(RitkPyError::io(format!(
                "Unknown anonymization profile '{other}'. \
                Choices: basic, basic_replace_uids, aggressive, enhanced"
            )));
        }
    };
    let (clean_pixel, clean_private) = match cleaning {
        PyCleaningPolicy::None => (CleaningPolicy::Skip, CleaningPolicy::Skip),
        PyCleaningPolicy::CleanPixelData => (CleaningPolicy::Clean, CleaningPolicy::Skip),
        PyCleaningPolicy::CleanPrivateTags => (CleaningPolicy::Skip, CleaningPolicy::Clean),
        PyCleaningPolicy::CleanAll => (CleaningPolicy::Clean, CleaningPolicy::Clean),
    };
    let options = AnonymizeOptions {
        profile: anon_profile,
        patient_name: patient_name.to_owned(),
        patient_id: patient_id.to_owned(),
        uid_salt: uid_salt.to_owned(),
        clean_pixel_data: clean_pixel,
        clean_private_tags: clean_private,
    };

    let input_owned = input_dir.to_string();
    let output_owned = output_dir.to_string();

    let stats = py.allow_threads(move || {
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
