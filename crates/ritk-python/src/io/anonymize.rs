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
    /// Zero-fill native top-level Pixel Data.
    CleanPixelData,
    /// Remove all private DICOM tags.
    CleanPrivateTags,
    /// Both pixel data zero-padding and private tag removal.
    CleanAll,
}

impl<'a, 'py> FromPyObject<'a, 'py> for PyCleaningPolicy {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        match s.to_lowercase().as_str() {
            "none" => Ok(Self::None),
            "clean_pixel_data" | "pixel" => Ok(Self::CleanPixelData),
            "clean_private_tags" | "private" => Ok(Self::CleanPrivateTags),
            "clean_all" | "all" => Ok(Self::CleanAll),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown cleaning policy '{}'. Choices: none, clean_pixel_data, clean_private_tags, clean_all",
                other
            ))) }
    }
}

/// Anonymize DICOM files directly in `input_dir` and write copies to `output_dir`.
///
/// Writes numbered names such as `000001.dcm` in source-path order, skipping
/// occupied destination names. Files that cannot be opened as DICOM and files
/// in subdirectories are skipped. DICOMDIR objects are reported as per-file
/// errors. Pass the directory containing the images. A fresh destination
/// keeps the exported batch separate from existing files.
///
/// Metadata handling follows the selected profile. By default, pixel data is
/// preserved. Explicit pixel cleaning destroys readable native top-level
/// Pixel Data (7FE0,0010) by zero-filling it. Encapsulated or otherwise
/// unreadable Pixel Data, Float Pixel Data, and Double Float Pixel Data
/// produce per-file errors without writing those instances.
/// The export does not detect burned-in identifiers or verify
/// PS 3.15 Annex E conformance.
///
/// Args:
/// input_dir: Source directory containing DICOM files (str).
/// output_dir: Separate destination directory, created with any missing parents (str).
/// profile: Anonymization profile name (str, default "basic").
/// Choices: `"basic"` | `"basic_replace_uids"` | `"aggressive"` | `"enhanced"`.
/// patient_name: Replacement for PatientName (str, default "ANONYMOUS").
/// patient_id: Replacement for PatientID (str, default "ANON001").
/// uid_salt: Salt for deterministic UID remapping (str, default "ritk-anon-salt").
/// cleaning: DICOM cleaning policy (str, default "none").
/// Choices: `"none"` | `"clean_pixel_data"` (or `"pixel"`) |
/// `"clean_private_tags"` (or `"private"`) | `"clean_all"` (or `"all"`).
///
/// Returns:
/// dict with keys:
/// - `"file_count"` (int): files opened as DICOM, including per-file failures.
/// - `"success_count"` (int): files successfully anonymized.
/// - `"error_count"` (int): files that failed.
/// - `"errors"` (list[list[str]]): `[[path, error_msg], ...]` for failures.
///   Source paths can contain patient identifiers.
///
/// Raises:
/// IOError: if the directories cannot be prepared or refer to the same directory.
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

    let stats = py.detach(move || {
        anonymize_dicom_directory(&input_owned, &output_owned, &options)
            .map_err(|e| RitkPyError::io(format!("Anonymization error: {e}")))
    })?;

    let dict = PyDict::new(py);
    dict.set_item("file_count", stats.file_count)?;
    dict.set_item("success_count", stats.success_count)?;
    dict.set_item("error_count", stats.error_count)?;

    let errors_list = PyList::empty(py);
    for (path, msg) in &stats.errors {
        let pair = PyList::empty(py);
        pair.append(path.display().to_string())?;
        pair.append(msg.clone())?;
        errors_list.append(pair)?;
    }
    dict.set_item("errors", errors_list)?;

    Ok(dict.into())
}
