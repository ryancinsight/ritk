//! Study-level de-identification: every instance under a folder tree.
//!
//! # Invariants
//! - The source is never modified: [`ExportTree`] refuses an output folder
//!   that overlaps the source or holds entries.
//! - The run is all-or-nothing: an instance that cannot be de-identified, or
//!   an exported instance that fails verification, removes every file the run
//!   wrote before the error is returned.
//! - Each patient name and patient ID read from the source is a prohibited
//!   value for the verification of every exported instance.
//! - A DICOMDIR is not exported: its records name the source files and repeat
//!   patient attributes, so the copy would describe another file set.
//! - Text burned into the pixel data is not detected.

use super::verify::{verify_dicom_instances, VerifyIssue, VerifyOptions};
use super::{anonymize_object, AnonymizeOptions};
use crate::format::dicom::export_tree::{io_error, ExportTree, ExportTreeError};
use dicom::core::Tag;
use dicom::object::{open_file, FileDicomObject, InMemDicomObject};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Media Storage Directory Storage, the SOP class a DICOMDIR declares; the
/// value the file-set index fixture writes.
const MEDIA_STORAGE_DIRECTORY: &str = "1.2.840.10008.1.3.10";
const PATIENT_NAME: Tag = Tag(0x0010, 0x0010);
const PATIENT_ID: Tag = Tag(0x0010, 0x0020);

/// What a completed study export wrote and left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StudyAnonymizeReport {
    /// Instances de-identified, written and verified.
    pub instances_written: usize,
    /// Source files that do not parse as DICOM and were not copied.
    pub non_dicom_skipped: usize,
    /// DICOMDIR indexes left out of the export.
    pub indexes_omitted: usize,
    /// Elements removed across all instances.
    pub tags_deleted: usize,
    /// Elements emptied across all instances.
    pub tags_zeroed: usize,
    /// UID values replaced across all instances.
    pub uids_remapped: usize,
    /// Private elements removed across all instances.
    pub private_tags_removed: usize,
}

/// Why a study export produced no output.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StudyAnonymizeError {
    /// The output folder cannot receive the export.
    #[error(transparent)]
    Tree(#[from] ExportTreeError),
    /// The source holds no DICOM instance to export.
    #[error("no DICOM instance found under {0:?}")]
    NoInstances(PathBuf),
    /// A source instance could not be de-identified or written.
    #[error("instance {path:?} could not be de-identified")]
    Instance {
        /// The source instance.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Exported instances failed the export gate.
    #[error("{} verification issue(s) in the export", .0.len())]
    Verification(Vec<(PathBuf, VerifyIssue)>),
    /// The run failed and some of the files it wrote could not be removed.
    #[error("{cause}; {} written file(s) could not be removed", .paths.len())]
    Retained {
        /// The failure that ended the run.
        cause: Box<Self>,
        /// Written files still on disk.
        paths: Vec<PathBuf>,
    },
}

fn identifier(object: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<String> {
    let element = object.element(tag).ok()?;
    let text = element.to_str().ok()?;
    let value = text.trim_matches(|c: char| c.is_whitespace() || c == '\0');
    (!value.is_empty()).then(|| value.to_owned())
}

fn remove_written(cause: StudyAnonymizeError, written: &[PathBuf]) -> StudyAnonymizeError {
    let paths: Vec<PathBuf> = written
        .iter()
        .filter(|path| path.exists() && std::fs::remove_file(path).is_err())
        .cloned()
        .collect();
    if paths.is_empty() {
        cause
    } else {
        StudyAnonymizeError::Retained {
            cause: Box::new(cause),
            paths,
        }
    }
}

/// De-identify one parsed instance and write it to `target`.
fn export_instance(
    object: FileDicomObject<InMemDicomObject>,
    source: &Path,
    target: &Path,
    options: &AnonymizeOptions,
    report: &mut StudyAnonymizeReport,
) -> Result<(), StudyAnonymizeError> {
    let instance_error =
        |error: Box<dyn std::error::Error + Send + Sync>| StudyAnonymizeError::Instance {
            path: source.to_path_buf(),
            source: error,
        };
    let (anonymized, result) =
        anonymize_object(object, options).map_err(|error| instance_error(error.into()))?;
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent).map_err(io_error("create", parent))?;
    }
    anonymized
        .write_to_file(target)
        .map_err(|error| instance_error(error.into()))?;
    report.tags_deleted += result.tags_deleted;
    report.tags_zeroed += result.tags_zeroed;
    report.uids_remapped += result.uids_remapped;
    report.private_tags_removed += result.private_tags_removed;
    Ok(())
}

/// De-identify every DICOM instance under `input` into the empty folder `output`.
///
/// `input` is a study folder or a single instance. The folder layout below
/// `input` is mirrored below `output`, which is created when missing. The
/// export then passes [`verify_dicom_instances`] with the source patient
/// names and patient IDs as prohibited values.
///
/// # Errors
/// Returns [`StudyAnonymizeError`] when the output folder overlaps the source
/// or is not empty, when the source holds no instance, when an instance cannot
/// be de-identified or written, or when verification finds an issue.
pub fn anonymize_dicom_study(
    input: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: &AnonymizeOptions,
) -> Result<StudyAnonymizeReport, StudyAnonymizeError> {
    let tree = ExportTree::plan(input.as_ref(), output.as_ref())?;
    let mut report = StudyAnonymizeReport::default();
    let mut written: Vec<PathBuf> = Vec::new();
    let mut prohibited: BTreeSet<String> = BTreeSet::new();
    for entry in tree.entries() {
        let (path, target) = (entry.source.as_path(), entry.target.as_path());
        let Ok(object) = open_file(path) else {
            report.non_dicom_skipped += 1;
            continue;
        };
        let sop_class = object.meta().media_storage_sop_class_uid();
        if sop_class.trim_end_matches(['\0', ' ']) == MEDIA_STORAGE_DIRECTORY {
            report.indexes_omitted += 1;
            continue;
        }
        prohibited.extend(
            [
                (PATIENT_NAME, options.patient_name.as_str()),
                (PATIENT_ID, options.patient_id.as_str()),
            ]
            .into_iter()
            .filter_map(|(tag, replacement)| {
                identifier(&object, tag).filter(|value| value != replacement)
            }),
        );
        // The target joins the removal list before the write so a truncated
        // file left by a failed write is removed with the rest.
        written.push(target.to_path_buf());
        if let Err(cause) = export_instance(object, path, target, options, &mut report) {
            return Err(remove_written(cause, &written));
        }
    }
    if written.is_empty() {
        return Err(StudyAnonymizeError::NoInstances(
            tree.source().to_path_buf(),
        ));
    }

    let gate = VerifyOptions {
        prohibited_values: prohibited.into_iter().collect(),
        ..VerifyOptions::default()
    };
    let issues = verify_dicom_instances(&written, &gate);
    if !issues.is_empty() {
        return Err(remove_written(
            StudyAnonymizeError::Verification(issues),
            &written,
        ));
    }
    report.instances_written = written.len();
    Ok(report)
}
