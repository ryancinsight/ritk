//! Study-level repair: a corrected copy of every instance under a folder tree.

use super::{repair_values, RepairError, RepairSummary};
use crate::format::dicom::anonymize::file_meta::align_media_storage_uid;
use crate::format::dicom::export_source::{read_source, remove_written, Source};
use crate::format::dicom::export_tree::{io_error, ExportTree, ExportTreeError};
use crate::format::dicom::identity::uid_is_valid;
use dicom::core::Tag;
use dicom::object::{FileDicomObject, InMemDicomObject};
use std::path::PathBuf;

const SOP_CLASS_UID: Tag = Tag(0x0008, 0x0016);

/// What a completed repair export wrote, corrected and left out.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct StudyRepairReport {
    /// Instances written to the output folder.
    pub instances_written: usize,
    /// Written instances with a rewritten value or a realigned file meta UID.
    pub instances_repaired: usize,
    /// Date, time and UID values rewritten across all instances.
    pub values: RepairSummary,
    /// Instances whose file meta group named another SOP class or SOP
    /// instance than the data set and was set to the data set's.
    pub file_meta_uids_aligned: usize,
    /// Source files that do not open as DICOM and were not copied.
    pub non_dicom_skipped: usize,
    /// DICOMDIR indexes left out of the export.
    pub indexes_omitted: usize,
}

/// Why a repair export produced no output.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum StudyRepairError {
    /// The source cannot be resolved or listed, or the output folder
    /// cannot receive the export.
    #[error(transparent)]
    Tree(#[from] ExportTreeError),
    /// The source holds no DICOM instance.
    #[error("no DICOM instance found under {0:?}")]
    NoInstances(PathBuf),
    /// A source instance could not be repaired.
    #[error("instance {path:?} could not be repaired")]
    Instance {
        /// The source instance.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: RepairError,
    },
    /// A repaired instance could not be written.
    #[error("repaired instance could not be written to {path:?}")]
    Write {
        /// The output path.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: Box<dicom::object::WriteError>,
    },
    /// The run failed and some of the files it wrote could not be removed.
    #[error("{} written file(s) could not be removed after a failed run", .paths.len())]
    Retained {
        /// The failure that ended the run.
        #[source]
        cause: Box<Self>,
        /// Written files still on disk.
        paths: Vec<PathBuf>,
    },
}

fn failed(cause: StudyRepairError, written: &[PathBuf]) -> StudyRepairError {
    let paths = remove_written(written);
    if paths.is_empty() {
        cause
    } else {
        StudyRepairError::Retained {
            cause: Box::new(cause),
            paths,
        }
    }
}

/// Set Media Storage SOP Class UID (0002,0002) to the data set's SOPClassUID
/// when that is a single conformant UID naming another class (PS3.10 Table
/// 7.1-1); returns whether the file meta group changed.
fn align_media_storage_class(object: &mut FileDicomObject<InMemDicomObject>) -> bool {
    let class = object
        .element(SOP_CLASS_UID)
        .ok()
        .and_then(|element| element.to_str().ok())
        .map(|text| text.trim_end_matches(['\0', ' ']).to_owned())
        .filter(|uid| uid_is_valid(uid));
    let Some(class) = class else {
        return false;
    };
    if object.meta().media_storage_sop_class_uid() == class {
        return false;
    }
    object.update_meta(|meta| meta.media_storage_sop_class_uid = class);
    true
}

/// Write a repaired copy of every DICOM instance under `input` into the empty
/// folder `output`.
///
/// `input` is a study folder or a single instance; its layout is mirrored
/// below `output`, which is created when missing, and the source is never
/// modified. Each instance gets the value rewrites of [`repair_values`];
/// then its file meta Media Storage SOP Class UID and SOP Instance UID are set
/// to the data set's SOPClassUID and SOPInstanceUID (PS3.10 Table 7.1-1), each
/// only when the data set value is a single conformant UID. Files that do not open as DICOM Part 10,
/// a data set stored without the Part 10 header among them, are counted and
/// not copied; a DICOMDIR is counted and left out, because its records name
/// the source files.
///
/// # Errors
/// Returns [`StudyRepairError`] when the source cannot be resolved or listed,
/// when the output folder overlaps the source or is not empty, when the source
/// holds no instance, or when an instance cannot be repaired or written. Every
/// file the run wrote is removed first; files that resist removal are named by
/// [`StudyRepairError::Retained`].
pub fn repair_dicom_study(
    input: impl AsRef<std::path::Path>,
    output: impl AsRef<std::path::Path>,
) -> Result<StudyRepairReport, StudyRepairError> {
    let tree = ExportTree::plan(input.as_ref(), output.as_ref())?;
    let mut report = StudyRepairReport::default();
    let mut written: Vec<PathBuf> = Vec::new();
    for entry in tree.entries() {
        let mut object = match read_source(&entry.source) {
            Source::Instance(object) => *object,
            Source::Index => {
                report.indexes_omitted += 1;
                continue;
            }
            Source::NotDicom => {
                report.non_dicom_skipped += 1;
                continue;
            }
        };
        let values = match repair_values(&mut object) {
            Ok(values) => values,
            Err(source) => {
                let path = entry.source.clone();
                return Err(failed(
                    StudyRepairError::Instance { path, source },
                    &written,
                ));
            }
        };
        let class_aligned = align_media_storage_class(&mut object);
        let aligned = align_media_storage_uid(&mut object) || class_aligned;
        // The target joins the removal list before the write so a truncated
        // file left by a failed write is removed with the rest.
        written.push(entry.target.clone());
        let stored = match entry.target.parent() {
            Some(parent) => std::fs::create_dir_all(parent)
                .map_err(io_error("create", parent))
                .map_err(StudyRepairError::Tree),
            None => Ok(()),
        }
        .and_then(|()| {
            object
                .write_to_file(&entry.target)
                .map_err(|source| StudyRepairError::Write {
                    path: entry.target.clone(),
                    source: Box::new(source),
                })
        });
        if let Err(cause) = stored {
            return Err(failed(cause, &written));
        }
        report.values.dates += values.dates;
        report.values.times += values.times;
        report.values.uids += values.uids;
        report.file_meta_uids_aligned += usize::from(aligned);
        report.instances_repaired += usize::from(aligned || values != RepairSummary::default());
    }
    if written.is_empty() {
        return Err(StudyRepairError::NoInstances(tree.source().to_path_buf()));
    }
    report.instances_written = written.len();
    Ok(report)
}

#[cfg(test)]
mod tests;
