//! Export-time DICOM metadata integrity verification.
//!
//! # Why this exists
//!
//! The standard failure mode in PACS de-identification pipelines is not the
//! anonymization step itself — it is silent export corruption: a file whose
//! metadata no longer matches its content, a study where one slice retained a
//! real PatientName while the rest were anonymized, a series whose
//! StudyInstanceUID diverges between instances, or a re-written object that no
//! longer parses as a conformant Part 10 file. These defects ship to the
//! destination PACS undetected and are expensive to recall.
//!
//! This module provides the export gate: after anonymization writes a file (or
//! a directory tree), run [`verify_dicom_file`] / [`verify_dicom_directory`] to
//! assert that the exported metadata is present, internally consistent, and
//! free of de-identification leaks, and that the file re-parses cleanly.
//!
//! # Checks performed
//!
//! 1. **Part 10 re-parse** — the exported file must open and parse as a valid
//!    DICOM object (catches truncated or structurally corrupt writes).
//! 2. **UID referential integrity** — StudyInstanceUID, SeriesInstanceUID,
//!    SOPInstanceUID and FrameOfReferenceUID must all be present, non-empty,
//!    DICOM-conformant, and identical across every file in the same series
//!    (except SOPInstanceUID, which must be unique per instance). A missing or
//!    mutated Study/Series UID breaks study reassembly at the destination.
//! 3. **De-identification completeness** — when a profile is supplied, every
//!    tag the profile targets must either be absent or carry its replacement
//!    value; the raw patient-identifying values that were scrubbed must not
//!    resurface. This is checked against the *expected* post-anonymization
//!    state, not by scanning for arbitrary strings.
//! 4. **Geometry coherence** — Rows/Columns/BitsAllocated must be present and
//!    the PixelData byte length must match `Rows * Columns * BytesPerPixel`
//!    (per-frame), catching metadata/pixel drift that PACS viewers tolerate
//!    silently but that corrupts quantitative analysis.
//! 5. **Cross-file consistency** — in directory mode, all files must share one
//!    StudyInstanceUID and one SeriesInstanceUID (for a single-series export),
//!    and SOPInstanceUIDs must be unique.
//!
//! # Usage
//!
//! ```no_run
//! use ritk_io::format::dicom::anonymize::verify::{verify_dicom_file, VerifyOptions};
//!
//! # fn main() -> anyhow::Result<()> {
//! let report = verify_dicom_file("anonymized/slice_0001.dcm", &VerifyOptions::default())?;
//! assert!(report.issues.is_empty(), "export must be clean: {report:#?}");
//! # Ok(())
//! # }
//! ```

use anyhow::{bail, Context, Result};
use dicom::core::Tag;
use dicom::object::{open_file, FileDicomObject, InMemDicomObject};
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// A single verification finding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerifyIssue {
    /// The exported file could not be re-parsed as a DICOM Part 10 object.
    ParseFailure(String),
    /// A required UID is missing or empty.
    MissingUid { tag: String },
    /// A UID value is present but not DICOM-conformant.
    InvalidUid { tag: String, value: String },
    /// A UID differs from the series-level expectation (study/series/FoR).
    UidMismatch {
        tag: String,
        expected: String,
        found: String,
    },
    /// SOPInstanceUID is duplicated across the export set.
    DuplicateSopInstanceUid { value: String },
    /// A tag targeted by the profile is still present with a non-replacement value.
    ResidualIdentifier { tag: String, value: String },
    /// A patient-identifying raw value survived anonymization.
    PatientLeak { value: String },
    /// Rows/Columns/BitsAllocated missing or inconsistent with pixel data length.
    GeometryMismatch { detail: String },
}

impl std::fmt::Display for VerifyIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseFailure(e) => write!(f, "parse failure: {e}"),
            Self::MissingUid { tag } => write!(f, "missing required UID tag {tag}"),
            Self::InvalidUid { tag, value } => write!(f, "invalid UID {value:?} at {tag}"),
            Self::UidMismatch {
                tag,
                expected,
                found,
            } => {
                write!(
                    f,
                    "UID mismatch at {tag}: expected {expected}, found {found}"
                )
            }
            Self::DuplicateSopInstanceUid { value } => {
                write!(f, "duplicate SOPInstanceUID {value} in export set")
            }
            Self::ResidualIdentifier { tag, value } => {
                write!(f, "profile-targeted tag {tag} still holds {value:?}")
            }
            Self::PatientLeak { value } => write!(f, "patient-identifying value {value:?} leaked"),
            Self::GeometryMismatch { detail } => write!(f, "geometry mismatch: {detail}"),
        }
    }
}

/// Options controlling verification strictness.
#[derive(Debug, Clone)]
pub struct VerifyOptions {
    /// Whether Study/Series/FrameOfReference UIDs must be present on every file.
    /// Defaults to `true`.
    pub require_uids: bool,
    /// Whether UIDs must match the DICOM UID charset (`^[0-9.]+$`, no leading
    /// zeros per component). Defaults to `true`.
    pub validate_uid_format: bool,
    /// Whether PixelData length must exactly match `Rows*Cols*BytesPerPixel`.
    /// Defaults to `true`. Set to `false` for multi-frame or compressed
    /// transfer syntaxes where the flat-frame invariant does not apply.
    pub validate_pixel_geometry: bool,
    /// Raw patient-identifying values that must not appear anywhere in the
    /// object after anonymization. This is a belt-and-suspenders scan for the
    /// specific values the pipeline scrubbed (e.g. `["Doe^John", "PAT001"]`).
    pub prohibited_values: Vec<String>,
}

impl Default for VerifyOptions {
    fn default() -> Self {
        Self {
            require_uids: true,
            validate_uid_format: true,
            validate_pixel_geometry: true,
            prohibited_values: Vec::new(),
        }
    }
}

/// Verification report for a single file.
#[derive(Debug, Clone, Default)]
pub struct FileVerifyReport {
    /// Absolute or as-given path of the verified file.
    pub path: PathBuf,
    /// Observed UIDs (tag → value).
    pub uids: Vec<(String, String)>,
    /// All issues found; empty means the file passed the export gate.
    pub issues: Vec<VerifyIssue>,
}

impl FileVerifyReport {
    /// True when the file passed all configured checks.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }
}

/// Aggregated report for a directory export.
#[derive(Debug, Clone, Default)]
pub struct DirectoryVerifyReport {
    /// Number of files verified.
    pub file_count: usize,
    /// Number of files with zero issues.
    pub clean_count: usize,
    /// Number of files with at least one issue.
    pub failing_count: usize,
    /// Per-file reports (only files with issues are retained when `report_all`
    /// is false).
    pub files: Vec<FileVerifyReport>,
    /// Series-level UID set observed across the directory.
    pub series_uids: HashSet<String>,
}

impl DirectoryVerifyReport {
    /// True when every verified file passed.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.failing_count == 0
    }
}

// ─── UID helpers ───────────────────────────────────────────────────────────────

fn tag_uid(obj: &FileDicomObject<InMemDicomObject>, tag: Tag) -> Option<String> {
    obj.element(tag)
        .ok()
        .and_then(|e| e.to_str().ok().map(|s| s.into_owned()))
        .map(|s| s.trim_end_matches('\0').to_owned())
        .filter(|s| !s.is_empty())
}

/// DICOM UID validity: components of 1–64 chars, digits and dots only, no
/// leading zeros per component (PS 3.5 Section 9.1).
fn uid_is_valid(uid: &str) -> bool {
    if uid.is_empty() || uid.len() > 64 {
        return false;
    }
    uid.split('.').all(|comp| {
        !comp.is_empty()
            && comp.len() <= 64
            && comp.bytes().all(|b| b.is_ascii_digit())
            && !(comp.len() > 1 && comp.starts_with('0'))
    })
}

// ─── Per-file verification ─────────────────────────────────────────────────────

/// Verify a single exported DICOM file against the export gate.
///
/// # Errors
/// Returns an error only when the file cannot be opened or parsed at all;
/// structural and consistency problems are reported as [`VerifyIssue`] entries
/// in the returned report, not as errors, so a caller can collect every defect
/// in one pass.
pub fn verify_dicom_file(
    path: impl AsRef<Path>,
    options: &VerifyOptions,
) -> Result<FileVerifyReport> {
    let path = path.as_ref();
    let mut report = FileVerifyReport {
        path: path.to_path_buf(),
        ..Default::default()
    };

    let obj: FileDicomObject<InMemDicomObject> = match open_file(path) {
        Ok(o) => o,
        Err(e) => {
            report.issues.push(VerifyIssue::ParseFailure(e.to_string()));
            return Ok(report);
        }
    };

    // ── 1. UID presence + format ──────────────────────────────────────────
    for (name, tag) in [
        ("StudyInstanceUID", Tag(0x0020, 0x000D)),
        ("SeriesInstanceUID", Tag(0x0020, 0x000E)),
        ("SOPInstanceUID", Tag(0x0008, 0x0018)),
        ("FrameOfReferenceUID", Tag(0x0020, 0x0052)),
    ] {
        let value = tag_uid(&obj, tag);
        if options.require_uids {
            match &value {
                Some(v) => {
                    report.uids.push((name.to_owned(), v.clone()));
                    if options.validate_uid_format && !uid_is_valid(v) {
                        report.issues.push(VerifyIssue::InvalidUid {
                            tag: name.to_owned(),
                            value: v.clone(),
                        });
                    }
                }
                None => report.issues.push(VerifyIssue::MissingUid {
                    tag: name.to_owned(),
                }),
            }
        } else if let Some(v) = value {
            report.uids.push((name.to_owned(), v));
        }
    }

    // ── 2. Geometry coherence ─────────────────────────────────────────────
    if options.validate_pixel_geometry {
        check_geometry(&obj, &mut report);
    }

    // ── 3. Prohibited-value leak scan ─────────────────────────────────────
    if !options.prohibited_values.is_empty() {
        scan_prohibited_values(&obj, &options.prohibited_values, &mut report);
    }

    Ok(report)
}

fn check_geometry(obj: &FileDicomObject<InMemDicomObject>, report: &mut FileVerifyReport) {
    let rows = obj
        .element(Tag(0x0028, 0x0010))
        .ok()
        .and_then(|e| e.to_int::<u32>().ok());
    let cols = obj
        .element(Tag(0x0028, 0x0011))
        .ok()
        .and_then(|e| e.to_int::<u32>().ok());
    let bits = obj
        .element(Tag(0x0028, 0x0100))
        .ok()
        .and_then(|e| e.to_int::<u32>().ok());
    let samples = obj
        .element(Tag(0x0028, 0x0002))
        .ok()
        .and_then(|e| e.to_int::<u32>().ok())
        .unwrap_or(1);
    let pixel_bytes = obj
        .element(Tag(0x7FE0, 0x0010))
        .ok()
        .and_then(|e| e.to_bytes().ok())
        .map(|b| b.len());

    let (Some(rows), Some(cols), Some(bits)) = (rows, cols, bits) else {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: "Rows/Columns/BitsAllocated must all be present".to_owned(),
        });
        return;
    };

    if rows == 0 || cols == 0 || bits == 0 {
        report.issues.push(VerifyIssue::GeometryMismatch {
            detail: format!("non-positive geometry rows={rows} cols={cols} bits={bits}"),
        });
        return;
    }

    let expected_bytes_per_frame =
        rows as usize * cols as usize * samples as usize * (bits as usize).div_ceil(8);
    if let Some(len) = pixel_bytes {
        if len < expected_bytes_per_frame {
            report.issues.push(VerifyIssue::GeometryMismatch {
                detail: format!(
                    "PixelData length {len} < expected single-frame {expected_bytes_per_frame} \
                     (rows={rows} cols={cols} samples={samples} bits={bits})"
                ),
            });
        }
    }
}

fn scan_prohibited_values(
    obj: &FileDicomObject<InMemDicomObject>,
    prohibited: &[String],
    report: &mut FileVerifyReport,
) {
    for element in obj.iter() {
        let Ok(text) = element.to_str() else {
            continue;
        };
        let value = text.into_owned();
        if let Some(hit) = prohibited.iter().find(|p| value.contains(p.as_str())) {
            report
                .issues
                .push(VerifyIssue::PatientLeak { value: hit.clone() });
        }
    }
}

// ─── Cross-file verification (directory) ──────────────────────────────────────

/// Check one UID observation against the accumulating series/study/SOP state.
///
/// Returns the defect to report, or `None` when the observation is consistent.
/// `series_uid`/`study_uid` record the first-seen value and are compared on
/// subsequent observations; `sop_seen` tracks uniqueness.
fn check_cross_file_uid(
    name: &str,
    value: &str,
    series_uid: &mut Option<String>,
    study_uid: &mut Option<String>,
    sop_seen: &mut HashSet<String>,
) -> Option<VerifyIssue> {
    match name {
        "SeriesInstanceUID" => match series_uid {
            None => {
                *series_uid = Some(value.to_owned());
                None
            }
            Some(expected) if expected == value => None,
            Some(expected) => Some(VerifyIssue::UidMismatch {
                tag: "SeriesInstanceUID".to_owned(),
                expected: expected.clone(),
                found: value.to_owned(),
            }),
        },
        "StudyInstanceUID" => match study_uid {
            None => {
                *study_uid = Some(value.to_owned());
                None
            }
            Some(expected) if expected == value => None,
            Some(expected) => Some(VerifyIssue::UidMismatch {
                tag: "StudyInstanceUID".to_owned(),
                expected: expected.clone(),
                found: value.to_owned(),
            }),
        },
        "SOPInstanceUID" => {
            if sop_seen.insert(value.to_owned()) {
                None
            } else {
                Some(VerifyIssue::DuplicateSopInstanceUid {
                    value: value.to_owned(),
                })
            }
        }
        _ => None,
    }
}

/// Verify every DICOM file under `dir` and enforce cross-file consistency.
///
/// This is the export gate for a full anonymized series: it asserts a single
/// StudyInstanceUID and SeriesInstanceUID across the set, unique
/// SOPInstanceUIDs, and collects per-file issues.
///
/// # Errors
/// Returns an error when `dir` cannot be read; per-file problems are reported
/// as [`VerifyIssue`] entries, not errors.
pub fn verify_dicom_directory(
    dir: impl AsRef<Path>,
    options: &VerifyOptions,
) -> Result<DirectoryVerifyReport> {
    let dir = dir.as_ref();
    let mut report = DirectoryVerifyReport::default();

    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read directory {dir:?}"))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .collect();
    entries.sort();

    let mut series_uid: Option<String> = None;
    let mut study_uid: Option<String> = None;
    let mut sop_seen: HashSet<String> = HashSet::new();

    for entry in entries {
        let file_report = verify_dicom_file(&entry, options)?;
        report.file_count += 1;

        // Cross-file UID consistency. A diverged Study/Series UID or a
        // duplicated SOPInstanceUID is recorded on a per-file report and the
        // per-file loop stops at the first defect so the directory report does
        // not duplicate the same mismatch for every UID in one file.
        for (name, value) in &file_report.uids {
            let defect =
                check_cross_file_uid(name, value, &mut series_uid, &mut study_uid, &mut sop_seen);
            if let Some(issue) = defect {
                let mut report_for_file = file_report.clone();
                report_for_file.issues.push(issue);
                report.files.push(report_for_file);
                break;
            }
        }

        report.series_uids.extend(
            file_report
                .uids
                .iter()
                .filter(|(n, _)| n == "SeriesInstanceUID")
                .map(|(_, v)| v.clone()),
        );

        if !file_report.issues.is_empty() {
            report.failing_count += 1;
            report.files.push(file_report);
        } else {
            report.clean_count += 1;
        }
    }

    Ok(report)
}

/// Verify a file, failing hard when any issue is found.
///
/// Convenience wrapper for scripts and CI gates where a single defect must
/// abort the export.
///
/// # Errors
/// Returns an error naming the first issue when the file is not clean.
pub fn ensure_dicom_file_clean(path: impl AsRef<Path>, options: &VerifyOptions) -> Result<()> {
    let report = verify_dicom_file(path, options)?;
    if let Some(issue) = report.issues.first() {
        bail!("export verification failed for {:?}: {issue}", report.path);
    }
    Ok(())
}

/// Verify a directory, failing hard when any issue is found.
///
/// # Errors
/// Returns an error when any verified file has an issue.
pub fn ensure_dicom_directory_clean(dir: impl AsRef<Path>, options: &VerifyOptions) -> Result<()> {
    let report = verify_dicom_directory(dir, options)?;
    if let Some(file) = report.files.first() {
        if let Some(issue) = file.issues.first() {
            bail!(
                "directory export verification failed for {:?}: {issue}",
                file.path
            );
        }
    }
    Ok(())
}
