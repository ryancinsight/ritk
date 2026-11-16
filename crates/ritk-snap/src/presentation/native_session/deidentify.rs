//! Export a de-identified copy of the open study from the native shell.
//!
//! The viewer never rewrites the study it displays: the copy goes to a folder
//! the reader picks, and RITK refuses a folder that overlaps the source or
//! already holds files. Pixel data is copied as stored, so text burned into
//! the image survives; the status line says so after every completed export.

use super::NativeViewerSession;
use anyhow::{anyhow, Context, Result};
use metis_platform::native::{pick, DialogSelection};
use ritk_io::format::dicom::anonymize::{
    anonymize_dicom_study, AnonymizationProfile, AnonymizeOptions, StudyAnonymizeError,
    StudyAnonymizeReport,
};
use std::path::Path;

/// Which RITK anonymization profile a de-identified export applies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::presentation::native_session) enum Deidentification {
    /// [`AnonymizationProfile::BasicReplaceUids`]: study, series, acquisition
    /// and content dates and times and the study and series descriptions stay.
    Basic,
    /// [`AnonymizationProfile::Enhanced`]: those dates and times are emptied
    /// and the study and series descriptions are removed.
    Strict,
}

impl Deidentification {
    const fn profile(self) -> AnonymizationProfile {
        match self {
            Self::Basic => AnonymizationProfile::BasicReplaceUids,
            Self::Strict => AnonymizationProfile::Enhanced,
        }
    }

    const fn summary(self) -> &'static str {
        match self {
            Self::Basic => "study dates and descriptions kept",
            Self::Strict => "study dates emptied, study and series descriptions removed",
        }
    }
}

fn export_status(level: Deidentification, report: StudyAnonymizeReport, output: &Path) -> String {
    let left_out = report.non_dicom_skipped + report.indexes_omitted;
    let omitted = if left_out == 0 {
        String::new()
    } else {
        format!(" {left_out} file(s) that are not DICOM instances left out.")
    };
    format!(
        "De-identified {} instance(s) to {} ({}).{omitted} Text burned into pixels is not removed.",
        report.instances_written,
        output.display(),
        level.summary(),
    )
}

/// The status line for a failed export; a verification failure names its
/// first issue and the file that raised it.
pub(super) fn failure_status(error: &anyhow::Error) -> String {
    let first = match error.downcast_ref::<StudyAnonymizeError>() {
        Some(StudyAnonymizeError::Verification(issues)) => issues.first(),
        _ => None,
    };
    match first {
        Some((path, issue)) => format!(
            "De-identified export failed: {error:#}; first: {issue} in {}",
            path.display()
        ),
        None => format!("De-identified export failed: {error:#}"),
    }
}

impl NativeViewerSession {
    /// Ask for an output folder and export the open study into it.
    ///
    /// Returns whether the session changed; a cancelled dialog changes nothing.
    pub(super) fn anonymize_study_from_dialog(&mut self, level: Deidentification) -> Result<bool> {
        self.end_held_gestures();
        if self.study_root.is_none() {
            self.active_app_mut().status_message =
                "Open a study from disk before exporting a de-identified copy.".to_owned();
            return Ok(true);
        }
        let selected =
            pick(DialogSelection::Folder).context("show native de-identified export picker")?;
        let Some(output) = selected else {
            return Ok(false);
        };
        self.anonymize_study_to(&output, level)?;
        Ok(true)
    }

    /// Export a de-identified copy of the open study into `output`.
    pub(super) fn anonymize_study_to(
        &mut self,
        output: &Path,
        level: Deidentification,
    ) -> Result<()> {
        let source = self
            .study_root
            .as_deref()
            .ok_or_else(|| anyhow!("no study folder is open"))?;
        let options = AnonymizeOptions::with_fresh_salt(level.profile())
            .context("draw a UID salt from the operating system")?;
        let report = anonymize_dicom_study(source, output, &options)?;
        self.active_app_mut().status_message = export_status(level, report, output);
        Ok(())
    }
}
