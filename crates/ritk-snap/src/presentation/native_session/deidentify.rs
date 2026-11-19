//! Export a de-identified copy of the study or series in the active pane.
//!
//! The viewer never rewrites the study it displays: the copy goes to a folder
//! the reader picks, and RITK refuses a folder that overlaps the source or
//! already holds files. Pixel data is copied as stored, so text burned into
//! the image survives; the status line says so after every completed export.

use super::series_browser::SeriesChoice;
use super::NativeViewerSession;
use crate::dicom::input_path::classify_dicom_input_path;
use anyhow::{anyhow, Context, Result};
use metis_platform::native::{pick, DialogSelection};
use ritk_io::format::dicom::anonymize::{
    anonymize_dicom_selection, verify::VerifyIssue, AnonymizationProfile, AnonymizeOptions,
    StudyAnonymizeError, StudyAnonymizeReport,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnonymizationScope {
    Study,
    Series,
}

impl AnonymizationScope {
    const fn label(self) -> &'static str {
        match self {
            Self::Study => "Active study",
            Self::Series => "Active series",
        }
    }
}

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

    const fn label(self) -> &'static str {
        match self {
            Self::Basic => "Basic",
            Self::Strict => "Strict",
        }
    }
}

fn export_status(
    scope: AnonymizationScope,
    level: Deidentification,
    report: StudyAnonymizeReport,
    output: &Path,
) -> String {
    format!(
        "{} ({}): {} DICOM instance(s) exported. Burned-in text remains. Destination {}",
        scope.label(),
        level.label(),
        report.instances_written,
        output.display(),
    )
}

/// The first issue of a verification failure, also when a failed cleanup
/// wraps that failure.
fn first_issue(error: &StudyAnonymizeError) -> Option<&(PathBuf, VerifyIssue)> {
    match error {
        StudyAnonymizeError::Verification(issues) => issues.first(),
        StudyAnonymizeError::Retained { cause, .. } => first_issue(cause),
        _ => None,
    }
}

/// The status line for a failed export: the error, followed by the first
/// verification issue and its file when the error holds one.
pub(super) fn failure_status(error: &anyhow::Error) -> String {
    let first = error
        .downcast_ref::<StudyAnonymizeError>()
        .and_then(first_issue);
    match first {
        Some((path, issue)) => format!(
            "De-identified export failed: {error:#}; first: {issue} in {}",
            path.display()
        ),
        None => format!("De-identified export failed: {error:#}"),
    }
}

fn selected_paths<'a>(choices: impl IntoIterator<Item = &'a SeriesChoice>) -> Vec<&'a Path> {
    choices
        .into_iter()
        .flat_map(|choice| choice.acquisition.file_paths.iter().map(PathBuf::as_path))
        .collect()
}

impl NativeViewerSession {
    /// Ask for an output folder and export the active DICOM selection into it.
    ///
    /// Returns whether the session changed; a cancelled dialog changes nothing.
    pub(super) fn anonymize_from_dialog(
        &mut self,
        scope: AnonymizationScope,
        level: Deidentification,
    ) -> Result<bool> {
        self.end_held_gestures();
        if self.anonymization_series_index().is_none() {
            self.active_app_mut().status_message =
                "Open a DICOM study from disk and select a series or a populated pane.".to_owned();
            return Ok(true);
        }
        let selected =
            pick(DialogSelection::Folder).context("show native de-identified export picker")?;
        let Some(output) = selected else {
            return Ok(false);
        };
        self.anonymize_to(&output, scope, level)?;
        Ok(true)
    }

    /// Export the cataloged instances of the active DICOM selection.
    pub(super) fn anonymize_to(
        &mut self,
        output: &Path,
        scope: AnonymizationScope,
        level: Deidentification,
    ) -> Result<()> {
        let path = self
            .study_root
            .as_deref()
            .ok_or_else(|| anyhow!("no DICOM source is open"))?;
        let input = classify_dicom_input_path(path);
        let source = input
            .dicom_root()
            .ok_or_else(|| anyhow!("no DICOM source is open"))?;
        let source = if source.as_os_str().is_empty() {
            Path::new(".")
        } else {
            source
        };
        let index = self
            .anonymization_series_index()
            .ok_or_else(|| anyhow!("no DICOM series is active"))?;
        let browser = self
            .series_browser
            .as_ref()
            .ok_or_else(|| anyhow!("no DICOM catalog is open"))?;
        let choice = browser
            .choice(index)
            .context("active DICOM series is outside the catalog")?;
        let instances = match scope {
            AnonymizationScope::Study => selected_paths(
                browser.study_choices(
                    choice
                        .study_uid
                        .as_deref()
                        .context("active series has no DICOM study UID")?,
                ),
            ),
            AnonymizationScope::Series => selected_paths([choice]),
        };
        let options = AnonymizeOptions::with_fresh_salt(level.profile())
            .context("draw a UID salt from the operating system")?;
        let report = anonymize_dicom_selection(source, &instances, output, &options)?;
        self.active_app_mut().status_message = export_status(scope, level, report, output);
        Ok(())
    }

    fn anonymization_series_index(&self) -> Option<usize> {
        self.study_root.as_ref()?;
        self.series_browser.as_ref()?;
        if self.workspace_layout.is_grid() && self.active_panel > 0 {
            return self.compare_panels.get(self.active_panel - 1)?.series_index;
        }
        self.primary_series_index
    }
}
