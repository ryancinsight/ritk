//! Export a de-identified copy of the study or series in the active pane.
//!
//! The viewer never rewrites the study it displays: the copy goes to a folder
//! the reader picks, and RITK refuses a folder that overlaps the source or
//! already holds files. Pixel data is copied as stored, so text burned into
//! the image survives; the status line says so after every completed export.

use super::series_browser::{SeriesBrowser, SeriesChoice};
use super::NativeViewerSession;
use crate::dicom::input_path::classify_dicom_input_path;
use anyhow::{anyhow, Context, Result};
use metis_platform::native::{pick, DialogSelection};
use ritk_io::format::dicom::anonymize::{
    anonymize_dicom_selection, verify::VerifyIssue, AnonymizationProfile, AnonymizeOptions,
    StudyAnonymizeError,
};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum AnonymizationScope {
    Study,
    Series,
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

/// Access every verification issue even when cleanup adds a second error.
fn verification_issues(error: &StudyAnonymizeError) -> &[(PathBuf, VerifyIssue)] {
    match error {
        StudyAnonymizeError::Verification(issues) => issues,
        StudyAnonymizeError::Retained { cause, .. } => verification_issues(cause),
        _ => &[],
    }
}

/// Report structural diagnostics unless any issue identifies the source patient.
/// The path attached to a structural issue can contain the same identifying text.
pub(super) fn failure_status(error: &anyhow::Error) -> String {
    let issues = error
        .downcast_ref::<StudyAnonymizeError>()
        .map(verification_issues)
        .unwrap_or_default();
    if issues.iter().any(|(_, issue)| {
        matches!(
            issue,
            VerifyIssue::PatientLeak { .. } | VerifyIssue::ResidualIdentifier { .. }
        )
    }) {
        return format!(
            "De-identified export failed: {error}; identifying metadata remains in the copy."
        );
    }
    match issues.first() {
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
        if self.anonymization_selection().is_none() {
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
        self.exported_copy = None;
        let (path, browser, choice) = self
            .anonymization_selection()
            .context("no cataloged DICOM series is active")?;
        let input = classify_dicom_input_path(path);
        let source = match input.dicom_root() {
            Some(root) if root.as_os_str().is_empty() => Path::new("."),
            Some(root) => root,
            None => return Err(anyhow!("no DICOM source is open")),
        };
        let (label, instances) = match scope {
            AnonymizationScope::Study => {
                let uid = choice
                    .study_uid
                    .as_deref()
                    .context("active series has no DICOM study UID")?;
                ("Active study", selected_paths(browser.study_choices(uid)))
            }
            AnonymizationScope::Series => ("Active series", selected_paths([choice])),
        };
        let options = AnonymizeOptions::with_fresh_salt(level.profile())
            .context("draw a UID salt from the operating system")?;
        let report = anonymize_dicom_selection(source, &instances, output, &options)?;
        self.exported_copy = Some(output.to_owned());
        self.active_app_mut().status_message = format!(
            "{label} ({}): {} DICOM instance(s) exported. Burned-in text remains. Destination {}",
            level.label(),
            report.instances_written,
            output.display(),
        );
        Ok(())
    }

    fn anonymization_selection(&self) -> Option<(&Path, &SeriesBrowser, &SeriesChoice)> {
        let path = self.study_root.as_deref()?;
        let browser = self.series_browser.as_ref()?;
        let index = if self.workspace_layout.is_grid() && self.active_panel > 0 {
            self.compare_panels.get(self.active_panel - 1)?.series_index
        } else {
            self.primary_series_index
        }?;
        Some((path, browser, browser.choice(index)?))
    }
}
