//! Write a repaired copy of the open study from the native shell.
//!
//! The copy goes to a folder the reader picks; the displayed study is not
//! rewritten, and RITK refuses a folder that overlaps the source or already
//! holds files.

use super::NativeViewerSession;
use anyhow::{anyhow, Context, Result};
use metis_platform::native::{pick, DialogSelection};
use ritk_io::format::dicom::repair::{repair_dicom_study, StudyRepairReport};
use std::path::Path;

pub(super) fn repair_status(report: StudyRepairReport, output: &Path) -> String {
    let left_out = report.non_dicom_skipped + report.indexes_omitted;
    let omitted = if left_out == 0 {
        String::new()
    } else {
        format!(" {left_out} file(s) left out (not DICOM, or a DICOMDIR).")
    };
    format!(
        "Wrote {} instance(s) to {}; {} changed: {} date(s), {} time(s) and {} UID(s) rewritten, \
         file meta UIDs aligned in {} instance(s).{omitted}",
        report.instances_written,
        output.display(),
        report.instances_repaired,
        report.values.dates,
        report.values.times,
        report.values.uids,
        report.file_meta_uids_aligned,
    )
}

impl NativeViewerSession {
    /// Ask for an output folder and write a repaired copy of the open study
    /// into it.
    ///
    /// Returns whether the session changed; a cancelled dialog changes nothing.
    pub(super) fn repair_study_from_dialog(&mut self) -> Result<bool> {
        self.end_held_gestures();
        if self.study_root.is_none() {
            self.active_app_mut().status_message =
                "Open a study from disk before writing a repaired copy.".to_owned();
            return Ok(true);
        }
        let selected = pick(DialogSelection::Folder).context("show native repaired copy picker")?;
        let Some(output) = selected else {
            return Ok(false);
        };
        self.repair_study_to(&output)?;
        Ok(true)
    }

    /// Write a repaired copy of the open study into `output`.
    pub(super) fn repair_study_to(&mut self, output: &Path) -> Result<()> {
        let source = self
            .study_root
            .as_deref()
            .ok_or_else(|| anyhow!("no study folder is open"))?;
        let report = repair_dicom_study(source, output)?;
        self.active_app_mut().status_message = repair_status(report, output);
        Ok(())
    }
}
