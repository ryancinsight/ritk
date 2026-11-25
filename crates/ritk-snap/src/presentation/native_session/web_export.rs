//! Browser delivery of verified DICOM copies. Only neutral Part 10 instances enter the ZIP.
use super::{
    export_profile::{AnonymizationScope, Deidentification},
    NativeViewerSession, SeriesBrowser,
};
use anyhow::{Context, Result};
use ritk_io::format::dicom::anonymize::{anonymize_dicom_buffers, AnonymizeOptions};
use std::io::{Cursor, Write};
use zip::{write::SimpleFileOptions, ZipWriter};

pub(super) struct WebExport {
    files: Vec<(String, Vec<u8>)>,
    pub(super) download: Option<Vec<u8>>,
    pub(super) filename: &'static str,
}

impl NativeViewerSession {
    pub(super) fn prepare_web_export(
        &mut self,
        scope: AnonymizationScope,
        level: Deidentification,
    ) -> Result<()> {
        self.web_export = None;
        let browser = self.series_browser.as_ref().context("no DICOM study")?;
        let index = if self.workspace_layout.is_grid() && self.active_panel > 0 {
            self.compare_panels
                .get(self.active_panel - 1)
                .and_then(|pane| pane.series_index)
        } else {
            self.primary_series_index
        }
        .context("no active series")?;
        let choice = browser.choice(index).context("missing selected series")?;
        let mut files = Vec::new();
        match scope {
            AnonymizationScope::Series => {
                files.extend(browser.export_files(choice)?.collect::<Result<Vec<_>>>()?)
            }
            AnonymizationScope::Study => {
                let uid = choice.study_uid.as_deref().context("missing study UID")?;
                for series in browser.study_choices(uid) {
                    files.extend(browser.export_files(series)?.collect::<Result<Vec<_>>>()?);
                }
            }
        }
        let options = AnonymizeOptions::with_fresh_salt(level.profile())?;
        let files = anonymize_dicom_buffers(files, &options)?;
        let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
        let options =
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        for (name, payload) in &files {
            zip.start_file(name, options)?;
            zip.write_all(payload)?;
        }
        let download = Some(zip.finish()?.into_inner());
        let filename = match level {
            Deidentification::Basic => "AsterSlice-Basic.zip",
            Deidentification::Strict => "AsterSlice-Strict.zip",
        };
        self.active_app_mut().status_message = format!(
            "{} copy prepared for download ({} DICOM instances). Burned-in text remains.",
            level.label(),
            files.len()
        );
        self.web_export = Some(WebExport {
            files,
            download,
            filename,
        });
        Ok(())
    }

    pub(super) fn reopen_web_export(&mut self) -> Result<()> {
        let export = self.web_export.as_ref().context("no completed export")?;
        let files: Vec<_> = export
            .files
            .iter()
            .map(|(name, bytes)| (name.as_str(), bytes.as_slice()))
            .collect();
        self.open_catalog(SeriesBrowser::from_bytes(&files)?)
    }
}
