//! Rendered-slice, MPR, and clinical-distribution export.

use crate::app::clinical_distribution::{
    build_clinical_distribution_report, current_slice_path, distribution_root, media_root,
    mpr_root, report_path, summary_from_loaded_volume, ClinicalDistributionExportSummary,
};
use crate::app::state::SnapApp;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

use crate::render::{SliceRenderer, WindowLevel};
use crate::ui::{apply_to_image, plan_all_mpr_exports};
use crate::LoadedVolume;
use tracing::{error, info};

use std::path::Path;

use super::dialog::{save_color_image_png, FileDialog};

impl SnapApp {
    pub fn export_current_slice(&mut self) {
        let Some(vol) = &self.loaded else {
            return;
        };

        let wc = self
            .viewer_state
            .window_center
            .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
        let ww = self
            .viewer_state
            .window_width
            .unwrap_or(DEFAULT_WINDOW_WIDTH)
            .max(1.0) as f64;
        let wl = WindowLevel::new(wc, ww);

        if let Some(path) = FileDialog::new()
            .set_file_name("slice.png")
            .add_filter("PNG", &["png"][..])
            .save_file()
        {
            match self.save_rendered_slice_png(
                vol,
                self.axis,
                self.viewer_state.slice_index,
                wl,
                &path,
            ) {
                Ok(()) => {
                    self.status_message = format!("Exported slice PNG: {}", path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message =
                        format!("PNG export failed for {}: {e:#}", path.display());
                    error!("{}", self.status_message);
                }
            }
        }
    }

    pub fn export_all_mpr_slices(&mut self) {
        let Some(vol) = &self.loaded else {
            self.status_message = "No volume loaded; MPR export skipped.".to_owned();
            return;
        };

        let Some(root) = FileDialog::new().pick_folder() else {
            return;
        };

        let wc = self
            .viewer_state
            .window_center
            .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
        let ww = self
            .viewer_state
            .window_width
            .unwrap_or(DEFAULT_WINDOW_WIDTH)
            .max(1.0) as f64;
        let wl = WindowLevel::new(wc, ww);

        match self.export_all_mpr_slices_to(vol, wl, &root) {
            Ok((success, failed)) => {
                self.status_message = format!(
                    "MPR export complete: {} succeeded, {} failed ({})",
                    success,
                    failed,
                    root.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("MPR export failed for {}: {e:#}", root.display());
                error!("{}", self.status_message);
            }
        }
    }

    fn save_rendered_slice_png(
        &self,
        vol: &LoadedVolume,
        axis: usize,
        slice_index: usize,
        wl: WindowLevel,
        path: &Path,
    ) -> anyhow::Result<()> {
        let color_image = SliceRenderer::render(vol, axis, slice_index, wl, self.colormap);
        let color_image = apply_to_image(&color_image, self.view_transform);
        save_color_image_png(path, &color_image)
    }

    fn export_all_mpr_slices_to(
        &self,
        vol: &LoadedVolume,
        wl: WindowLevel,
        root: &Path,
    ) -> anyhow::Result<(usize, usize)> {
        let plan = plan_all_mpr_exports(vol.shape);
        let mut success = 0usize;
        let mut failed = 0usize;

        for export in plan {
            let axis_dir = root.join(export.axis_folder);
            if let Err(e) = std::fs::create_dir_all(&axis_dir) {
                failed += 1;
                error!(path = %axis_dir.display(), error = %e, "failed to create axis export directory");
                continue;
            }

            let path = axis_dir.join(export.file_name);
            match self.save_rendered_slice_png(vol, export.axis, export.slice_index, wl, &path) {
                Ok(()) => success += 1,
                Err(e) => {
                    failed += 1;
                    error!(path = %path.display(), error = %e, "failed to export MPR PNG slice");
                }
            }
        }

        Ok((success, failed))
    }

    pub(crate) fn export_clinical_distribution_to(
        &self,
        base: &Path,
    ) -> anyhow::Result<ClinicalDistributionExportSummary> {
        let Some(vol) = &self.loaded else {
            return Err(anyhow::anyhow!(
                "clinical distribution requires a loaded volume"
            ));
        };

        let root = distribution_root(base);
        let media = media_root(&root);
        let mpr = mpr_root(&root);
        std::fs::create_dir_all(&mpr)?;
        std::fs::create_dir_all(&media)?;

        let summary = summary_from_loaded_volume(
            vol,
            &self.viewer_state,
            self.axis,
            self.colormap,
            self.active_tool,
            self.annotations.len(),
            self.label_editor.is_some(),
            self.rt_struct.is_some(),
            self.rt_dose.is_some(),
        );
        let report = build_clinical_distribution_report(&summary);
        let report_path_buf = report_path(&root);
        std::fs::write(&report_path_buf, report)?;

        let current_slice_path_buf = current_slice_path(&root);
        let wc = self
            .viewer_state
            .window_center
            .unwrap_or(DEFAULT_WINDOW_CENTER) as f64;
        let ww = self
            .viewer_state
            .window_width
            .unwrap_or(DEFAULT_WINDOW_WIDTH)
            .max(1.0) as f64;
        let wl = WindowLevel::new(wc, ww);
        self.save_rendered_slice_png(
            vol,
            self.axis,
            self.viewer_state.slice_index,
            wl,
            &current_slice_path_buf,
        )?;
        let (mpr_written, mpr_failed) = self.export_all_mpr_slices_to(vol, wl, &mpr)?;

        Ok(ClinicalDistributionExportSummary {
            root,
            report_path: report_path_buf,
            current_slice_path: current_slice_path_buf,
            mpr_root: mpr,
            current_slice_written: true,
            mpr_written,
            mpr_failed,
        })
    }

    pub fn export_clinical_distribution_dialog(&mut self) {
        let Some(root) = FileDialog::new().pick_folder() else {
            return;
        };

        match self.export_clinical_distribution_to(&root) {
            Ok(summary) => {
                self.status_message = format!(
                    "Clinical distribution package exported: report={}, current slice={}, {} MPR slices written, {} failed ({})",
                    summary.report_path.display(),
                    summary.current_slice_path.display(),
                    summary.mpr_written,
                    summary.mpr_failed,
                    summary.root.display()
                );
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Clinical distribution export failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }
}
