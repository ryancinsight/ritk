//! Rendered-slice, MPR, and clinical-distribution export.

use crate::app::clinical_distribution::{
    build_clinical_distribution_report, current_slice_path, distribution_root, media_root,
    mpr_root, report_path, summary_from_loaded_volume, ClinicalDistributionExportSummary,
};
use crate::app::state::SnapApp;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

use crate::render::{SliceRenderer, WindowLevel};
use crate::ui::{apply_to_rgba_into, plan_all_mpr_exports};
use crate::LoadedVolume;
use tracing::{error, info};

use std::path::Path;

/// Write row-major RGBA bytes as an opaque PNG, whatever the path's extension.
fn save_rgba_png(path: &Path, size: [usize; 2], rgba: &[u8]) -> anyhow::Result<()> {
    let rgb = rgba
        .chunks_exact(4)
        .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
        .collect();
    let width = u32::try_from(size[0])?;
    let height = u32::try_from(size[1])?;
    let image = image::RgbImage::from_raw(width, height, rgb)
        .ok_or_else(|| anyhow::anyhow!("buffer length mismatch"))?;
    image.save_with_format(path, image::ImageFormat::Png)?;
    Ok(())
}

impl SnapApp {
    pub(crate) fn export_current_slice_to(&mut self, path: &Path) {
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

        match self.save_rendered_slice_png(vol, self.axis, self.viewer_state.slice_index, wl, path)
        {
            Ok(()) => {
                self.status_message = format!("Exported slice PNG: {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("PNG export failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    pub(crate) fn export_all_mpr_slices_into(&mut self, root: &Path) {
        let Some(vol) = &self.loaded else {
            self.status_message = "No volume loaded; MPR export skipped.".to_owned();
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

        match self.export_all_mpr_slices_to(vol, wl, root) {
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
        let image = SliceRenderer::render(vol, axis, slice_index, wl, self.colormap);
        let mut transformed = Vec::new();
        let size = apply_to_rgba_into(
            image.size(),
            image.pixels(),
            self.view_transform,
            &mut transformed,
        )?;
        let rgba = if self.view_transform.is_identity() {
            image.pixels()
        } else {
            &transformed
        };
        save_rgba_png(path, size, rgba)
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

    pub(crate) fn export_clinical_distribution_into(&mut self, root: &Path) {
        match self.export_clinical_distribution_to(root) {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn png_writer_ignores_a_jpeg_extension() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("rendered.jpg");
        let rgba = [12, 34, 56, 255, 12, 34, 56, 255];

        save_rgba_png(&path, [2, 1], &rgba)?;

        let bytes = std::fs::read(&path)?;
        assert_eq!(bytes.get(..8), Some(b"\x89PNG\r\n\x1a\n".as_slice()));
        let decoded = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)?;
        assert_eq!(decoded.to_rgb8().into_raw(), vec![12, 34, 56, 12, 34, 56]);
        Ok(())
    }
}
