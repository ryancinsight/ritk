use super::clinical_distribution::{
    build_clinical_distribution_report, current_slice_path, distribution_root, media_root,
    mpr_root, report_path, summary_from_loaded_volume, ClinicalDistributionExportSummary,
};
use super::state::SnapApp;

use crate::render::slice_render::{SliceRenderer, WindowLevel};
use crate::ui::{apply_to_image, plan_all_mpr_exports};
use crate::LoadedVolume;
use tracing::{error, info};

use std::path::Path;

#[cfg(not(target_arch = "wasm32"))]
use rfd::FileDialog;

fn color_image_to_rgb_bytes(color_image: &egui::ColorImage) -> Vec<u8> {
    let mut rgb_bytes = Vec::with_capacity(color_image.pixels.len() * 3);
    for pixel in &color_image.pixels {
        rgb_bytes.extend_from_slice(&[pixel.r(), pixel.g(), pixel.b()]);
    }
    rgb_bytes
}

fn save_color_image_png(path: &Path, color_image: &egui::ColorImage) -> anyhow::Result<()> {
    let rgb_bytes = color_image_to_rgb_bytes(color_image);
    let [w, h] = color_image.size;
    image::RgbImage::from_raw(w as u32, h as u32, rgb_bytes)
        .ok_or_else(|| anyhow::anyhow!("buffer length mismatch"))
        .and_then(|img| img.save(path).map_err(anyhow::Error::from))
}

impl SnapApp {
    pub fn export_current_slice(&mut self) {
        let Some(vol) = &self.loaded else {
            return;
        };

        let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
        let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
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

        let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
        let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
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
        let wc = self.viewer_state.window_center.unwrap_or(128.0) as f64;
        let ww = self.viewer_state.window_width.unwrap_or(256.0).max(1.0) as f64;
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

    pub fn save_session_dialog(&mut self) {
        let Some(path) = FileDialog::new()
            .set_file_name("ritk-snap-session.json")
            .add_filter("JSON", &["json"][..])
            .save_file()
        else {
            return;
        };

        let snapshot = self.session_snapshot();

        match crate::session::save_to_file(&snapshot, &path) {
            Ok(()) => {
                self.status_message = format!("Saved session to {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Session save failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    pub fn load_session_dialog(&mut self) {
        let Some(path) = FileDialog::new()
            .add_filter("JSON", &["json"][..])
            .pick_file()
        else {
            return;
        };

        match crate::session::load_from_file(&path) {
            Ok(snapshot) => {
                self.apply_session_snapshot(snapshot);
                self.status_message = format!("Loaded session from {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Session load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    /// Save the current label map to a NIfTI file.
    ///
    /// Requires a loaded volume (for geometry) and an initialised label editor.
    /// The dialog is a no-op when either is absent; a status message explains
    /// the missing precondition.
    pub fn save_segmentation_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message = "Save segmentation: no volume or segmentation loaded.".to_owned();
            return;
        };

        let map = editor.current_map();

        let origin = [
            vol.origin[0] as f32,
            vol.origin[1] as f32,
            vol.origin[2] as f32,
        ];
        let spacing = [
            vol.spacing[0] as f32,
            vol.spacing[1] as f32,
            vol.spacing[2] as f32,
        ];
        let direction: [f32; 9] = std::array::from_fn(|i| vol.direction[i] as f32);

        let Some(path) = FileDialog::new()
            .set_file_name("segmentation.nii.gz")
            .add_filter("NIfTI", &["nii", "gz"][..])
            .save_file()
        else {
            return;
        };

        match ritk_io::write_nifti_labels(
            &path,
            map.as_slice(),
            map.shape,
            origin,
            spacing,
            direction,
        ) {
            Ok(()) => {
                self.status_message = format!("Saved segmentation to {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Segmentation save failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    pub fn save_segmentation_dicom_seg_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message = "Save DICOM-SEG: no volume or segmentation loaded.".to_owned();
            return;
        };

        let map = editor.current_map();
        let origin = vol.origin;
        let spacing = vol.spacing;
        let direction = vol.direction;

        let Some(path) = FileDialog::new()
            .set_file_name("segmentation.dcm")
            .add_filter("DICOM SEG", &["dcm"][..])
            .save_file()
        else {
            return;
        };

        match ritk_io::label_map_to_dicom_seg(map, origin, spacing, direction, true) {
            Ok(seg) => match ritk_io::write_dicom_seg(&path, &seg) {
                Ok(()) => {
                    self.status_message = format!("Saved DICOM-SEG to {}", path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message = format!("DICOM-SEG write failed: {e:#}");
                    error!("{}", self.status_message);
                }
            },
            Err(e) => {
                self.status_message = format!("DICOM-SEG conversion failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a NIfTI file and replace the current segmentation.
    pub fn load_segmentation_dialog(&mut self) {
        let Some(vol) = self.loaded.as_ref() else {
            self.status_message = "Load segmentation: no volume loaded.".to_owned();
            return;
        };
        let expected_shape = vol.shape;

        let Some(path) = FileDialog::new()
            .add_filter("NIfTI", &["nii", "gz"][..])
            .pick_file()
        else {
            return;
        };

        match ritk_io::read_nifti_labels(&path) {
            Ok((labels, shape)) => {
                if shape != expected_shape {
                    self.status_message = format!(
                        "Segmentation shape {:?} does not match volume {:?}",
                        shape, expected_shape
                    );
                    error!("{}", self.status_message);
                    return;
                }
                match ritk_core::annotation::LabelMap::from_data(
                    shape,
                    labels,
                    crate::label::default_label_table(),
                ) {
                    Ok(map) => {
                        self.label_editor = Some(crate::label::LabelEditor::from_label_map(map));
                        self.status_message =
                            format!("Loaded segmentation from {}", path.display());
                        info!("{}", self.status_message);
                    }
                    Err(e) => {
                        self.status_message = format!("Segmentation data error: {e}");
                        error!("{}", self.status_message);
                    }
                }
            }
            Err(e) => {
                self.status_message = format!("Segmentation load failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a DICOM-SEG file and replace the current segmentation.
    ///
    /// The reconstructed shape must match the currently loaded volume.
    pub fn load_segmentation_dicom_seg_file(&mut self, path: &std::path::Path) {
        let Some(vol) = self.loaded.as_ref() else {
            self.status_message = "Load DICOM-SEG: no volume loaded.".to_owned();
            return;
        };
        let expected_shape = vol.shape;

        match ritk_io::read_dicom_seg(path) {
            Ok(seg) => match ritk_io::dicom_seg_to_label_map(&seg) {
                Ok(map) => {
                    if map.shape != expected_shape {
                        self.status_message = format!(
                            "DICOM-SEG shape {:?} does not match volume {:?}",
                            map.shape, expected_shape
                        );
                        error!("{}", self.status_message);
                        return;
                    }
                    self.label_editor = Some(crate::label::LabelEditor::from_label_map(map));
                    self.status_message = format!("Loaded DICOM-SEG from {}", path.display());
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message = format!("DICOM-SEG decode failed: {e:#}");
                    error!("{}", self.status_message);
                }
            },
            Err(e) => {
                self.status_message = format!("DICOM-SEG load failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a label map from a DICOM-SEG file and replace the current segmentation.
    ///
    /// The reconstructed shape must match the currently loaded volume.
    pub fn load_segmentation_dicom_seg_dialog(&mut self) {
        let Some(path) = FileDialog::new()
            .add_filter("DICOM SEG", &["dcm", "dicom"][..])
            .pick_file()
        else {
            return;
        };
        self.load_segmentation_dicom_seg_file(&path);
    }
}
