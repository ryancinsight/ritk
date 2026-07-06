//! DICOM series scanning and loading methods.

use std::sync::Arc;
use tracing::{error, info};

use super::state::{LoadBackend, SnapApp};
use crate::dicom::select_hanging_protocol;
use crate::label::LabelEditor;
use crate::tools::interaction::ToolState;
use crate::ui::LinkedCursor;
use crate::{LoadedVolume, ViewerState};

impl SnapApp {
    pub(crate) fn scan_for_series(&mut self, folder: std::path::PathBuf) {
        let scan_root = crate::dicom::classify_dicom_input_path(&folder)
            .dicom_root()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| folder.clone());

        match crate::dicom::loader::scan_folder_for_series(&scan_root) {
            Ok(tree) => {
                let n = tree.total_series();
                self.series_tree = tree;
                self.status_message = format!("Found {n} series in {}", scan_root.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                let msg = format!("Scan failed for {}: {e:#}", scan_root.display());
                error!("{msg}");
                self.status_message = msg;
            }
        }
    }

    // ── File loading ──────────────────────────────────────────────────────────

    /// Load a DICOM series from `path` using the NdArray CPU backend.
    ///
    /// Delegates to [`crate::dicom::loader::load_dicom_volume`], which wraps
    /// `ritk_io::load_dicom_series_with_metadata`. On success all viewer
    /// state and texture handles are reset; on failure `status_message` is
    /// updated and any previously loaded volume is preserved.
    pub(crate) fn load_from_path(&mut self, path: std::path::PathBuf) {
        let load_root = crate::dicom::classify_dicom_input_path(&path)
            .dicom_root()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| path.clone());

        info!(
            "loading DICOM series from {} (resolved root: {})",
            path.display(),
            load_root.display()
        );

        self.cine.stop();

        let device: <LoadBackend as ritk_image::tensor::Backend>::Device = Default::default();

        match ritk_io::load_dicom_series_with_metadata::<LoadBackend, _>(&load_root, &device) {
            Ok((image, meta)) => {
                let shape = image.shape();
                let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
                let origin = [image.origin()[0], image.origin()[1], image.origin()[2]];
                let dir = image.direction().inner();
                let direction = [
                    dir[(0, 0)],
                    dir[(0, 1)],
                    dir[(0, 2)],
                    dir[(1, 0)],
                    dir[(1, 1)],
                    dir[(1, 2)],
                    dir[(2, 0)],
                    dir[(2, 1)],
                    dir[(2, 2)],
                ];

                let data = match image.try_data_vec() {
                    Ok(v) => Arc::new(v),
                    Err(e) => {
                        let msg = format!("pixel data extraction failed: {e:?}");
                        error!("{msg}");
                        self.status_message = msg;
                        return;
                    }
                };

                let protocol = select_hanging_protocol(
                    meta.modality.as_deref(),
                    meta.series_description.as_deref(),
                    shape,
                );

                let mut state = ViewerState::new();
                state.window_center = Some(protocol.window_center);
                state.window_width = Some(protocol.window_width);
                state.slice_index = shape[0] / 2;

                let modality = meta.modality;
                let patient_name = meta.patient_name.clone();
                let patient_id = meta.patient_id.clone();
                let study_date = meta.study_date;
                let series_description = meta.series_description.clone();
                let series_time = meta.series_time;
                let patient_weight_kg = meta.patient_weight_kg;
                let injected_dose_bq = meta.radionuclide_total_dose_bq;
                let radionuclide_half_life_s = meta.radionuclide_half_life_s;
                let radiopharmaceutical_start_time = meta.radiopharmaceutical_start_time;
                let decay_correction = meta.decay_correction;

                self.loaded = Some(LoadedVolume {
                    data,
                    shape,
                    channels: 1,
                    spacing,
                    origin,
                    direction,
                    metadata: Some(Box::new(meta)),
                    source: Some(load_root.clone()),
                    modality,
                    patient_name,
                    patient_id,
                    study_date,
                    series_description,
                    series_time,
                    patient_weight_kg,
                    injected_dose_bq,
                    radionuclide_half_life_s,
                    radiopharmaceutical_start_time,
                    decay_correction,
                });
                self.viewer_state = state;
                self.axis = protocol.preferred_axis.min(2);
                self.coronal_slice = shape[1] / 2;
                self.sagittal_slice = shape[2] / 2;
                self.multi_planar = protocol.layout
                    == crate::dicom::hanging_protocol::LayoutSuggestion::MultiPlanarReformat;
                self.dual_plane = false;
                self.compare_side_by_side = false;
                self.compare_fused_overlay = false;
                self.linked_cursor = Some(LinkedCursor::from_slices(
                    shape,
                    self.viewer_state.slice_index,
                    self.coronal_slice,
                    self.sagittal_slice,
                ));
                self.annotations.clear();
                self.label_editor = Some(LabelEditor::new(shape));
                self.rt_struct = None;
                self.rt_dose = None;
                self.rt_dose_max_gy = None;
                self.rt_plan = None;
                self.rt_dvh_selected_roi = None;
                self.rt_dvh_cache = None;
                self.clear_rt_dose_overlay_cache();
                self.tool_state = ToolState::Idle;
                self.pan_offset = egui::Vec2::ZERO;
                self.zoom = 1.0;
                self.pointer_intensity = 0.0;
                self.pointer_suv = None;
                self.colormap = Self::colormap_for_modality(
                    self.loaded.as_ref().and_then(|v| v.modality.as_deref()),
                );
                self.texture = None;
                self.texture_dirty = true;
                self.coronal_tex = None;
                self.coronal_dirty = true;
                self.sagittal_tex = None;
                self.sagittal_dirty = true;
                self.mip_tex = None;
                self.mip_dirty = true;
                self.status_message = format!(
                    "Loaded {} (root {}) — shape [{}, {}, {}] — protocol {}",
                    path.display(),
                    load_root.display(),
                    shape[0],
                    shape[1],
                    shape[2],
                    protocol.protocol_name,
                );
                self.refresh_cached_histogram();
                info!("{}", self.status_message);
            }
            Err(e) => {
                let msg = format!(
                    "DICOM load failed for {} (root {}): {e:#}",
                    path.display(),
                    load_root.display()
                );
                error!("{msg}");
                self.status_message = msg;
            }
        }
    }

    pub(crate) fn load_secondary_from_path(&mut self, path: std::path::PathBuf) {
        let load_root = crate::dicom::classify_dicom_input_path(&path)
            .dicom_root()
            .map(std::path::Path::to_path_buf)
            .unwrap_or_else(|| path.clone());

        let device: <LoadBackend as ritk_image::tensor::Backend>::Device = Default::default();

        match ritk_io::load_dicom_series_with_metadata::<LoadBackend, _>(&load_root, &device) {
            Ok((image, meta)) => {
                let shape = image.shape();
                let spacing = [image.spacing()[0], image.spacing()[1], image.spacing()[2]];
                let origin = [image.origin()[0], image.origin()[1], image.origin()[2]];
                let dir = image.direction().inner();
                let direction = [
                    dir[(0, 0)],
                    dir[(0, 1)],
                    dir[(0, 2)],
                    dir[(1, 0)],
                    dir[(1, 1)],
                    dir[(1, 2)],
                    dir[(2, 0)],
                    dir[(2, 1)],
                    dir[(2, 2)],
                ];

                let data = match image.try_data_vec() {
                    Ok(v) => Arc::new(v),
                    Err(e) => {
                        self.status_message = format!("Secondary pixel extraction failed: {e:?}");
                        return;
                    }
                };

                self.loaded_secondary = Some(LoadedVolume {
                    data,
                    shape,
                    channels: 1,
                    spacing,
                    origin,
                    direction,
                    metadata: Some(Box::new(meta.clone())),
                    source: Some(load_root.clone()),
                    modality: meta.modality,
                    patient_name: meta.patient_name.clone(),
                    patient_id: meta.patient_id.clone(),
                    study_date: meta.study_date,
                    series_description: meta.series_description.clone(),
                    series_time: meta.series_time,
                    patient_weight_kg: meta.patient_weight_kg,
                    injected_dose_bq: meta.radionuclide_total_dose_bq,
                    radionuclide_half_life_s: meta.radionuclide_half_life_s,
                    radiopharmaceutical_start_time: meta.radiopharmaceutical_start_time,
                    decay_correction: meta.decay_correction,
                });

                let protocol = select_hanging_protocol(
                    meta.modality.as_deref(),
                    meta.series_description.as_deref(),
                    shape,
                );
                self.secondary_window_center = Some(protocol.window_center);
                self.secondary_window_width = Some(protocol.window_width);
                self.secondary_texture = None;
                self.secondary_texture_dirty = true;
                self.secondary_colormap = Self::colormap_for_modality(meta.modality.as_deref());
                self.compare_side_by_side = true;
                self.multi_planar = false;
                self.dual_plane = false;
                self.status_message = format!("Loaded secondary series: {}", load_root.display());
            }
            Err(e) => {
                self.status_message = format!(
                    "Secondary DICOM load failed for {}: {e:#}",
                    load_root.display()
                );
            }
        }
    }
}
