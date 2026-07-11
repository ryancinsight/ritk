//! DICOM series scanning and loading methods.

use tracing::{error, info};

use super::state::SnapApp;
use crate::dicom::select_hanging_protocol;
use crate::label::LabelEditor;
use crate::tools::interaction::ToolState;
use crate::ui::LinkedCursor;
use crate::ViewerState;

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

    /// Load a DICOM series from `path` through the canonical native loader.
    ///
    /// On success all viewer state and texture handles are reset; on failure
    /// `status_message` is updated and any previously loaded volume is
    /// preserved.
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

        match crate::dicom::loader::load_dicom_volume(&load_root) {
            Ok(volume) => {
                let shape = volume.shape;
                let protocol = select_hanging_protocol(
                    volume.modality.as_deref(),
                    volume.series_description.as_deref(),
                    shape,
                );

                let mut state = ViewerState::new();
                state.window_center = Some(protocol.window_center);
                state.window_width = Some(protocol.window_width);
                state.slice_index = shape[0] / 2;

                self.loaded = Some(volume);
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

        match crate::dicom::loader::load_dicom_volume(&load_root) {
            Ok(volume) => {
                let shape = volume.shape;
                let protocol = select_hanging_protocol(
                    volume.modality.as_deref(),
                    volume.series_description.as_deref(),
                    shape,
                );
                let modality = volume.modality;
                self.loaded_secondary = Some(volume);
                self.secondary_window_center = Some(protocol.window_center);
                self.secondary_window_width = Some(protocol.window_width);
                self.secondary_texture = None;
                self.secondary_texture_dirty = true;
                self.secondary_colormap = Self::colormap_for_modality(modality.as_deref());
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
