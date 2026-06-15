//! Generic volume loading (file/bytes), study lifecycle, and histogram cache.

use std::sync::Arc;
use tracing::{error, info};

use super::state::{ProjectionMode, SeriesLoadTarget, SnapApp, DEFAULT_FUSION_ALPHA};
use crate::dicom::select_hanging_protocol;
use crate::label::LabelEditor;
use crate::render::colormap::Colormap;
use crate::tools::interaction::ToolState;
use crate::ui::LinkedCursor;
use crate::LoadedVolume;
use crate::ViewerState;

impl SnapApp {
    /// Apply a newly loaded [`LoadedVolume`] to the viewer state.
    ///
    /// Sets up the viewer state (slice index, W/L, axis selection, multi-planar),
    /// clears annotations, resets textures, and updates the status message.
    pub(crate) fn load_volume(&mut self, vol: LoadedVolume, status_msg: String) {
        let shape = vol.shape;
        let protocol = select_hanging_protocol(
            vol.modality.as_deref(),
            vol.series_description.as_deref(),
            shape,
        );

        let mut state = ViewerState::new();
        state.window_center = Some(protocol.window_center);
        state.window_width = Some(protocol.window_width);
        state.slice_index = shape[0] / 2;

        self.cine.stop();
        self.loaded = Some(vol);
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
        self.colormap =
            Self::colormap_for_modality(self.loaded.as_ref().and_then(|v| v.modality.as_deref()));
        self.texture = None;
        self.texture_dirty = true;
        self.coronal_tex = None;
        self.coronal_dirty = true;
        self.sagittal_tex = None;
        self.sagittal_dirty = true;
        self.mip_tex = None;
        self.mip_dirty = true;
        self.status_message = status_msg;
        self.refresh_cached_histogram();
        info!("{}", self.status_message);
    }

    pub(crate) fn load_volume_file(&mut self, path: std::path::PathBuf) {
        match crate::dicom::loader::load_volume_from_path(&path) {
            Ok(vol) => {
                let msg = format!(
                    "Loaded {} — shape {:?} — protocol {}",
                    path.display(),
                    vol.shape,
                    select_hanging_protocol(
                        vol.modality.as_deref(),
                        vol.series_description.as_deref(),
                        vol.shape,
                    )
                    .protocol_name
                );
                self.load_volume(vol, msg);
            }
            Err(e) => {
                self.status_message = format!("Volume load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a medical image volume from pathless dropped bytes.
    pub(crate) fn load_volume_bytes(&mut self, name_hint: String, bytes: &[u8]) {
        match crate::dicom::loader::load_volume_from_bytes(&name_hint, bytes) {
            Ok(vol) => {
                let msg = format!(
                    "Loaded dropped in-memory volume '{}' — shape {:?} — protocol {}",
                    name_hint,
                    vol.shape,
                    select_hanging_protocol(
                        vol.modality.as_deref(),
                        vol.series_description.as_deref(),
                        vol.shape,
                    )
                    .protocol_name
                );
                self.load_volume(vol, msg);
            }
            Err(e) => {
                self.status_message =
                    format!("Volume load failed for dropped '{}': {e:#}", name_hint);
                error!("{}", self.status_message);
            }
        }
    }

    /// Load a DICOM series from pathless dropped named byte payloads.
    pub(crate) fn load_dicom_series_bytes(&mut self, files: Vec<(String, Arc<[u8]>)>) {
        let borrowed: Vec<(String, &[u8])> = files
            .iter()
            .map(|(name, bytes)| (name.clone(), bytes.as_ref()))
            .collect();

        match crate::dicom::loader::load_dicom_series_from_named_bytes(&borrowed) {
            Ok(vol) => {
                let msg = format!(
                    "Loaded dropped in-memory DICOM series ({} files) — shape {:?} — protocol {}",
                    files.len(),
                    vol.shape,
                    select_hanging_protocol(
                        vol.modality.as_deref(),
                        vol.series_description.as_deref(),
                        vol.shape,
                    )
                    .protocol_name
                );
                self.load_volume(vol, msg);
            }
            Err(e) => {
                self.status_message =
                    format!("Volume load failed for dropped in-memory DICOM series: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Drop the currently loaded study and reset all study-owned state.
    pub(crate) fn close_study(&mut self) {
        self.loaded = None;
        self.loaded_secondary = None;
        self.series_load_target = SeriesLoadTarget::Primary;
        self.secondary_window_center = None;
        self.secondary_window_width = None;
        self.secondary_colormap = Colormap::Grayscale;
        self.multi_planar = false;
        self.dual_plane = false;
        self.compare_side_by_side = false;
        self.compare_fused_overlay = false;
        self.compare_fusion_alpha = DEFAULT_FUSION_ALPHA;
        self.compare_axes = [0, 0];
        self.dual_axes = [0, 1];
        self.annotations.clear();
        self.label_editor = None;
        self.rt_struct = None;
        self.rt_dose = None;
        self.rt_dose_max_gy = None;
        self.rt_plan = None;
        self.rt_dvh_selected_roi = None;
        self.rt_dvh_cache = None;
        self.clear_rt_dose_overlay_cache();
        self.viewer_state = ViewerState::new();
        self.linked_cursor = None;
        self.pointer_intensity = 0.0;
        self.pointer_suv = None;
        self.cached_histogram = None;
        self.selected_series = None;
        self.pan_offset = egui::Vec2::ZERO;
        self.zoom = 1.0;
        self.texture = None;
        self.secondary_texture = None;
        self.coronal_tex = None;
        self.sagittal_tex = None;
        self.mip_tex = None;
        self.projection_mode = ProjectionMode::Mip;
        self.texture_dirty = false;
        self.secondary_texture_dirty = false;
        self.secondary_texture_axis = 0;
        self.secondary_texture_slice = 0;
        self.coronal_dirty = false;
        self.sagittal_dirty = false;
        self.mip_dirty = false;
        self.cine.stop();
        self.status_message = "Study closed.".to_owned();
    }

    pub(crate) fn refresh_cached_histogram(&mut self) {
        use crate::render::histogram::compute_histogram;

        if let Some(vol) = &self.loaded {
            let data: &[f32] = &vol.data;

            // Single pass: compute exact (min, max) over all finite voxels.
            let (mut mn, mut mx) = (f32::MAX, f32::MIN);
            for &v in data {
                if v.is_finite() {
                    if v < mn {
                        mn = v;
                    }
                    if v > mx {
                        mx = v;
                    }
                }
            }

            // Guard against pathological all-NaN or empty data.
            if mn < mx {
                self.cached_histogram = Some(compute_histogram(data, mn, mx, 256));
            } else {
                self.cached_histogram = None;
            }
        } else {
            self.cached_histogram = None;
        }
    }
}
