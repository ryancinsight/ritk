//! Viewer session snapshots and segmentation edit history.

use super::state::SnapApp;
use super::volume_input::VolumeInput;
use crate::session::ViewerSessionSnapshot;
use crate::tools::interaction::{ToolState, ViewportOffset};
use crate::ui::{LinkedCursor, MAX_ZOOM, MIN_ZOOM};

impl SnapApp {
    pub(crate) fn undo_label_edit_shortcut(&mut self) {
        let Some(editor) = self.label_editor.as_mut() else {
            return;
        };
        if editor.undo() {
            self.status_message = "Segmentation undo.".to_owned();
        }
    }

    pub(crate) fn redo_label_edit_shortcut(&mut self) {
        let Some(editor) = self.label_editor.as_mut() else {
            return;
        };
        if editor.redo() {
            self.status_message = "Segmentation redo.".to_owned();
        }
    }

    pub(crate) fn session_snapshot(&self) -> ViewerSessionSnapshot {
        ViewerSessionSnapshot {
            format: crate::session::SessionFormat,
            source: self.loaded.as_ref().and_then(|volume| {
                if let Some(acquisition) = VolumeInput::acquisition(volume) {
                    Some(crate::session::StudySource::Dicom {
                        series_uid: acquisition.series_instance_uid().to_owned(),
                        files: acquisition.file_paths.clone(),
                    })
                } else {
                    volume.source.clone().map(crate::session::StudySource::Path)
                }
            }),
            viewer_state: self.viewer_state,
            colormap: self.colormap,
            axis: self.axis,
            active_tool: self.active_tool,
            multi_planar: self.multi_planar,
            show_overlay: self.show_overlay,
            show_crosshair: self.show_crosshair,
            show_rt_struct_overlay: self.show_rt_struct_overlay,
            show_rt_dose_overlay: self.show_rt_dose_overlay,
            rt_dose_opacity: self.rt_dose_opacity,
            show_series_browser: self.show_series_browser,
            sidebar_tab: self.sidebar_tab,
            coronal_slice: self.coronal_slice,
            sagittal_slice: self.sagittal_slice,
            pan_offset: [self.pan_offset.x(), self.pan_offset.y()],
            zoom: self.zoom,
            cine_enabled: self.cine.enabled,
            cine_fps: self.cine.fps,
            annotations: self.annotations.clone(),
        }
    }

    pub(crate) fn apply_session_snapshot(
        &mut self,
        mut snapshot: ViewerSessionSnapshot,
    ) -> anyhow::Result<()> {
        // Decode and validate before changing either presentation or loaded state.
        if let Some(source) = snapshot.source.take() {
            let volume = VolumeInput::restore(source)?.load()?;
            self.load_volume(volume, "Restored session acquisition".to_owned());
        }
        self.viewer_state = snapshot.viewer_state;
        self.colormap = snapshot.colormap;
        self.axis = snapshot.axis.min(2);
        self.active_tool = snapshot.active_tool;
        self.multi_planar = snapshot.multi_planar;
        self.show_overlay = snapshot.show_overlay;
        self.show_crosshair = snapshot.show_crosshair;
        self.show_rt_struct_overlay = snapshot.show_rt_struct_overlay;
        self.show_rt_dose_overlay = snapshot.show_rt_dose_overlay;
        self.rt_dose_opacity = snapshot.rt_dose_opacity;
        self.show_series_browser = snapshot.show_series_browser;
        self.sidebar_tab = snapshot.sidebar_tab;
        self.coronal_slice = snapshot.coronal_slice;
        self.sagittal_slice = snapshot.sagittal_slice;
        self.pan_offset = ViewportOffset::new(snapshot.pan_offset[0], snapshot.pan_offset[1]);
        self.zoom = snapshot.zoom.clamp(MIN_ZOOM, MAX_ZOOM);
        self.cine.restore(snapshot.cine_enabled, snapshot.cine_fps);
        self.annotations = snapshot.annotations;
        self.tool_state = ToolState::Idle;
        self.bump_visual_revision();
        self.linked_cursor = self.loaded.as_ref().map(|vol| {
            LinkedCursor::from_slices(
                vol.shape,
                self.viewer_state.slice_index,
                self.coronal_slice,
                self.sagittal_slice,
            )
        });
        Ok(())
    }
}
