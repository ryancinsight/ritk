//! DICOM series scanning and loading methods.

use tracing::{error, info};

use super::state::SnapApp;
use crate::dicom::select_hanging_protocol;

impl SnapApp {
    pub(crate) fn process_pending_loads(&mut self) {
        if let Some(input) = self.pending_load.take() {
            self.load_primary(input);
        }
        if let Some(input) = self.pending_secondary_load.take() {
            self.load_secondary(input);
        }
    }

    pub(crate) fn scan_for_series(&mut self, folder: std::path::PathBuf) {
        match crate::dicom::loader::scan_folder_for_series(&folder) {
            Ok(tree) => {
                let n = tree.total_series();
                self.series_tree = tree;
                self.status_message = format!("Found {n} series in {}", folder.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                let msg = format!("Scan failed for {}: {e:#}", folder.display());
                error!("{msg}");
                self.status_message = msg;
            }
        }
    }

    /// Decode completely before replacing primary state.
    pub(crate) fn load_primary(&mut self, input: super::volume_input::VolumeInput) {
        match input.load() {
            Ok(volume) => {
                let message = format!("Loaded volume — shape {:?}", volume.shape);
                self.load_volume(volume, message);
            }
            Err(error) => {
                self.status_message = format!("Volume load failed: {error:#}");
                error!("{}", self.status_message);
            }
        }
    }

    /// Decode completely before replacing the comparison acquisition.
    pub(crate) fn load_secondary(&mut self, input: super::volume_input::VolumeInput) {
        match input.load() {
            Ok(volume) => {
                let shape = volume.shape;
                let protocol = select_hanging_protocol(
                    volume.modality.as_deref(),
                    volume.series_description.as_deref(),
                    shape,
                );
                let modality = volume.modality;
                self.selected_series = super::volume_input::VolumeInput::acquisition(&volume);
                self.loaded_secondary = Some(volume);
                self.secondary_window_center = Some(protocol.window_center);
                self.secondary_window_width = Some(protocol.window_width);
                self.secondary_texture = None;
                self.secondary_texture_dirty = true;
                self.secondary_colormap = Self::colormap_for_modality(modality.as_deref());
                self.compare_side_by_side = true;
                self.multi_planar = false;
                self.dual_plane = false;
                self.status_message = "Loaded secondary acquisition".to_owned();
            }
            Err(error) => self.status_message = format!("Secondary volume load failed: {error:#}"),
        }
    }
}
