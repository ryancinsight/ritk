//! DICOM series scanning and loading methods.

use tracing::{error, info};

use super::state::SnapApp;
use super::volume_state::metadata_window_level;
use crate::dicom::select_hanging_protocol;

impl SnapApp {
    pub(crate) fn process_pending_loads(&mut self) {
        if let Some(input) = self.pending_load.take() {
            self.queue_primary(input);
        }
        if let Some(input) = self.pending_secondary_load.take() {
            self.queue_secondary(input);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn queue_primary(&mut self, input: super::volume_input::VolumeInput) {
        self.queue_load(super::load_tasks::LoadTarget::Primary, input);
    }

    #[cfg(target_arch = "wasm32")]
    fn queue_primary(&mut self, input: super::volume_input::VolumeInput) {
        self.load_primary(input);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn queue_secondary(&mut self, input: super::volume_input::VolumeInput) {
        self.queue_load(super::load_tasks::LoadTarget::Secondary, input);
    }

    #[cfg(target_arch = "wasm32")]
    fn queue_secondary(&mut self, input: super::volume_input::VolumeInput) {
        self.load_secondary(input);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn queue_load(
        &mut self,
        target: super::load_tasks::LoadTarget,
        input: super::volume_input::VolumeInput,
    ) {
        let slot = match target {
            super::load_tasks::LoadTarget::Primary => 0,
            super::load_tasks::LoadTarget::Secondary => 1,
        };
        let Some(generation) = self.load_generations[slot].checked_add(1) else {
            self.report_load_error(target, "load generation exhausted".to_owned());
            return;
        };
        self.load_generations[slot] = generation;
        if let Some(task) = self.load_tasks[slot].take() {
            task.cancellation.cancel();
        }
        self.load_tasks[slot] = Some(super::load_tasks::LoadTask::spawn(
            target,
            self.load_generations[slot],
            input,
        ));
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn poll_load_tasks(&mut self) {
        for slot in 0..self.load_tasks.len() {
            let finished = self.load_tasks[slot]
                .as_ref()
                .is_some_and(|task| task.handle.is_finished());
            if !finished {
                continue;
            }
            let Some(task) = self.load_tasks[slot].take() else {
                continue;
            };
            self.publish_task(slot, task);
        }
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn wait_for_load_tasks(&mut self) {
        for slot in 0..self.load_tasks.len() {
            let Some(task) = self.load_tasks[slot].take() else {
                continue;
            };
            self.publish_task(slot, task);
        }
    }

    #[cfg(all(test, target_arch = "wasm32"))]
    pub(crate) fn wait_for_load_tasks(&mut self) {}

    #[cfg(not(target_arch = "wasm32"))]
    fn publish_task(&mut self, slot: usize, task: super::load_tasks::LoadTask) {
        let result = task.handle.join();
        if task.generation != self.load_generations[slot] {
            return;
        }
        let Some(result) = result else {
            self.report_load_error(task.target, "load task returned no result".to_owned());
            return;
        };
        match result {
            Ok(Ok(volume)) => self.publish_loaded(task.target, volume),
            Ok(Err(error)) => self.report_load_error(task.target, format!("{error:#}")),
            Err(error) => self.report_load_error(task.target, error.to_string()),
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn poll_load_tasks(&mut self) {}

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn cancel_load_tasks(&mut self) {
        for task in &self.load_tasks {
            if let Some(task) = task {
                task.cancellation.cancel();
            }
        }
        self.load_tasks = std::array::from_fn(|_| None);
        self.pending_load = None;
        self.pending_secondary_load = None;
        for generation in &mut self.load_generations {
            *generation = next_generation(*generation);
        }
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn cancel_load_tasks(&mut self) {
        self.pending_load = None;
        self.pending_secondary_load = None;
    }

    fn publish_loaded(
        &mut self,
        target: super::load_tasks::LoadTarget,
        volume: crate::LoadedVolume,
    ) {
        match target {
            super::load_tasks::LoadTarget::Primary => {
                let message = format!("Loaded volume — shape {:?}", volume.shape);
                self.load_volume(volume, message);
            }
            super::load_tasks::LoadTarget::Secondary => self.publish_secondary(volume),
        }
    }

    fn publish_secondary(&mut self, volume: crate::LoadedVolume) {
        let shape = volume.shape;
        let protocol = select_hanging_protocol(
            volume.modality.as_deref(),
            volume.series_description.as_deref(),
            shape,
        );
        let (window_center, window_width) = metadata_window_level(&volume)
            .unwrap_or((protocol.window_center, protocol.window_width));
        let modality = volume.modality.clone();
        self.selected_series = super::volume_input::VolumeInput::acquisition(&volume);
        self.loaded_secondary = Some(volume);
        self.secondary_window_center = Some(window_center);
        self.secondary_window_width = Some(window_width);
        self.secondary_texture = None;
        self.secondary_texture_dirty = true;
        self.secondary_colormap = Self::colormap_for_modality(modality.as_deref());
        self.compare_side_by_side = true;
        self.multi_planar = false;
        self.dual_plane = false;
        self.status_message = "Loaded secondary acquisition".to_owned();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn report_load_error(&mut self, target: super::load_tasks::LoadTarget, error: String) {
        let message = match target {
            super::load_tasks::LoadTarget::Primary => format!("Volume load failed: {error}"),
            super::load_tasks::LoadTarget::Secondary => {
                format!("Secondary volume load failed: {error}")
            }
        };
        error!("{message}");
        self.status_message = message;
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
    #[cfg(any(test, target_arch = "wasm32"))]
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
    #[cfg(any(test, target_arch = "wasm32"))]
    pub(crate) fn load_secondary(&mut self, input: super::volume_input::VolumeInput) {
        match input.load() {
            Ok(volume) => self.publish_secondary(volume),
            Err(error) => self.status_message = format!("Secondary volume load failed: {error:#}"),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn next_generation(generation: u64) -> u64 {
    match generation.checked_add(1) {
        Some(next) => next,
        None => u64::MAX,
    }
}
