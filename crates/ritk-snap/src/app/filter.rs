use super::state::SnapApp;

mod native;

impl SnapApp {
    pub(crate) fn apply_filter_to_loaded_volume(&mut self) {
        let Some(volume) = self.loaded.as_ref() else {
            self.status_message = "No volume loaded.".to_owned();
            return;
        };

        let result = native::apply_if_supported(volume, &self.active_filter)
            .expect("invariant: every current Snap filter has a native implementation");
        match result {
            Err(error) => {
                self.status_message = format!("Filter failed: {error:#}");
            }
            Ok(output) => self.replace_loaded_volume_native(output) }
    }

    fn replace_loaded_volume_native(&mut self, output: native::NativeFilterOutput) {
        let volume = self
            .loaded
            .as_mut()
            .expect("invariant: a filter result exists only when a volume is loaded");
        volume.data = std::sync::Arc::new(output.data);
        volume.shape = output.shape;
        volume.origin = output.origin;
        volume.spacing = output.spacing;
        volume.direction = output.direction;
        self.mark_filter_applied();
    }

    fn mark_filter_applied(&mut self) {
        self.texture_dirty = true;
        self.coronal_dirty = true;
        self.sagittal_dirty = true;
        self.mip_dirty = true;
        self.status_message = "Filter applied.".to_owned();
    }
}
