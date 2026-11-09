use super::state::SnapApp;
use std::path::Path;
use tracing::{error, info};

impl SnapApp {
    pub(super) fn save_rt_struct_to(&mut self, path: &Path) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message = "Save RT-STRUCT: no volume or segmentation loaded.".to_owned();
            return;
        };

        let map = editor.current_map();

        let origin = vol.origin;
        let spacing = vol.spacing;
        let direction = vol.direction;

        match ritk_io::label_map_to_rt_struct(map, origin, spacing, direction) {
            Ok(ss) => match ritk_io::write_rt_struct(path, &ss) {
                Ok(()) => {
                    self.status_message = format!(
                        "Saved RT-STRUCT ({} ROIs) to {}",
                        ss.rois.len(),
                        path.display()
                    );
                    info!("{}", self.status_message);
                }
                Err(e) => {
                    self.status_message = format!("RT-STRUCT write failed: {e:#}");
                    error!("{}", self.status_message);
                }
            },
            Err(e) => {
                self.status_message = format!("RT-STRUCT conversion failed: {e:#}");
                error!("{}", self.status_message);
            }
        }
    }
}

#[cfg(test)]
#[path = "tests/rt_struct_export.rs"]
mod tests;
