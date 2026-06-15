use super::state::SnapApp;
use tracing::{error, info};

#[cfg(not(target_arch = "wasm32"))]
use rfd::FileDialog;

impl SnapApp {
    pub(super) fn save_rt_struct_dialog(&mut self) {
        let (Some(vol), Some(editor)) = (self.loaded.as_ref(), self.label_editor.as_ref()) else {
            self.status_message = "Save RT-STRUCT: no volume or segmentation loaded.".to_owned();
            return;
        };

        let map = editor.current_map();

        let Some(path) = FileDialog::new()
            .set_file_name("rtstruct.dcm")
            .add_filter("DICOM RT-STRUCT", &["dcm"][..])
            .save_file()
        else {
            return;
        };

        let origin = vol.origin;
        let spacing = vol.spacing;
        let direction = vol.direction;

        match ritk_io::label_map_to_rt_struct(map, origin, spacing, direction) {
            Ok(ss) => match ritk_io::write_rt_struct(&path, &ss) {
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
