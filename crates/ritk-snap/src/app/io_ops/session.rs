//! Session save and restore.

use crate::app::state::SnapApp;

use tracing::{error, info};

use std::path::Path;

use super::dialog::FileDialog;

impl SnapApp {
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

        self.load_session_from_path(&path);
    }

    pub(crate) fn load_session_from_path(&mut self, path: &Path) {
        match crate::session::load_from_file(path)
            .and_then(|snapshot| self.apply_session_snapshot(snapshot))
        {
            Ok(()) => {
                self.status_message = format!("Loaded session from {}", path.display());
                info!("{}", self.status_message);
            }
            Err(e) => {
                self.status_message = format!("Session load failed for {}: {e:#}", path.display());
                error!("{}", self.status_message);
            }
        }
    }
}
