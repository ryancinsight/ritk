//! Typed browser control for the host-neutral linked MPR crosshair.

use super::SnapApp;

/// Failure to change browser crosshair state without mutating the viewer.
#[derive(Debug, thiserror::Error, PartialEq)]
pub(crate) enum BrowserCrosshairError {
    /// No browser viewer is mounted.
    #[cfg(target_arch = "wasm32")]
    #[error("RITK browser viewer is not mounted")]
    ViewerNotMounted,
    /// Another browser callback currently owns the viewer state.
    #[cfg(target_arch = "wasm32")]
    #[error("RITK browser viewer is handling another callback")]
    ViewerBusy,
    /// The mounted viewer has not loaded a study.
    #[error("RITK browser viewer has no loaded study")]
    StudyNotLoaded,
}

impl SnapApp {
    /// Toggles the linked MPR crosshair for the loaded browser study.
    pub(crate) fn toggle_browser_crosshair(&mut self) -> Result<bool, BrowserCrosshairError> {
        if self.loaded.is_none() {
            return Err(BrowserCrosshairError::StudyNotLoaded);
        }
        self.show_crosshair = !self.show_crosshair;
        Ok(self.show_crosshair)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::test_volume;

    #[test]
    fn unloaded_crosshair_control_is_rejected_without_mutation() {
        let mut app = SnapApp::default();
        assert_eq!(
            app.toggle_browser_crosshair(),
            Err(BrowserCrosshairError::StudyNotLoaded)
        );
        assert!(!app.show_crosshair);
    }

    #[test]
    fn loaded_crosshair_control_toggles_only_visibility() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 3, 2]));
        let cursor = app.linked_cursor;
        assert_eq!(app.toggle_browser_crosshair(), Ok(true));
        assert!(app.show_crosshair);
        assert_eq!(app.linked_cursor, cursor);
        assert_eq!(app.toggle_browser_crosshair(), Ok(false));
        assert!(!app.show_crosshair);
    }
}
