//! Resolve queued keyboard series navigation against the loaded viewer state.

use super::super::NativeViewerSession;
use crate::presentation::PresentationModifiers;

impl NativeViewerSession {
    pub(super) fn resolve_keyboard_series(
        &self,
        virtual_key: u32,
        modifiers: PresentationModifiers,
        panel_index: usize,
    ) -> Option<usize> {
        if modifiers != PresentationModifiers::NONE {
            return None;
        }
        let browser = self.series_browser.as_ref()?;
        let displayed = if panel_index == 0 {
            self.primary_series_index
        } else {
            self.compare_panels
                .get(panel_index.saturating_sub(1))
                .and_then(|panel| panel.series_index)
        };
        browser.keyboard_target(virtual_key, displayed)
    }
}
