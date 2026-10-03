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
        let current = if panel_index == 0 {
            self.primary_series_index
        } else {
            self.compare_panels
                .get(panel_index.saturating_sub(1))
                .and_then(|panel| panel.series_index)
        }
        .or_else(|| Some(browser.active_index()))?;
        match virtual_key {
            0x24 if current != 0 => Some(0),
            0x23 if current.saturating_add(1) < browser.len() => {
                Some(browser.len().saturating_sub(1))
            }
            0x25 => current
                .checked_sub(1)
                .filter(|index| browser.choice(*index).is_some()),
            0x27 => current
                .checked_add(1)
                .filter(|index| browser.choice(*index).is_some()),
            _ => None,
        }
    }
}
