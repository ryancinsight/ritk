//! Shared window/level state transitions for every RITK presentation host.

use super::SnapApp;
use crate::ui::window_presets::WindowPreset;

/// Failure to save or apply a numbered custom window/level preset.
///
/// Windows-only: custom presets live in native Display-menu state, which is
/// compiled out on other platforms.
#[cfg(any(windows, target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum CustomPresetError {
    /// The requested slot is outside the numbered custom table.
    #[error("custom window preset slot {index} is outside the numbered table")]
    IndexOutOfRange {
        /// Rejected zero-based custom slot index.
        index: usize,
    },
    /// The requested slot was never saved.
    #[error("custom window preset slot {index} is empty")]
    SlotEmpty {
        /// Zero-based custom slot index with no saved window.
        index: usize,
    },
    /// Every custom slot already holds a saved window.
    #[error("every custom window preset slot is full")]
    NoEmptySlot,
}

impl SnapApp {
    /// Applies a validated clinical window preset and advances the pixel revision.
    pub(crate) fn apply_preset(&mut self, preset: WindowPreset) {
        self.viewer_state.window_center = Some(preset.center as f32);
        self.viewer_state.window_width = Some(preset.width as f32);
        self.bump_visual_revision();
    }

    /// Applies one saved custom window preset and reports whether pixels changed.
    #[cfg(any(windows, target_arch = "wasm32"))]
    pub(crate) fn apply_custom_preset(&mut self, index: usize) -> Result<bool, CustomPresetError> {
        let slot = self
            .custom_presets
            .get(index)
            .copied()
            .ok_or(CustomPresetError::IndexOutOfRange { index })?;
        let (center, width) = slot.ok_or(CustomPresetError::SlotEmpty { index })?;
        let current = self.browser_window_level_values();
        if current == (center, width) {
            return Ok(false);
        }
        self.viewer_state.window_center = Some(center);
        self.viewer_state.window_width = Some(width);
        self.bump_visual_revision();
        Ok(true)
    }

    /// Saves the current window into one custom slot and returns that slot.
    ///
    /// `None` fills the first empty slot; `Some(index)` overwrites that slot.
    #[cfg(any(windows, target_arch = "wasm32"))]
    pub(crate) fn save_custom_preset(
        &mut self,
        index: Option<usize>,
    ) -> Result<usize, CustomPresetError> {
        let values = self.browser_window_level_values();
        let slot = match index {
            Some(index) => {
                *self
                    .custom_presets
                    .get_mut(index)
                    .ok_or(CustomPresetError::IndexOutOfRange { index })? = Some(values);
                index
            }
            None => {
                let empty = self
                    .custom_presets
                    .iter()
                    .position(|slot| slot.is_none())
                    .ok_or(CustomPresetError::NoEmptySlot)?;
                self.custom_presets[empty] = Some(values);
                empty
            }
        };
        self.bump_visual_revision();
        Ok(slot)
    }

    /// Returns the custom slot matching the current window, if any.
    #[cfg(any(windows, target_arch = "wasm32"))]
    pub(crate) fn custom_preset_index(&self) -> Option<usize> {
        let current = self.browser_window_level_values();
        self.custom_presets
            .iter()
            .position(|slot| *slot == Some(current))
    }
}
