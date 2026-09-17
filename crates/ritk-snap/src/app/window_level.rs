//! Shared window/level state transitions for every RITK presentation host.

use super::SnapApp;
use crate::ui::window_presets::WindowPreset;

impl SnapApp {
    /// Applies a validated clinical window preset and advances the pixel revision.
    pub(crate) fn apply_preset(&mut self, preset: WindowPreset) {
        self.viewer_state.window_center = Some(preset.center as f32);
        self.viewer_state.window_width = Some(preset.width as f32);
        self.bump_visual_revision();
    }
}
