//! Typed browser access to the RITK window/level preset tables.

use super::SnapApp;
use crate::ui::window_presets::WindowPreset;
use crate::viewer::{DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH};

/// Failure to apply a browser window/level preset without changing state.
#[derive(Debug, thiserror::Error, PartialEq)]
pub(crate) enum BrowserWindowPresetError {
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
    /// The requested preset is outside the loaded modality's table.
    #[error("browser window preset index {index} is outside 0..{preset_count}")]
    IndexOutOfRange {
        /// Rejected zero-based preset index.
        index: usize,
        /// Number of presets available for the loaded modality.
        preset_count: usize,
    },
    /// A JavaScript number is not finite.
    #[error("browser window preset index {value} must be finite")]
    CoordinateNotFinite {
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript number does not denote an integer.
    #[error("browser window preset index {value} must be an integer")]
    CoordinateNotInteger {
        /// Rejected JavaScript number.
        value: f64,
    },
    /// A JavaScript integer cannot be represented by the WASM ABI.
    #[error("browser window preset index {value} is outside 0..={maximum}")]
    CoordinateOutOfRange {
        /// Rejected JavaScript number.
        value: f64,
        /// Inclusive upper bound of the WASM ABI integer representation.
        maximum: u32,
    },
}

/// Validates a JavaScript number before narrowing it to a preset index.
pub(crate) fn parse_browser_window_preset_request(
    value: f64,
) -> Result<usize, BrowserWindowPresetError> {
    if !value.is_finite() {
        return Err(BrowserWindowPresetError::CoordinateNotFinite { value });
    }
    if value.fract() != 0.0 {
        return Err(BrowserWindowPresetError::CoordinateNotInteger { value });
    }
    if value < 0.0 || value > f64::from(u32::MAX) {
        return Err(BrowserWindowPresetError::CoordinateOutOfRange {
            value,
            maximum: u32::MAX,
        });
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite integral input is bounded to the complete u32 range above"
    )]
    let value = value as u32;
    usize::try_from(value).map_err(|_| BrowserWindowPresetError::CoordinateOutOfRange {
        value: f64::from(value),
        maximum: u32::MAX,
    })
}

impl SnapApp {
    /// Returns the effective window centre and width used by browser frames.
    pub(crate) fn browser_window_level_values(&self) -> (f32, f32) {
        (
            self.viewer_state
                .window_center
                .unwrap_or(DEFAULT_WINDOW_CENTER),
            self.viewer_state
                .window_width
                .unwrap_or(DEFAULT_WINDOW_WIDTH)
                .max(1.0),
        )
    }

    /// Returns the preset table selected by the loaded DICOM modality.
    pub(crate) fn browser_window_presets(&self) -> &'static [WindowPreset] {
        let modality = self
            .loaded
            .as_ref()
            .and_then(|volume| volume.modality.as_ref().map(|value| value.as_str()));
        WindowPreset::for_modality(modality)
    }

    /// Returns the active preset index when the current values match a table entry.
    pub(crate) fn browser_window_preset_index(&self) -> Option<usize> {
        self.loaded.as_ref()?;
        let (center, width) = self.browser_window_level_values();
        self.browser_window_presets().iter().position(|preset| {
            preset.center == f64::from(center) && preset.width == f64::from(width)
        })
    }

    /// Applies one loaded-modality preset and reports whether pixels changed.
    pub(crate) fn apply_browser_window_preset(
        &mut self,
        index: usize,
    ) -> Result<bool, BrowserWindowPresetError> {
        if self.loaded.is_none() {
            return Err(BrowserWindowPresetError::StudyNotLoaded);
        }
        let presets = self.browser_window_presets();
        let preset = *presets
            .get(index)
            .ok_or(BrowserWindowPresetError::IndexOutOfRange {
                index,
                preset_count: presets.len(),
            })?;
        let current = self.browser_window_level_values();
        let next = (preset.center as f32, preset.width as f32);
        if current == next {
            return Ok(false);
        }
        self.apply_preset(preset);
        Ok(true)
    }

    /// Returns the number of presets available for the loaded modality.
    pub(crate) fn browser_window_preset_count(&self) -> Result<usize, BrowserWindowPresetError> {
        if self.loaded.is_none() {
            return Err(BrowserWindowPresetError::StudyNotLoaded);
        }
        Ok(self.browser_window_presets().len())
    }

    /// Returns one loaded-modality preset name for browser control construction.
    pub(crate) fn browser_window_preset_name(
        &self,
        index: usize,
    ) -> Result<&'static str, BrowserWindowPresetError> {
        if self.loaded.is_none() {
            return Err(BrowserWindowPresetError::StudyNotLoaded);
        }
        self.browser_window_presets()
            .get(index)
            .map(|preset| preset.name)
            .ok_or(BrowserWindowPresetError::IndexOutOfRange {
                index,
                preset_count: self.browser_window_presets().len(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::test_volume;
    use crate::viewer::DEFAULT_WINDOW_CENTER;

    #[test]
    fn parser_rejects_non_integral_and_unrepresentable_indices() {
        assert_eq!(
            parse_browser_window_preset_request(1.5),
            Err(BrowserWindowPresetError::CoordinateNotInteger { value: 1.5 })
        );
        assert!(matches!(
            parse_browser_window_preset_request(f64::NAN),
            Err(BrowserWindowPresetError::CoordinateNotFinite { value }) if value.is_nan()
        ));
        assert_eq!(
            parse_browser_window_preset_request(-1.0),
            Err(BrowserWindowPresetError::CoordinateOutOfRange {
                value: -1.0,
                maximum: u32::MAX,
            })
        );
    }

    #[test]
    fn unloaded_window_state_uses_display_defaults_without_a_preset() {
        let app = SnapApp::default();
        assert_eq!(
            app.browser_window_level_values(),
            (DEFAULT_WINDOW_CENTER, DEFAULT_WINDOW_WIDTH)
        );
        assert_eq!(app.browser_window_preset_index(), None);
        assert_eq!(
            app.browser_window_preset_count(),
            Err(BrowserWindowPresetError::StudyNotLoaded)
        );
    }

    #[test]
    fn valid_preset_changes_state_and_revision() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 3, 2]));
        let revision = app.visual_revision;
        assert_eq!(app.apply_browser_window_preset(4), Ok(true));
        assert_eq!(app.viewer_state.window_center, Some(-400.0));
        assert_eq!(app.viewer_state.window_width, Some(1500.0));
        assert!(app.visual_revision > revision);
        assert_eq!(app.browser_window_preset_index(), Some(4));
        assert_eq!(app.browser_window_preset_count(), Ok(14));
        assert_eq!(app.browser_window_preset_name(4), Ok("Lung"));
    }

    #[test]
    fn invalid_preset_leaves_state_and_revision_unchanged() {
        let mut app = SnapApp::default();
        app.loaded = Some(test_volume([4, 3, 2]));
        app.viewer_state.window_center = Some(77.0);
        app.viewer_state.window_width = Some(88.0);
        let state = app.viewer_state;
        let revision = app.visual_revision;
        assert_eq!(
            app.apply_browser_window_preset(usize::MAX),
            Err(BrowserWindowPresetError::IndexOutOfRange {
                index: usize::MAX,
                preset_count: WindowPreset::ct_presets().len(),
            })
        );
        assert_eq!(app.viewer_state, state);
        assert_eq!(app.visual_revision, revision);
    }
}
