//! Stable RITK-owned browser state used by visual and workflow drivers.

use crate::presentation::PresentationFrame;

/// Whether the RITK viewer has a primary volume for a browser canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BrowserLoadState {
    /// No primary volume is loaded.
    Empty,
    /// A primary volume is loaded and can supply viewer state.
    Ready,
}

/// Format-neutral semantic state published beside one RITK browser canvas.
///
/// The state contains only viewer evidence needed by a browser driver: load
/// state, axis and slice selection, the dimensions of the presented frame,
/// cine playback state and rate, and the effective window/level display values.
/// It deliberately excludes paths, identifiers, metadata and pixel values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BrowserCanvasSemantics {
    /// Whether RITK has a primary volume.
    pub(crate) load_state: BrowserLoadState,
    /// RITK axis represented by the canvas (`0` axial, `1` coronal, `2` sagittal).
    pub(crate) axis: usize,
    /// Zero-based slice index selected for the axis.
    pub(crate) slice_index: usize,
    /// Number of slices available on the axis.
    pub(crate) slice_count: usize,
    /// Presented frame dimensions, when a frame is available.
    pub(crate) frame_dimensions: Option<(u32, u32)>,
    /// Whether cine playback is enabled for the loaded study.
    pub(crate) cine_enabled: bool,
    /// Active cine playback rate in frames per second.
    pub(crate) cine_fps: f32,
    /// Effective window centre used for presentation.
    pub(crate) window_center: f32,
    /// Effective window width used for presentation.
    pub(crate) window_width: f32,
    /// Active modality preset, when the current values match one.
    pub(crate) window_preset_index: Option<usize>,
}

impl BrowserCanvasSemantics {
    /// Builds the browser evidence from RITK state and an optional frame.
    #[must_use]
    pub(crate) fn from_state(
        loaded: bool,
        axis: usize,
        slice_index: usize,
        slice_count: usize,
        frame: Option<&PresentationFrame>,
        cine_enabled: bool,
        cine_fps: f32,
        window_center: f32,
        window_width: f32,
        window_preset_index: Option<usize>,
    ) -> Self {
        debug_assert!(axis < 3, "RITK browser axes are limited to three planes");
        debug_assert!(slice_count > 0, "RITK browser slice counts are non-zero");
        debug_assert!(
            cine_fps.is_finite() && (1.0..=60.0).contains(&cine_fps),
            "RITK browser cine rate stays within the supported range"
        );
        Self {
            load_state: if loaded {
                BrowserLoadState::Ready
            } else {
                BrowserLoadState::Empty
            },
            axis,
            slice_index,
            slice_count,
            frame_dimensions: frame.map(|frame| (frame.width(), frame.height())),
            cine_enabled,
            cine_fps,
            window_center,
            window_width,
            window_preset_index,
        }
    }

    /// Returns the stable DOM value for the active cine rate.
    #[must_use]
    pub(crate) fn cine_fps_value(self) -> String {
        self.cine_fps.to_string()
    }

    /// Returns the stable DOM value for whether cine playback is enabled.
    #[must_use]
    pub(crate) const fn cine_enabled_value(self) -> &'static str {
        if self.cine_enabled {
            "true"
        } else {
            "false"
        }
    }

    /// Returns the stable DOM value for the effective window centre.
    #[must_use]
    pub(crate) fn window_center_value(self) -> String {
        self.window_center.to_string()
    }

    /// Returns the stable DOM value for the effective window width.
    #[must_use]
    pub(crate) fn window_width_value(self) -> String {
        self.window_width.to_string()
    }

    /// Returns the stable DOM value for the active preset index.
    #[must_use]
    pub(crate) fn window_preset_index_value(self) -> String {
        self.window_preset_index
            .map_or_else(String::new, |index| index.to_string())
    }

    /// Returns the stable DOM value for the load state.
    #[must_use]
    pub(crate) const fn load_state_value(self) -> &'static str {
        match self.load_state {
            BrowserLoadState::Empty => "empty",
            BrowserLoadState::Ready => "ready",
        }
    }

    /// Returns the stable DOM value for the frame state.
    #[must_use]
    pub(crate) const fn frame_state_value(self) -> &'static str {
        if self.frame_dimensions.is_some() {
            "presented"
        } else {
            "empty"
        }
    }

    /// Returns frame dimensions, using zero for an empty canvas state.
    #[must_use]
    pub(crate) const fn frame_dimensions_or_zero(self) -> (u32, u32) {
        match self.frame_dimensions {
            Some(dimensions) => dimensions,
            None => (0, 0),
        }
    }
}
