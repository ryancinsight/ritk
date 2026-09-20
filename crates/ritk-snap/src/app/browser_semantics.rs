//! Stable RITK-owned browser state used by visual and workflow drivers.

use crate::presentation::{PresentationFrame, PresentationSnapshot};

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
    /// Shared RITK viewer state for this canvas.
    pub(crate) snapshot: PresentationSnapshot,
    /// Presented frame dimensions, when a frame is available.
    frame_dimensions: Option<(u32, u32)>,
}

impl BrowserCanvasSemantics {
    /// Builds browser evidence from one shared RITK snapshot and an optional frame.
    #[must_use]
    pub(crate) fn from_snapshot(
        snapshot: PresentationSnapshot,
        frame: Option<&PresentationFrame>,
    ) -> Self {
        Self {
            snapshot,
            frame_dimensions: frame.map(|frame| (frame.width(), frame.height())),
        }
    }

    /// Returns whether RITK has a primary volume.
    #[must_use]
    pub(crate) const fn load_state(self) -> BrowserLoadState {
        if self.snapshot.loaded() {
            BrowserLoadState::Ready
        } else {
            BrowserLoadState::Empty
        }
    }

    /// Returns the axis represented by the canvas.
    #[must_use]
    pub(crate) const fn axis(self) -> usize {
        self.snapshot.axis()
    }

    /// Returns the selected slice index for the canvas axis.
    #[must_use]
    pub(crate) const fn slice_index(self) -> usize {
        self.snapshot
            .slice_index(self.axis())
            .expect("invariant: browser snapshot axis has a slice index")
    }

    /// Returns the number of slices available for the canvas axis.
    #[must_use]
    pub(crate) const fn slice_count(self) -> usize {
        self.snapshot
            .slice_count(self.axis())
            .expect("invariant: browser snapshot axis has a slice count")
    }

    /// Returns the stable DOM value for the active cine rate.
    #[must_use]
    pub(crate) fn cine_fps_value(self) -> String {
        self.snapshot.cine_fps().to_string()
    }

    /// Returns the stable DOM value for whether cine playback is enabled.
    #[must_use]
    pub(crate) const fn cine_enabled_value(self) -> &'static str {
        if self.snapshot.cine_enabled() {
            "true"
        } else {
            "false"
        }
    }

    /// Returns the stable DOM value for the effective window centre.
    #[must_use]
    pub(crate) fn window_center_value(self) -> String {
        self.snapshot.window_level().center.to_string()
    }

    /// Returns the stable DOM value for the effective window width.
    #[must_use]
    pub(crate) fn window_width_value(self) -> String {
        self.snapshot.window_level().width.to_string()
    }

    /// Returns the stable DOM value for the active preset index.
    #[must_use]
    pub(crate) fn window_preset_index_value(self) -> String {
        self.snapshot
            .window_preset_index()
            .map_or_else(String::new, |index| index.to_string())
    }

    /// Returns the stable DOM value for the active interaction-tool index.
    #[must_use]
    pub(crate) fn active_tool_index_value(self) -> String {
        self.snapshot.active_tool_index().to_string()
    }

    /// Returns the stable DOM value for the load state.
    #[must_use]
    pub(crate) const fn load_state_value(self) -> &'static str {
        match self.load_state() {
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

    /// Returns the stable active interaction-tool label.
    #[must_use]
    pub(crate) const fn active_tool_name(self) -> &'static str {
        self.snapshot.active_tool_name()
    }
}
