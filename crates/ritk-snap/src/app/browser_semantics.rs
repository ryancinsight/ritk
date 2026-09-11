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
/// state, axis and slice selection, and the dimensions of the presented frame.
/// It deliberately excludes paths, identifiers, metadata and pixel values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    ) -> Self {
        debug_assert!(axis < 3, "RITK browser axes are limited to three planes");
        debug_assert!(slice_count > 0, "RITK browser slice counts are non-zero");
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
        }
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
