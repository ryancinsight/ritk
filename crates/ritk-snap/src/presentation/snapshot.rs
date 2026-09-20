//! Host-neutral viewer state shared by native and browser presentations.

use crate::render::WindowLevel;
use crate::tools::interaction::ViewportOffset;

/// Value-semantic viewer state exposed beside host presentation frames.
///
/// The snapshot contains only state needed to correlate a host frame with the
/// viewer reducer. It excludes DICOM identifiers, paths, metadata, volume
/// storage and pixel bytes. Hosts can therefore compare native and browser
/// behavior without interpreting clinical data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresentationSnapshot {
    revision: u64,
    loaded: bool,
    axis: usize,
    slice_indices: [usize; 3],
    slice_counts: [usize; 3],
    window_level: WindowLevel,
    cine_enabled: bool,
    cine_fps: f32,
    zoom: f32,
    pan: ViewportOffset,
    window_preset_index: Option<usize>,
    active_tool_index: usize,
    active_tool_name: &'static str,
}

impl PresentationSnapshot {
    /// Returns the monotonic viewer revision represented by this snapshot.
    #[must_use]
    pub const fn revision(self) -> u64 {
        self.revision
    }

    /// Returns whether a volume is loaded for presentation.
    #[must_use]
    pub const fn loaded(self) -> bool {
        self.loaded
    }

    /// Returns the active orthogonal axis (`0` axial, `1` coronal, `2` sagittal).
    #[must_use]
    pub const fn axis(self) -> usize {
        self.axis
    }

    /// Returns the selected slice index for one orthogonal axis.
    #[must_use]
    pub const fn slice_index(self, axis: usize) -> Option<usize> {
        match axis {
            0..=2 => Some(self.slice_indices[axis]),
            _ => None,
        }
    }

    /// Returns the number of slices available for one orthogonal axis.
    #[must_use]
    pub const fn slice_count(self, axis: usize) -> Option<usize> {
        match axis {
            0..=2 => Some(self.slice_counts[axis]),
            _ => None,
        }
    }

    /// Returns all selected orthogonal slice indices in axis order.
    #[must_use]
    pub const fn slice_indices(self) -> [usize; 3] {
        self.slice_indices
    }

    /// Returns all orthogonal slice counts in axis order.
    #[must_use]
    pub const fn slice_counts(self) -> [usize; 3] {
        self.slice_counts
    }

    /// Returns the effective display window centre and width.
    #[must_use]
    pub const fn window_level(self) -> WindowLevel {
        self.window_level
    }

    /// Returns whether cine playback is enabled.
    #[must_use]
    pub const fn cine_enabled(self) -> bool {
        self.cine_enabled
    }

    /// Returns the bounded cine playback rate in frames per second.
    #[must_use]
    pub const fn cine_fps(self) -> f32 {
        self.cine_fps
    }

    /// Returns the current viewport zoom multiplier.
    #[must_use]
    pub const fn zoom(self) -> f32 {
        self.zoom
    }

    /// Returns the current viewport pan offset in display pixels.
    #[must_use]
    pub const fn pan(self) -> ViewportOffset {
        self.pan
    }

    /// Returns the active modality window preset, when one is selected.
    #[must_use]
    pub const fn window_preset_index(self) -> Option<usize> {
        self.window_preset_index
    }

    /// Returns the active interaction-tool index in the stable tool table.
    #[must_use]
    pub const fn active_tool_index(self) -> usize {
        self.active_tool_index
    }

    /// Returns the stable label for the active interaction tool.
    #[must_use]
    pub const fn active_tool_name(self) -> &'static str {
        self.active_tool_name
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) const fn with_window_preset_index(mut self, index: Option<usize>) -> Self {
        self.window_preset_index = index;
        self
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) const fn with_axis(mut self, axis: usize) -> Self {
        debug_assert!(
            axis < 3,
            "presentation axis is one of three orthogonal views"
        );
        self.axis = axis;
        self
    }

    pub(crate) fn from_parts(
        revision: u64,
        loaded: bool,
        axis: usize,
        slice_indices: [usize; 3],
        slice_counts: [usize; 3],
        window_level: WindowLevel,
        cine_enabled: bool,
        cine_fps: f32,
        zoom: f32,
        pan: ViewportOffset,
        window_preset_index: Option<usize>,
        active_tool_index: usize,
        active_tool_name: &'static str,
    ) -> Self {
        debug_assert!(
            axis < 3,
            "presentation axis is one of three orthogonal views"
        );
        debug_assert!(slice_counts.iter().all(|count| *count > 0));
        debug_assert!(cine_fps.is_finite() && (1.0..=60.0).contains(&cine_fps));
        debug_assert!(zoom.is_finite() && zoom > 0.0);
        debug_assert!(pan.x().is_finite() && pan.y().is_finite());
        Self {
            revision,
            loaded,
            axis,
            slice_indices,
            slice_counts,
            window_level,
            cine_enabled,
            cine_fps,
            zoom,
            pan,
            window_preset_index,
            active_tool_index,
            active_tool_name,
        }
    }
}
