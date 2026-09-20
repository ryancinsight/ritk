//! Host-neutral viewer state shared by native and browser presentations.

use crate::render::WindowLevel;
use crate::tools::interaction::ViewportOffset;
use crate::ui::ViewTransform;

/// Stable kind labels for completed browser-visible annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum AnnotationKind {
    /// A two-point physical length measurement.
    Length,
    /// A three-point angle measurement.
    Angle,
    /// A rectangle region-of-interest statistic.
    RoiRect,
    /// An ellipse region-of-interest statistic.
    RoiEllipse,
    /// A single-point intensity measurement.
    HuPoint,
}

impl AnnotationKind {
    /// Returns the stable lower-case DOM label for this annotation kind.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Length => "length",
            Self::Angle => "angle",
            Self::RoiRect => "roi-rect",
            Self::RoiEllipse => "roi-ellipse",
            Self::HuPoint => "hu-point",
        }
    }
}

/// The input-sensitive value published for the most recently completed annotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AnnotationSummary {
    kind: AnnotationKind,
    primary_value: f32,
}

impl AnnotationSummary {
    /// Constructs a summary after checking that its displayed value is finite.
    pub(crate) fn new(kind: AnnotationKind, primary_value: f32) -> Self {
        debug_assert!(primary_value.is_finite());
        Self {
            kind,
            primary_value,
        }
    }

    /// Returns the completed annotation kind.
    #[must_use]
    pub const fn kind(self) -> AnnotationKind {
        self.kind
    }

    /// Returns the primary computed value in the kind's documented units.
    #[must_use]
    pub const fn primary_value(self) -> f32 {
        self.primary_value
    }
}

/// Value-semantic viewer state exposed beside host presentation frames.
///
/// The snapshot contains only state needed to correlate a host frame with the
/// viewer reducer. It excludes DICOM identifiers, paths, metadata, volume
/// storage and pixel bytes. Hosts can therefore compare native and browser
/// behavior without interpreting clinical data. Completed annotations are
/// represented only by their bounded kind, count, and primary computed value.
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
    view_transform: ViewTransform,
    crosshair_visible: bool,
    linked_cursor_voxel: Option<[usize; 3]>,
    window_preset_index: Option<usize>,
    active_tool_index: usize,
    active_tool_name: &'static str,
    annotation_count: usize,
    last_annotation: Option<AnnotationSummary>,
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

    /// Returns the orientation applied to each presented slice.
    #[must_use]
    pub const fn view_transform(self) -> ViewTransform {
        self.view_transform
    }

    /// Returns whether hosts should render the linked MPR crosshair.
    #[must_use]
    pub const fn crosshair_visible(self) -> bool {
        self.crosshair_visible
    }

    /// Returns the linked cursor in volume voxel order `[z, y, x]`.
    #[must_use]
    pub const fn linked_cursor_voxel(self) -> Option<[usize; 3]> {
        self.linked_cursor_voxel
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

    /// Returns the number of completed annotations stored by the viewer.
    #[must_use]
    pub const fn annotation_count(self) -> usize {
        self.annotation_count
    }

    /// Returns the most recently completed annotation and its computed value.
    #[must_use]
    pub const fn last_annotation(self) -> Option<AnnotationSummary> {
        self.last_annotation
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
        view_transform: ViewTransform,
        crosshair_visible: bool,
        linked_cursor_voxel: Option<[usize; 3]>,
        window_preset_index: Option<usize>,
        active_tool_index: usize,
        active_tool_name: &'static str,
        annotation_count: usize,
        last_annotation: Option<AnnotationSummary>,
    ) -> Self {
        debug_assert!(
            axis < 3,
            "presentation axis is one of three orthogonal views"
        );
        debug_assert!(slice_counts.iter().all(|count| *count > 0));
        debug_assert!(cine_fps.is_finite() && (1.0..=60.0).contains(&cine_fps));
        debug_assert!(zoom.is_finite() && zoom > 0.0);
        debug_assert!(pan.x().is_finite() && pan.y().is_finite());
        debug_assert!(last_annotation.is_none_or(|summary| summary.primary_value().is_finite()));
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
            view_transform,
            crosshair_visible,
            linked_cursor_voxel,
            window_preset_index,
            active_tool_index,
            active_tool_name,
            annotation_count,
            last_annotation,
        }
    }
}
