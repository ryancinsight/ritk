//! In-progress tool interaction state and ROI shape discriminant.

use super::super::kind::ToolKind;
use egui::Pos2;

// â”€â”€ In-progress tool state â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// In-progress interaction state for the active tool.
///
/// Only one `ToolState` variant is active per viewport at a time. `Idle` is
/// the resting state; every other variant represents a partially completed
/// gesture that will either be confirmed (producing an `Annotation`) or
/// cancelled on the next interaction event.
#[derive(Debug, Clone)]
pub enum ToolState {
    /// No gesture in progress.
    Idle,

    /// Pan drag in progress: stores the pointer start position in screen space
    /// and the viewport origin at the time the drag began.
    Panning {
        /// Pointer position (screen pixels) where the drag started.
        start: Pos2,
        /// Viewport pan offset at the moment the drag started.
        viewport_origin: Pos2 },

    /// Zoom drag in progress.
    Zooming {
        /// Pointer position (screen pixels) where the drag started.
        start: Pos2,
        /// Zoom multiplier at the moment the drag started.
        original_zoom: f32 },

    /// Window/Level drag in progress.
    WindowLevelDrag {
        /// Pointer position (screen pixels) where the drag started.
        start: Pos2,
        /// Window centre at the moment the drag started.
        original_center: f64,
        /// Window width at the moment the drag started.
        original_width: f64 },

    /// First point of a two-click length measurement has been placed.
    MeasureLength1 {
        /// First measurement point in image pixel coordinates `[row, col]`.
        p1: Pos2 },

    /// First two points of a three-click angle measurement have been placed.
    MeasureAngle2 {
        /// First point in image pixel coordinates `[row, col]`.
        p1: Pos2,
        /// Second point (vertex) in image pixel coordinates `[row, col]`.
        p2: Pos2 },

    /// ROI drag in progress.
    RoiDrag {
        /// Drag start in image pixel coordinates `[row, col]`.
        start: Pos2,
        /// Current pointer position in image pixel coordinates `[row, col]`.
        current: Pos2,
        /// Whether the ROI is rectangular or elliptical.
        kind: RoiKind } }

/// Discriminant for the two supported ROI shapes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoiKind {
    Rect,
    Ellipse }

impl ToolState {
    /// Returns `true` when no gesture is in progress.
    pub fn is_idle(&self) -> bool {
        matches!(self, ToolState::Idle)
    }

    /// Returns the [`ToolKind`] that owns this state, or `None` when idle.
    pub fn tool_kind(&self) -> Option<ToolKind> {
        match self {
            ToolState::Idle => None,
            ToolState::Panning { .. } => Some(ToolKind::Pan),
            ToolState::Zooming { .. } => Some(ToolKind::Zoom),
            ToolState::WindowLevelDrag { .. } => Some(ToolKind::WindowLevel),
            ToolState::MeasureLength1 { .. } => Some(ToolKind::MeasureLength),
            ToolState::MeasureAngle2 { .. } => Some(ToolKind::MeasureAngle),
            ToolState::RoiDrag {
                kind: RoiKind::Rect,
                ..
            } => Some(ToolKind::RoiRect),
            ToolState::RoiDrag {
                kind: RoiKind::Ellipse,
                ..
            } => Some(ToolKind::RoiEllipse) }
    }
}
