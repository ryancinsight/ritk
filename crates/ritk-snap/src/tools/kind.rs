//! Viewer interaction tool identifiers.
//!
//! Each [`ToolKind`] variant maps to a distinct cursor-interaction mode. The
//! enum is used as a discriminant in the toolbar, in tool-state transitions,
//! and in serialised session snapshots.
//!
//! # Invariants
//! - [`ToolKind::all()`] returns every variant exactly once, in toolbar display order.
//! - [`ToolKind::label()`], [`ToolKind::tooltip()`], and [`ToolKind::icon()`]
//!   return non-empty `&'static str` values for every variant.

/// Available viewer interaction tools.
///
/// The active tool determines how pointer events in the viewport are
/// interpreted. Only one tool may be active at a time per viewport.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum ToolKind {
    /// Drag to pan the current viewport.
    Pan,
    /// Scroll to zoom, or drag with the secondary pointer button.
    Zoom,
    /// Click-drag to adjust window/level
    /// (horizontal drag â†’ window width, vertical drag â†’ window centre).
    WindowLevel,
    /// Click two points to measure distance in mm.
    MeasureLength,
    /// Click three points to measure angle in degrees.
    MeasureAngle,
    /// Draw a rectangle ROI and compute mean/std/min/max HU and area.
    RoiRect,
    /// Draw an ellipse ROI and compute mean/std/min/max HU and area.
    RoiEllipse,
    /// Click to place a crosshair that links all MPR views.
    Crosshair,
    /// Click to display the HU value at the cursor position.
    PointHu,
    /// Paint the active segmentation label onto voxels under the cursor.
    LabelPaint,
    /// Erase segmentation labels to background under the cursor.
    LabelErase }

impl ToolKind {
    /// Short human-readable label used in the toolbar button and menu entries.
    pub fn label(&self) -> &'static str {
        match self {
            ToolKind::Pan => "Pan",
            ToolKind::Zoom => "Zoom",
            ToolKind::WindowLevel => "W/L",
            ToolKind::MeasureLength => "Length",
            ToolKind::MeasureAngle => "Angle",
            ToolKind::RoiRect => "ROI Rect",
            ToolKind::RoiEllipse => "ROI Ellipse",
            ToolKind::Crosshair => "Crosshair",
            ToolKind::PointHu => "HU Point",
            ToolKind::LabelPaint => "Label Paint",
            ToolKind::LabelErase => "Label Erase" }
    }

    /// Longer description shown in the toolbar tooltip on hover.
    pub fn tooltip(&self) -> &'static str {
        match self {
            ToolKind::Pan => "Pan: drag to scroll the viewport without changing zoom or W/L.",
            ToolKind::Zoom => {
                "Zoom: scroll wheel to zoom in/out; drag vertically for continuous zoom."
            }
            ToolKind::WindowLevel => {
                "Window/Level: drag horizontally to change window width, \
                 vertically to change window centre."
            }
            ToolKind::MeasureLength => {
                "Measure Length: click two points to display the straight-line \
                 distance in millimetres."
            }
            ToolKind::MeasureAngle => {
                "Measure Angle: click three points â€” the second point is the vertex â€” \
                 to display the included angle in degrees."
            }
            ToolKind::RoiRect => {
                "ROI Rectangle: drag a rectangle to compute mean, std, min, max \
                 HU and area (mmÂ²) within the selection."
            }
            ToolKind::RoiEllipse => {
                "ROI Ellipse: drag an ellipse to compute mean, std, min, max \
                 HU and area (mmÂ²) within the selection."
            }
            ToolKind::Crosshair => {
                "Crosshair: click to set a reference point that is synchronised \
                 across all three MPR views."
            }
            ToolKind::PointHu => {
                "HU Point: click to display the Hounsfield Unit value at the \
                 cursor position."
            }
            ToolKind::LabelPaint => {
                "Label Paint: click or drag to paint the active segmentation \
                 label with the current brush radius."
            }
            ToolKind::LabelErase => {
                "Label Erase: click or drag to erase segmentation labels \
                 back to background with the current brush radius."
            }
        }
    }

    /// Unicode icon or emoji representing the tool in the toolbar.
    pub fn icon(&self) -> &'static str {
        match self {
            ToolKind::Pan => "âœ‹",
            ToolKind::Zoom => "ðŸ”",
            ToolKind::WindowLevel => "â˜€",
            ToolKind::MeasureLength => "ðŸ“",
            ToolKind::MeasureAngle => "ðŸ“",
            ToolKind::RoiRect => "â–­",
            ToolKind::RoiEllipse => "â¬­",
            ToolKind::Crosshair => "âŠ•",
            ToolKind::PointHu => "âŠ™",
            ToolKind::LabelPaint => "P",
            ToolKind::LabelErase => "E" }
    }

    /// All variants in toolbar display order.
    ///
    /// Every variant appears exactly once. This slice is the single source of
    /// truth for iteration â€” toolbar rendering, serialisation round-trips, and
    /// tests all consume this slice.
    pub fn all() -> &'static [ToolKind] {
        &[
            ToolKind::Pan,
            ToolKind::Zoom,
            ToolKind::WindowLevel,
            ToolKind::MeasureLength,
            ToolKind::MeasureAngle,
            ToolKind::RoiRect,
            ToolKind::RoiEllipse,
            ToolKind::Crosshair,
            ToolKind::PointHu,
            ToolKind::LabelPaint,
            ToolKind::LabelErase,
        ]
    }
}

#[cfg(test)]
#[path = "tests_kind.rs"]
mod tests;
