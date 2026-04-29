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
    /// (horizontal drag → window width, vertical drag → window centre).
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
}

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
        }
    }

    /// Longer description shown in the toolbar tooltip on hover.
    pub fn tooltip(&self) -> &'static str {
        match self {
            ToolKind::Pan =>
                "Pan: drag to scroll the viewport without changing zoom or W/L.",
            ToolKind::Zoom =>
                "Zoom: scroll wheel to zoom in/out; right-drag for continuous zoom.",
            ToolKind::WindowLevel =>
                "Window/Level: drag horizontally to change window width, \
                 vertically to change window centre.",
            ToolKind::MeasureLength =>
                "Measure Length: click two points to display the straight-line \
                 distance in millimetres.",
            ToolKind::MeasureAngle =>
                "Measure Angle: click three points — the second point is the vertex — \
                 to display the included angle in degrees.",
            ToolKind::RoiRect =>
                "ROI Rectangle: drag a rectangle to compute mean, std, min, max \
                 HU and area (mm²) within the selection.",
            ToolKind::RoiEllipse =>
                "ROI Ellipse: drag an ellipse to compute mean, std, min, max \
                 HU and area (mm²) within the selection.",
            ToolKind::Crosshair =>
                "Crosshair: click to set a reference point that is synchronised \
                 across all three MPR views.",
            ToolKind::PointHu =>
                "HU Point: click to display the Hounsfield Unit value at the \
                 cursor position.",
        }
    }

    /// Unicode icon or emoji representing the tool in the toolbar.
    pub fn icon(&self) -> &'static str {
        match self {
            ToolKind::Pan => "✋",
            ToolKind::Zoom => "🔍",
            ToolKind::WindowLevel => "☀",
            ToolKind::MeasureLength => "📏",
            ToolKind::MeasureAngle => "📐",
            ToolKind::RoiRect => "▭",
            ToolKind::RoiEllipse => "⬭",
            ToolKind::Crosshair => "⊕",
            ToolKind::PointHu => "⊙",
        }
    }

    /// All variants in toolbar display order.
    ///
    /// Every variant appears exactly once. This slice is the single source of
    /// truth for iteration — toolbar rendering, serialisation round-trips, and
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
        ]
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// `all()` must enumerate every variant exactly once with no duplicates.
    ///
    /// Correctness criterion: `all().len() == 9` and pairwise distinctness.
    #[test]
    fn test_tool_kind_all_complete_and_distinct() {
        let all = ToolKind::all();
        assert_eq!(
            all.len(),
            9,
            "ToolKind::all() must contain all 9 variants, found {}",
            all.len()
        );
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(
                    all[i], all[j],
                    "ToolKind::all() contains duplicate at indices {i} and {j}: {:?}",
                    all[i]
                );
            }
        }
    }

    /// `label()` must return a non-empty string for every variant.
    #[test]
    fn test_tool_kind_label_non_empty() {
        for tool in ToolKind::all() {
            let label = tool.label();
            assert!(
                !label.is_empty(),
                "{:?}.label() must not be empty",
                tool
            );
        }
    }

    /// `tooltip()` must return a non-empty string for every variant, and must
    /// be distinct from `label()` (a tooltip adds information beyond the label).
    #[test]
    fn test_tool_kind_tooltip_non_empty_and_distinct_from_label() {
        for tool in ToolKind::all() {
            let tooltip = tool.tooltip();
            assert!(
                !tooltip.is_empty(),
                "{:?}.tooltip() must not be empty",
                tool
            );
            assert_ne!(
                tooltip,
                tool.label(),
                "{:?}.tooltip() must differ from label()",
                tool
            );
        }
    }

    /// `icon()` must return a non-empty string for every variant.
    #[test]
    fn test_tool_kind_icon_non_empty() {
        for tool in ToolKind::all() {
            let icon = tool.icon();
            assert!(
                !icon.is_empty(),
                "{:?}.icon() must not be empty",
                tool
            );
        }
    }

    /// Serde round-trip: every variant must survive JSON serialisation and
    /// deserialisation with value equality.
    #[test]
    fn test_tool_kind_serde_round_trip() {
        for &tool in ToolKind::all() {
            let json = serde_json::to_string(&tool)
                .unwrap_or_else(|e| panic!("{tool:?} serde_json::to_string failed: {e}"));
            let recovered: ToolKind = serde_json::from_str(&json)
                .unwrap_or_else(|e| panic!("{tool:?} serde_json::from_str failed: {e}"));
            assert_eq!(
                tool, recovered,
                "{:?} serde round-trip must preserve value equality",
                tool
            );
        }
    }

    /// `ToolKind::Pan` must be the first element (toolbar primary position).
    #[test]
    fn test_tool_kind_pan_is_first() {
        assert_eq!(
            ToolKind::all()[0],
            ToolKind::Pan,
            "Pan must be the first tool in ToolKind::all()"
        );
    }
}
