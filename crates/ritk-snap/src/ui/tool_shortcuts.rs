//! Tool keyboard shortcut SSOT.
//!
//! Defines single-key access to viewer interaction tools, allowing users to
//! activate measurement, segmentation, and navigation tools without clicking
//! the toolbar.
//!
//! # Shortcut design
//!
//! Shortcuts follow ITK-SNAP conventions and common imaging software patterns:
//! - **L**   â†’ Measure Length (distance between two points)
//! - **A**   â†’ Measure Angle (angle between three points)
//! - **R**   â†’ ROI Rectangle (rectangular region of interest)
//! - **E**   â†’ ROI Ellipse (elliptical region of interest)
//! - **H**   â†’ HU Point (single-point intensity readout)
//! - **P**   â†’ Pan (drag to scroll viewport without changing zoom/W&L)
//! - **Z**   â†’ Zoom (scroll wheel or drag to zoom)
//! - **W**   â†’ Window/Level (adjust display intensity mapping)
//! - **B**   â†’ LabelPaint (paint segmentation labels)
//! - **E**   â†’ LabelErase (erase segmentation labels; conflicts with ROI Ellipse â€” E prioritizes paint, use Shift+E for erase)
//!
//! Tools not directly selectable via single key (Crosshair) remain toolbar-only or
//! are context-active (e.g., Crosshair is always active for viewport clicks).
//!
//! # Implementation
//!
//! [`tool_kind_for_key`] is the SSOT that maps egui::Key to optional ToolKind.
//! Return value is `Some(ToolKind)` if the key corresponds to a shortcut,
//! or `None` if the key has no tool binding.
//!
//! The caller invokes this function when a key-press event is detected and
//! applies the returned tool via the app-shell's existing `set_active_tool` path.

use crate::tools::kind::ToolKind;
use egui::Key;

// â”€â”€ Shortcut constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Single-key shortcut for Measure Length tool.
pub const KEY_MEASURE_LENGTH: Key = Key::L;

/// Single-key shortcut for Measure Angle tool.
pub const KEY_MEASURE_ANGLE: Key = Key::A;

/// Single-key shortcut for ROI Rectangle tool.
pub const KEY_ROI_RECT: Key = Key::R;

/// Single-key shortcut for ROI Ellipse tool.
pub const KEY_ROI_ELLIPSE: Key = Key::E;

/// Single-key shortcut for HU Point tool.
pub const KEY_HU_POINT: Key = Key::H;

/// Single-key shortcut for Pan tool.
pub const KEY_PAN: Key = Key::P;

/// Single-key shortcut for Zoom tool.
pub const KEY_ZOOM: Key = Key::Z;

/// Single-key shortcut for Window/Level tool.
pub const KEY_WINDOW_LEVEL: Key = Key::W;

/// Single-key shortcut for Label Paint tool.
pub const KEY_LABEL_PAINT: Key = Key::B;

// â”€â”€ SSOT function â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Map a key press to an optional tool kind.
///
/// Returns `Some(ToolKind)` if the key corresponds to a tool shortcut,
/// or `None` if the key has no binding.
///
/// # Design
/// This function is the single authoritative mapping from keyboard input
/// to tool activation. All app-shell key-handling code must call this
/// function to determine which tool (if any) should be activated.
///
/// # Convention
/// - **L** â†’ Length measurement
/// - **A** â†’ Angle measurement
/// - **R** â†’ Rectangle ROI
/// - **E** â†’ Ellipse ROI
/// - **H** â†’ HU point readout
/// - **P** â†’ Pan
/// - **Z** â†’ Zoom
/// - **W** â†’ Window/Level
/// - **B** â†’ Label Paint
///
/// All other keys return `None`.
#[inline]
pub fn tool_kind_for_key(key: Key) -> Option<ToolKind> {
    match key {
        KEY_MEASURE_LENGTH => Some(ToolKind::MeasureLength),
        KEY_MEASURE_ANGLE => Some(ToolKind::MeasureAngle),
        KEY_ROI_RECT => Some(ToolKind::RoiRect),
        KEY_ROI_ELLIPSE => Some(ToolKind::RoiEllipse),
        KEY_HU_POINT => Some(ToolKind::PointHu),
        KEY_PAN => Some(ToolKind::Pan),
        KEY_ZOOM => Some(ToolKind::Zoom),
        KEY_WINDOW_LEVEL => Some(ToolKind::WindowLevel),
        KEY_LABEL_PAINT => Some(ToolKind::LabelPaint),
        _ => None }
}

#[cfg(test)]
#[path = "tests_tool_shortcuts.rs"]
mod tests;
