//! Tool keyboard shortcut SSOT.
//!
//! Defines single-key access to viewer interaction tools, allowing users to
//! activate measurement, segmentation, and navigation tools from the keyboard.
//!
//! # Shortcut design
//!
//! Shortcuts follow ITK-SNAP conventions and common imaging software patterns:
//! - **L**   → Measure Length (distance between two points)
//! - **A**   → Measure Angle (angle between three points)
//! - **R**   → ROI Rectangle (rectangular region of interest)
//! - **E**   → ROI Ellipse (elliptical region of interest)
//! - **H**   → HU Point (single-point intensity readout)
//! - **P**   → Pan (drag to scroll viewport without changing zoom/W&L)
//! - **Z**   → Zoom (drag to zoom; Ctrl or Cmd with the wheel zooms under any tool)
//! - **W**   → Window/Level (adjust display intensity mapping)
//! - **B**   → LabelPaint (paint segmentation labels)
//!
//! No key selects LabelErase or Crosshair. Crosshair is active for viewport clicks.
//!
//! # Implementation
//!
//! `tool_kind_for_virtual_key` is the SSOT that maps host keyboard values to
//! optional [`ToolKind`] values.
//! Return value is `Some(ToolKind)` if the key corresponds to a shortcut,
//! or `None` if the key has no tool binding.
//!
//! The caller invokes this function when a key-press event is detected and
//! applies the returned tool through `select_tool`.

use crate::tools::kind::ToolKind;

// ── Shortcut constants ────────────────────────────────────────────────────────

/// Host virtual-key value for the Measure Length shortcut.
pub const VIRTUAL_KEY_MEASURE_LENGTH: u32 = 0x4c;

/// Host virtual-key value for the Measure Angle shortcut.
pub const VIRTUAL_KEY_MEASURE_ANGLE: u32 = 0x41;

/// Host virtual-key value for the ROI Rectangle shortcut.
pub const VIRTUAL_KEY_ROI_RECT: u32 = 0x52;

/// Host virtual-key value for the ROI Ellipse shortcut.
pub const VIRTUAL_KEY_ROI_ELLIPSE: u32 = 0x45;

/// Host virtual-key value for the HU Point shortcut.
pub const VIRTUAL_KEY_HU_POINT: u32 = 0x48;

/// Host virtual-key value for the Pan shortcut.
pub const VIRTUAL_KEY_PAN: u32 = 0x50;

/// Host virtual-key value for the Zoom shortcut.
pub const VIRTUAL_KEY_ZOOM: u32 = 0x5a;

/// Host virtual-key value for the Window/Level shortcut.
pub const VIRTUAL_KEY_WINDOW_LEVEL: u32 = 0x57;

/// Host virtual-key value for the Label Paint shortcut.
pub const VIRTUAL_KEY_LABEL_PAINT: u32 = 0x42;

// ── SSOT function ────────────────────────────────────────────────────────────

/// Map a host virtual-key value to an optional tool kind.
///
/// The values are the stable ASCII virtual-key values used by the native
/// presentation contract. Browser and native hosts can therefore select the
/// same RITK tool without depending on a shell key type.
///
/// Returns `Some(ToolKind)` if the value corresponds to a tool shortcut, or
/// `None` if it has no binding.
#[inline]
pub fn tool_kind_for_virtual_key(virtual_key: u32) -> Option<ToolKind> {
    match virtual_key {
        VIRTUAL_KEY_MEASURE_LENGTH => Some(ToolKind::MeasureLength),
        VIRTUAL_KEY_MEASURE_ANGLE => Some(ToolKind::MeasureAngle),
        VIRTUAL_KEY_ROI_RECT => Some(ToolKind::RoiRect),
        VIRTUAL_KEY_ROI_ELLIPSE => Some(ToolKind::RoiEllipse),
        VIRTUAL_KEY_HU_POINT => Some(ToolKind::PointHu),
        VIRTUAL_KEY_PAN => Some(ToolKind::Pan),
        VIRTUAL_KEY_ZOOM => Some(ToolKind::Zoom),
        VIRTUAL_KEY_WINDOW_LEVEL => Some(ToolKind::WindowLevel),
        VIRTUAL_KEY_LABEL_PAINT => Some(ToolKind::LabelPaint),
        _ => None,
    }
}

#[cfg(test)]
#[path = "tests_tool_shortcuts.rs"]
mod tests;
