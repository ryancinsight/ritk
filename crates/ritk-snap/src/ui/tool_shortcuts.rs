//! Tool keyboard shortcut SSOT.
//!
//! Defines single-key access to viewer interaction tools, allowing users to
//! activate measurement, segmentation, and navigation tools without clicking
//! the toolbar.
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
//! - **Z**   → Zoom (scroll wheel or drag to zoom)
//! - **W**   → Window/Level (adjust display intensity mapping)
//! - **B**   → LabelPaint (paint segmentation labels)
//! - **E**   → LabelErase (erase segmentation labels; conflicts with ROI Ellipse — E prioritizes paint, use Shift+E for erase)
//!
//! Tools not directly selectable via single key (Crosshair) remain toolbar-only or
//! are context-active (e.g., Crosshair is always active for viewport clicks).
//!
//! # Implementation
//!
//! [`tool_kind_for_key`] and [`tool_kind_for_virtual_key`] are the SSOTs that
//! map host keyboard values to optional [`ToolKind`] values.
//! Return value is `Some(ToolKind)` if the key corresponds to a shortcut,
//! or `None` if the key has no tool binding.
//!
//! The caller invokes this function when a key-press event is detected and
//! applies the returned tool via the app-shell's existing `set_active_tool` path.

use crate::tools::kind::ToolKind;
use egui::Key;

// ── Shortcut constants ────────────────────────────────────────────────────────

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
/// same RITK tool without depending on egui's key type.
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

/// Map an egui key press to an optional tool kind.
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
/// - **L** → Length measurement
/// - **A** → Angle measurement
/// - **R** → Rectangle ROI
/// - **E** → Ellipse ROI
/// - **H** → HU point readout
/// - **P** → Pan
/// - **Z** → Zoom
/// - **W** → Window/Level
/// - **B** → Label Paint
///
/// All other keys return `None`.
#[inline]
pub fn tool_kind_for_key(key: Key) -> Option<ToolKind> {
    match key {
        KEY_MEASURE_LENGTH => tool_kind_for_virtual_key(VIRTUAL_KEY_MEASURE_LENGTH),
        KEY_MEASURE_ANGLE => tool_kind_for_virtual_key(VIRTUAL_KEY_MEASURE_ANGLE),
        KEY_ROI_RECT => tool_kind_for_virtual_key(VIRTUAL_KEY_ROI_RECT),
        KEY_ROI_ELLIPSE => tool_kind_for_virtual_key(VIRTUAL_KEY_ROI_ELLIPSE),
        KEY_HU_POINT => tool_kind_for_virtual_key(VIRTUAL_KEY_HU_POINT),
        KEY_PAN => tool_kind_for_virtual_key(VIRTUAL_KEY_PAN),
        KEY_ZOOM => tool_kind_for_virtual_key(VIRTUAL_KEY_ZOOM),
        KEY_WINDOW_LEVEL => tool_kind_for_virtual_key(VIRTUAL_KEY_WINDOW_LEVEL),
        KEY_LABEL_PAINT => tool_kind_for_virtual_key(VIRTUAL_KEY_LABEL_PAINT),
        _ => None,
    }
}

#[cfg(test)]
#[path = "tests_tool_shortcuts.rs"]
mod tests;
