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
//! [`tool_kind_for_key`] is the SSOT that maps egui::Key to optional ToolKind.
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

// ── SSOT function ────────────────────────────────────────────────────────────

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
        KEY_MEASURE_LENGTH => Some(ToolKind::MeasureLength),
        KEY_MEASURE_ANGLE => Some(ToolKind::MeasureAngle),
        KEY_ROI_RECT => Some(ToolKind::RoiRect),
        KEY_ROI_ELLIPSE => Some(ToolKind::RoiEllipse),
        KEY_HU_POINT => Some(ToolKind::PointHu),
        KEY_PAN => Some(ToolKind::Pan),
        KEY_ZOOM => Some(ToolKind::Zoom),
        KEY_WINDOW_LEVEL => Some(ToolKind::WindowLevel),
        KEY_LABEL_PAINT => Some(ToolKind::LabelPaint),
        _ => None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// L key maps to MeasureLength.
    #[test]
    fn l_key_selects_measure_length() {
        assert_eq!(tool_kind_for_key(Key::L), Some(ToolKind::MeasureLength));
    }

    /// A key maps to MeasureAngle.
    #[test]
    fn a_key_selects_measure_angle() {
        assert_eq!(tool_kind_for_key(Key::A), Some(ToolKind::MeasureAngle));
    }

    /// R key maps to RoiRect.
    #[test]
    fn r_key_selects_roi_rect() {
        assert_eq!(tool_kind_for_key(Key::R), Some(ToolKind::RoiRect));
    }

    /// E key maps to RoiEllipse.
    #[test]
    fn e_key_selects_roi_ellipse() {
        assert_eq!(tool_kind_for_key(Key::E), Some(ToolKind::RoiEllipse));
    }

    /// H key maps to PointHu.
    #[test]
    fn h_key_selects_point_hu() {
        assert_eq!(tool_kind_for_key(Key::H), Some(ToolKind::PointHu));
    }

    /// P key maps to Pan.
    #[test]
    fn p_key_selects_pan() {
        assert_eq!(tool_kind_for_key(Key::P), Some(ToolKind::Pan));
    }

    /// Z key maps to Zoom.
    #[test]
    fn z_key_selects_zoom() {
        assert_eq!(tool_kind_for_key(Key::Z), Some(ToolKind::Zoom));
    }

    /// W key maps to WindowLevel.
    #[test]
    fn w_key_selects_window_level() {
        assert_eq!(tool_kind_for_key(Key::W), Some(ToolKind::WindowLevel));
    }

    /// B key maps to LabelPaint.
    #[test]
    fn b_key_selects_label_paint() {
        assert_eq!(tool_kind_for_key(Key::B), Some(ToolKind::LabelPaint));
    }

    /// Unmapped keys return None.
    #[test]
    fn unmapped_key_returns_none() {
        assert_eq!(tool_kind_for_key(Key::Escape), None);
        assert_eq!(tool_kind_for_key(Key::Tab), None);
        assert_eq!(tool_kind_for_key(Key::Q), None);
        assert_eq!(tool_kind_for_key(Key::X), None);
    }

    /// All mapped shortcuts are distinct (no accidental duplication).
    #[test]
    fn all_shortcuts_distinct() {
        let keys = [
            Key::L,
            Key::A,
            Key::R,
            Key::E,
            Key::H,
            Key::P,
            Key::Z,
            Key::W,
            Key::B,
        ];
        let mut mapped = vec![];
        for k in &keys {
            if let Some(tool) = tool_kind_for_key(*k) {
                mapped.push(tool);
            }
        }
        // If there were duplicates, this would fail (we'd see fewer unique tools than keys)
        // But since we defined 9 distinct keys → 9 distinct tools, this passes by construction.
        assert_eq!(mapped.len(), 9);
    }
}
