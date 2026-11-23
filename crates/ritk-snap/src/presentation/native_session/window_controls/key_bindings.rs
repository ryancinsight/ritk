//! Global native-window keys: series picker, panel switching, display
//! presets, negative display and colormap cycling.
//!
//! The match lives beside the pointer layer that calls it without growing
//! `events` past the file budget; series-navigation keys stay in `events`
//! next to the browser state they read.

use super::WindowAction;
use crate::app::action_adapter::{
    VIRTUAL_KEY_COLORMAP_CYCLE, VIRTUAL_KEY_FLIP_HORIZONTAL, VIRTUAL_KEY_FLIP_VERTICAL,
    VIRTUAL_KEY_INVERT_DISPLAY, VIRTUAL_KEY_ORIENTATION_RESET, VIRTUAL_KEY_PRESET_KEYS,
    VIRTUAL_KEY_ROTATE, VIRTUAL_KEY_ZOOM_FIT,
};
use crate::app::SnapApp;
use crate::tools::kind::ToolKind;

/// Route one non-repeated key press to its window action, if any.
///
/// Digit keys select the loaded modality's preset table entries; `I`
/// toggles negative display; `C` selects Cobb and `Shift+C` advances the
/// intensity colormap.
/// `Alt` plus a digit applies a numbered custom slot, while `Shift` plus a
/// digit saves the current window into that custom slot.
pub(super) fn global_key_action(
    virtual_key: u32,
    ctrl: bool,
    shift: bool,
    alt: bool,
    app: &SnapApp,
) -> Option<WindowAction> {
    let custom_slot = || {
        VIRTUAL_KEY_PRESET_KEYS
            .iter()
            .position(|preset_key| *preset_key == virtual_key)
            .filter(|index| *index < crate::app::state::MAX_CUSTOM_PRESETS)
    };
    if alt
        && !ctrl
        && !shift
        && let Some(index) = custom_slot()
    {
        return Some(WindowAction::ApplyCustomPreset { index });
    }
    if shift
        && !ctrl
        && !alt
        && let Some(index) = custom_slot()
    {
        return Some(WindowAction::SaveCustomPreset { index: Some(index) });
    }
    match (virtual_key, ctrl, shift) {
        (0x4d, false, false) => Some(WindowAction::SelectTool(ToolKind::Pan)),
        (key, false, false) if key == crate::ui::tool_shortcuts::VIRTUAL_KEY_MEASURE_COBB => {
            Some(WindowAction::SelectTool(ToolKind::MeasureCobb))
        }
        (key, false, true) if key == VIRTUAL_KEY_COLORMAP_CYCLE => {
            let maps = crate::render::NamedColorMap::ALL;
            let position = maps
                .iter()
                .position(|map| *map == app.colormap)
                .unwrap_or(0);
            let next = maps
                .get(position.saturating_add(1).rem_euclid(maps.len()))
                .copied()
                .unwrap_or(maps[0]);
            Some(WindowAction::SetColormap(next))
        }
        (key, false, false) if key == crate::ui::tool_shortcuts::VIRTUAL_KEY_ARROW => {
            Some(WindowAction::SelectTool(ToolKind::Arrow))
        }
        (key, false, false) if key == crate::ui::tool_shortcuts::VIRTUAL_KEY_PENCIL => {
            Some(WindowAction::SelectTool(ToolKind::Pencil))
        }
        (0x47, false, false) => Some(WindowAction::SelectTool(ToolKind::RoiClosedPolygon)),
        (0x7b, true, false) if !alt => Some(WindowAction::ToggleReferenceLines),
        (0x74, false, true) if !alt => Some(WindowAction::AlignCurrentSlices),
        (0x74, false, false) if !alt => Some(WindowAction::TogglePositionSync),
        (0x73, false, false) => Some(WindowAction::OpenSeriesPicker),
        (0x73, true, false) => Some(WindowAction::CloseActivePanel),
        (0x73, false, true) => Some(WindowAction::CloseAllPanels),
        (0x4d, true, false) => Some(WindowAction::ToggleActivePanel),
        (0x09, _, true) => Some(WindowAction::ActivatePreviousPanel),
        (0x09, _, false) => Some(WindowAction::ActivateNextPanel),
        (key, true, false) if key == VIRTUAL_KEY_ZOOM_FIT => Some(WindowAction::ZoomFit),
        (key, false, false) if key == VIRTUAL_KEY_FLIP_HORIZONTAL => {
            Some(WindowAction::FlipHorizontal)
        }
        (key, false, false) if key == VIRTUAL_KEY_FLIP_VERTICAL => Some(WindowAction::FlipVertical),
        (key, false, false) if key == VIRTUAL_KEY_ROTATE => Some(WindowAction::RotateCw),
        (key, false, true) if key == VIRTUAL_KEY_ROTATE => Some(WindowAction::RotateCcw),
        (key, false, true) if key == VIRTUAL_KEY_ORIENTATION_RESET => {
            Some(WindowAction::ResetOrientation)
        }
        (key, false, false) => {
            if key == VIRTUAL_KEY_INVERT_DISPLAY {
                Some(WindowAction::ToggleInvert)
            } else {
                VIRTUAL_KEY_PRESET_KEYS
                    .iter()
                    .position(|preset_key| *preset_key == key)
                    .filter(|index| *index < app.browser_window_presets().len())
                    .map(|index| WindowAction::ApplyWindowPreset { index })
            }
        }
        _ => None,
    }
}

pub(super) fn global_key_event(
    menu: Option<super::Menu>,
    action: WindowAction,
) -> super::WindowChromeEvent {
    if menu.is_some()
        && matches!(
            action,
            WindowAction::SelectTool(ToolKind::Arrow | ToolKind::Pencil | ToolKind::MeasureCobb)
        )
    {
        return super::WindowChromeEvent::consumed(false);
    }
    super::WindowChromeEvent {
        consumed: true,
        repaint: true,
        action: Some(action),
    }
}
