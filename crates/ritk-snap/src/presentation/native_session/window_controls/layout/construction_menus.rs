//! Menu rows, counts and tab geometry for the native window chrome.
//!
//! The menu tables live beside the panel constructor without growing it past
//! the file budget; `construction` reimports what `ChromeLayout::new` needs.

use super::super::super::layout::WorkspaceLayout;
use super::super::{Menu, WindowAction};
use crate::app::SnapApp;
#[cfg(windows)]
use crate::presentation::native_session::deidentify::{AnonymizationScope, Deidentification};
use crate::presentation::PointerButton;
use crate::tools::kind::ToolKind;
use anyhow::{anyhow, Result};

/// Reassignable pointer buttons in Tools-menu row order with static labels.
///
/// Selecting a row binds the currently selected tool to that button, so the
/// labels stay static while the action carries the tool chosen at click time.
pub(super) const POINTER_BUTTON_ROWS: [(&str, PointerButton); 4] = [
    ("Bind middle button", PointerButton::Middle),
    ("Bind right button", PointerButton::Right),
    ("Bind back button", PointerButton::X1),
    ("Bind forward button", PointerButton::X2),
];

const FILE_ITEMS: &[(&str, WindowAction)] = &[
    ("Open Study...", WindowAction::OpenStudy),
    #[cfg(windows)]
    ("Repair Study Copy...", WindowAction::RepairStudy),
    ("Exit", WindowAction::Exit),
];

const DISPLAY_CONTROLS: &[(&str, WindowAction)] = &[
    ("Flip horizontal  H", WindowAction::FlipHorizontal),
    ("Flip vertical  V", WindowAction::FlipVertical),
    ("Rotate 90 CW  R", WindowAction::RotateCw),
    ("Rotate 90 CCW  Shift+R", WindowAction::RotateCcw),
    ("Rotate 180", WindowAction::RotateHalfTurn),
    ("Reset orientation  Shift+O", WindowAction::ResetOrientation),
    ("Zoom to fit  Ctrl+0", WindowAction::ZoomFit),
    ("Zoom 1:1", WindowAction::ZoomOneToOne),
    (
        "Save W/L as custom...",
        WindowAction::SaveCustomPreset { index: None },
    ),
];

pub(super) const MENU_TABS: &[(Menu, &str, u32)] = &[
    (Menu::File, "File", 54),
    (Menu::View, "View", 58),
    (Menu::Tools, "Tools", 66),
    (Menu::Window, "Window", 78),
    #[cfg(windows)]
    (Menu::Anonymize, "Anonymize", 102),
    (Menu::Display, "Display", 76),
];

#[cfg(windows)]
pub(super) const ANONYMIZE_ITEMS: [(&str, AnonymizationScope, Deidentification); 4] = {
    use AnonymizationScope::{Series, Study};
    use Deidentification::{Basic, Strict};
    [
        ("Active study (Basic)...", Study, Basic),
        ("Active study (Strict)...", Study, Strict),
        ("Active series (Basic)...", Series, Basic),
        ("Active series (Strict)...", Series, Strict),
    ]
};

pub(super) fn menu_tab_geometry(menu: Menu) -> u32 {
    let mut x = 8_u32;
    for (candidate, _, width) in MENU_TABS.iter().copied() {
        if candidate == menu {
            return x;
        }
        x = x.saturating_add(width);
    }
    0
}

pub(super) fn menu_item_count(
    menu: Menu,
    app: &SnapApp,
    workspace_layout: WorkspaceLayout,
) -> usize {
    match menu {
        Menu::File => FILE_ITEMS.len(),
        #[cfg(windows)]
        Menu::Anonymize => ANONYMIZE_ITEMS.len(),
        Menu::View => {
            let base = if workspace_layout.is_grid() { 4 } else { 6 };
            base + 2 * usize::from(has_enhanced_pet_volumes(app))
        }
        Menu::Tools => {
            ToolKind::all()
                .iter()
                .filter(|tool| !(workspace_layout.is_grid() && **tool == ToolKind::Crosshair))
                .count()
                + POINTER_BUTTON_ROWS.len()
                + usize::from(app.has_length_calibration() || app.can_calibrate_length())
        }
        Menu::Window => 6,
        Menu::Display => app
            .browser_window_presets()
            .len()
            .saturating_add(1)
            .saturating_add(crate::render::NamedColorMap::ALL.len())
            .saturating_add(DISPLAY_CONTROLS.len())
            .saturating_add(crate::app::state::MAX_CUSTOM_PRESETS),
        Menu::GridPicker => 0,
    }
}

fn has_enhanced_pet_volumes(app: &SnapApp) -> bool {
    app.enhanced_pet_selection()
        .is_some_and(|selection| selection.volumes().len() > 1)
}

fn pet_view_item(index: usize, app: &SnapApp) -> Result<(&'static str, WindowAction, bool)> {
    match (has_enhanced_pet_volumes(app), index) {
        (true, 0) => Ok((
            "Previous PET volume",
            WindowAction::StepEnhancedPetVolume(-1),
            false,
        )),
        (true, 1) => Ok((
            "Next PET volume",
            WindowAction::StepEnhancedPetVolume(1),
            false,
        )),
        _ => Err(anyhow!("native View menu row is outside its items")),
    }
}

pub(super) fn menu_item(
    menu: Menu,
    index: usize,
    app: &SnapApp,
    workspace_layout: WorkspaceLayout,
) -> Result<(&'static str, WindowAction, bool)> {
    match menu {
        Menu::File => FILE_ITEMS
            .get(index)
            .map(|&(label, action)| (label, action, false))
            .ok_or_else(|| anyhow!("native File menu row is outside its items")),
        #[cfg(windows)]
        Menu::Anonymize => {
            let &(label, scope, level) = ANONYMIZE_ITEMS
                .get(index)
                .ok_or_else(|| anyhow!("native Anonymize menu row is outside its items"))?;
            Ok((label, WindowAction::Anonymize(scope, level), false))
        }
        Menu::View if workspace_layout.is_grid() => match index {
            0 => Ok((
                "Toggle series browser",
                WindowAction::ToggleSeriesPreview,
                false,
            )),
            1 => Ok((
                "Toggle cine playback",
                WindowAction::ToggleCine,
                app.cine.enabled,
            )),
            2 => Ok(("Reset view", WindowAction::ResetView, false)),
            3 => Ok((
                "Toggle position sync  F5",
                WindowAction::TogglePositionSync,
                false,
            )),
            _ => pet_view_item(index - 4, app),
        },
        Menu::View => match index {
            0 => Ok((
                "Toggle series browser",
                WindowAction::ToggleSeriesPreview,
                false,
            )),
            1 => Ok((
                "Toggle linked crosshair",
                WindowAction::ToggleCrosshair,
                app.show_crosshair,
            )),
            2 => Ok((
                "Toggle clinical overlay",
                WindowAction::ToggleOverlay,
                app.show_overlay,
            )),
            // Chrome-level state is invisible to the menu layout, so the row
            // never shows checked; see ToggleSeriesPreview above.
            3 => Ok((
                "Toggle anonymized overlay",
                WindowAction::ToggleAnonymizedOverlay,
                false,
            )),
            4 => Ok((
                "Toggle cine playback",
                WindowAction::ToggleCine,
                app.cine.enabled,
            )),
            5 => Ok(("Reset view", WindowAction::ResetView, false)),
            _ => pet_view_item(index - 6, app),
        },
        Menu::Tools => {
            let tool_rows = ToolKind::all()
                .iter()
                .copied()
                .filter(|tool| !(workspace_layout.is_grid() && *tool == ToolKind::Crosshair))
                .count();
            if index < tool_rows {
                let tool = ToolKind::all()
                    .iter()
                    .copied()
                    .filter(|tool| !(workspace_layout.is_grid() && *tool == ToolKind::Crosshair))
                    .nth(index)
                    .ok_or_else(|| anyhow!("native Tools menu row is outside its items"))?;
                return Ok((
                    tool.label(),
                    WindowAction::SelectTool(tool),
                    app.active_tool == tool,
                ));
            }
            let calibrated = app.has_length_calibration();
            let calibration_row = calibrated || app.can_calibrate_length();
            if calibration_row && index == tool_rows {
                let label = if calibrated {
                    "Reset length calibration"
                } else {
                    "Calibrate last distance..."
                };
                return Ok((label, WindowAction::LengthCalibration, false));
            }
            let (label, button) = POINTER_BUTTON_ROWS
                .get(index - tool_rows - usize::from(calibration_row))
                .ok_or_else(|| anyhow!("native Tools menu row is outside its items"))?;
            Ok((
                label,
                WindowAction::AssignPointerTool {
                    button: *button,
                    tool: app.active_tool,
                },
                app.pointer.bindings.tool_for(*button, app.active_tool) == app.active_tool,
            ))
        }
        Menu::Window => match index {
            0 => Ok((
                "Three-plane MPR",
                WindowAction::SetLayout(WorkspaceLayout::Orthogonal),
                !workspace_layout.is_grid(),
            )),
            1 => Ok((
                "Panel layout...",
                WindowAction::OpenMenu(Menu::GridPicker),
                workspace_layout.is_grid(),
            )),
            2 => Ok((
                "Maximize / restore panel  Ctrl+M",
                WindowAction::ToggleActivePanel,
                false,
            )),
            3 => Ok((
                "Close active panel  Ctrl+F4",
                WindowAction::CloseActivePanel,
                false,
            )),
            4 => Ok((
                "Close all panels  Shift+F4",
                WindowAction::CloseAllPanels,
                false,
            )),
            5 => Ok((
                "Open multiple series...  F4",
                WindowAction::OpenSeriesPicker,
                false,
            )),
            _ => Err(anyhow!("native Window menu row is outside its items")),
        },
        Menu::Display => {
            let presets = app.browser_window_presets();
            if let Some(preset) = presets.get(index) {
                return Ok((
                    preset.name,
                    WindowAction::ApplyWindowPreset { index },
                    app.browser_window_preset_index() == Some(index),
                ));
            }
            if index == presets.len() {
                return Ok((
                    "Invert display  I",
                    WindowAction::ToggleInvert,
                    app.invert_display,
                ));
            }
            let colormap_index = index.saturating_sub(presets.len().saturating_add(1));
            if colormap_index < crate::render::NamedColorMap::ALL.len() {
                let map = crate::render::NamedColorMap::ALL[colormap_index];
                return Ok((
                    map.label(),
                    WindowAction::SetColormap(map),
                    app.colormap == map,
                ));
            }
            let control_index = colormap_index - crate::render::NamedColorMap::ALL.len();
            if let Some(&(label, action)) = DISPLAY_CONTROLS.get(control_index) {
                let checked = match action {
                    WindowAction::FlipHorizontal => app.view_transform.flip_h,
                    WindowAction::FlipVertical => app.view_transform.flip_v,
                    _ => false,
                };
                return Ok((label, action, checked));
            }
            let custom = control_index - DISPLAY_CONTROLS.len();
            let label = crate::app::state::CUSTOM_PRESET_LABELS
                .get(custom)
                .ok_or_else(|| anyhow!("native Display menu row is outside its items"))?;
            Ok((
                label,
                WindowAction::ApplyCustomPreset { index: custom },
                app.custom_preset_index() == Some(custom),
            ))
        }
        Menu::GridPicker => Err(anyhow!("panel grid picker has no menu rows")),
    }
}
