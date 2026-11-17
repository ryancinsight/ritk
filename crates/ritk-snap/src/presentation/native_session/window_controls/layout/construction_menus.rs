//! Menu rows, counts and tab geometry for the native window chrome.
//!
//! The menu tables live beside the panel constructor without growing it past
//! the file budget; `construction` reimports what `ChromeLayout::new` needs.

use super::super::super::layout::WorkspaceLayout;
use super::super::{Menu, WindowAction};
use crate::app::SnapApp;
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

pub(super) const MENU_TABS: [(Menu, &str, u32); 6] = [
    (Menu::File, "File", 54),
    (Menu::View, "View", 58),
    (Menu::Tools, "Tools", 66),
    (Menu::Window, "Window", 78),
    (Menu::Display, "Display", 76),
    (Menu::Anonymize, "Anonymize", 102),
];

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
    for (candidate, _, width) in MENU_TABS {
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
        Menu::File => 3,
        Menu::Anonymize => ANONYMIZE_ITEMS.len(),
        Menu::View => {
            if workspace_layout.is_grid() {
                3
            } else {
                6
            }
        }
        Menu::Tools => {
            ToolKind::all()
                .iter()
                .filter(|tool| !(workspace_layout.is_grid() && **tool == ToolKind::Crosshair))
                .count()
                + POINTER_BUTTON_ROWS.len()
        }
        Menu::Window => 6,
        Menu::Display => app.browser_window_presets().len(),
        Menu::GridPicker => 0,
    }
}

pub(super) fn menu_item(
    menu: Menu,
    index: usize,
    app: &SnapApp,
    workspace_layout: WorkspaceLayout,
) -> Result<(&'static str, WindowAction, bool)> {
    match menu {
        Menu::File => match index {
            0 => Ok(("Open Study...", WindowAction::OpenStudy, false)),
            1 => Ok(("Repair Study Copy...", WindowAction::RepairStudy, false)),
            2 => Ok(("Exit", WindowAction::Exit, false)),
            _ => Err(anyhow!("native File menu row is outside its items")),
        },
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
            _ => Err(anyhow!("native View menu row is outside its items")),
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
            _ => Err(anyhow!("native View menu row is outside its items")),
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
            let (label, button) = POINTER_BUTTON_ROWS
                .get(index - tool_rows)
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
            let preset = presets
                .get(index)
                .ok_or_else(|| anyhow!("native Display menu row is outside its preset table"))?;
            Ok((
                preset.name,
                WindowAction::ApplyWindowPreset { index },
                app.browser_window_preset_index() == Some(index),
            ))
        }
        Menu::GridPicker => Err(anyhow!("panel grid picker has no menu rows")),
    }
}
