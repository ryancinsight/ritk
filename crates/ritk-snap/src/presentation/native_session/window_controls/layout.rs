//! RadiAnt-style native menus, grouped tools, series preview and status bar.

use super::super::layout::{PanelGrid, WorkspaceLayout, MAX_GRID_COLUMNS, MAX_GRID_ROWS};
#[cfg(test)]
use super::super::layout::ViewportArea;
mod geometry;
mod render;
use super::super::series_browser::SeriesBrowser;
use super::series;
use super::{Menu, WindowAction};
use crate::app::SnapApp;
use crate::tools::kind::ToolKind;
use anyhow::{anyhow, Result};
use arrayvec::ArrayVec;
pub(super) use geometry::ChromeGeometry;
use metis_platform::Rect;

const MENU_ROW_HEIGHT: u32 = 28;
const CONTROL_GAP: u32 = 6;
const CONTROL_CAPACITY: usize = 40;
const GRID_CELL_WIDTH: u32 = 50;
const GRID_CELL_HEIGHT: u32 = 34;
const GRID_CELL_GAP: u32 = 4;
const GRID_POPUP_PADDING: u32 = 8;
const GRID_POPUP_HEADER_HEIGHT: u32 = 38;
const GRID_POPUP_WIDTH: u32 = GRID_POPUP_PADDING * 2
    + MAX_GRID_COLUMNS * GRID_CELL_WIDTH
    + (MAX_GRID_COLUMNS - 1) * GRID_CELL_GAP;
const GRID_POPUP_HEIGHT: u32 = GRID_POPUP_PADDING * 2
    + GRID_POPUP_HEADER_HEIGHT
    + MAX_GRID_ROWS * GRID_CELL_HEIGHT
    + (MAX_GRID_ROWS - 1) * GRID_CELL_GAP;

const MENU_TABS: [(Menu, &str, u32); 4] = [
    (Menu::File, "File", 54),
    (Menu::View, "View", 58),
    (Menu::Tools, "Tools", 66),
    (Menu::Window, "Window", 78),
];

const TOOLBAR_ITEMS: [(&str, u32, WindowAction, &str); 9] = [
    ("Open Study", 98, WindowAction::OpenStudy, "STUDY"),
    (
        "W/L",
        58,
        WindowAction::SelectTool(ToolKind::WindowLevel),
        "NAVIGATE",
    ),
    (
        "Pan",
        58,
        WindowAction::SelectTool(ToolKind::Pan),
        "NAVIGATE",
    ),
    (
        "Zoom",
        64,
        WindowAction::SelectTool(ToolKind::Zoom),
        "NAVIGATE",
    ),
    (
        "Length",
        76,
        WindowAction::SelectTool(ToolKind::MeasureLength),
        "MEASURE",
    ),
    (
        "Angle",
        64,
        WindowAction::SelectTool(ToolKind::MeasureAngle),
        "MEASURE",
    ),
    (
        "Crosshair",
        90,
        WindowAction::SelectTool(ToolKind::Crosshair),
        "DISPLAY",
    ),
    ("Cine", 60, WindowAction::ToggleCine, "DISPLAY"),
    (
        "Split screen",
        116,
        WindowAction::OpenMenu(Menu::GridPicker),
        "LAYOUT",
    ),
];

#[derive(Clone, Copy)]
enum ControlKind {
    MenuTab,
    Toolbar,
    MenuItem,
    Grid,
}

#[derive(Clone, Copy)]
struct ChromeControl {
    rect: Rect,
    label: &'static str,
    action: WindowAction,
    kind: ControlKind,
    active: bool,
}

#[derive(Clone, Copy)]
struct ToolbarGroup {
    x: i32,
    label: &'static str,
}

pub(super) struct ChromeLayout {
    geometry: ChromeGeometry,
    controls: ArrayVec<ChromeControl, CONTROL_CAPACITY>,
    groups: ArrayVec<ToolbarGroup, 8>,
    grid_popup: Option<Rect>,
}

impl ChromeLayout {
    pub(super) fn new(
        width: u32,
        height: u32,
        open_menu: Option<Menu>,
        app: &SnapApp,
        show_series_preview: bool,
        workspace_layout: WorkspaceLayout,
    ) -> Result<Self> {
        let geometry = ChromeGeometry::new(width, height, show_series_preview)?;
        let mut controls = ArrayVec::new();
        let mut groups = ArrayVec::new();
        let mut tab_x = 8_u32;
        for (menu, label, requested_width) in MENU_TABS {
            let tab_width = requested_width.min(width.saturating_sub(tab_x));
            if tab_width > 0 && geometry.menu_height > 0 {
                push_control(
                    &mut controls,
                    ChromeControl {
                        rect: make_rect(tab_x, 0, tab_width, geometry.menu_height)?,
                        label,
                        action: WindowAction::OpenMenu(menu),
                        kind: ControlKind::MenuTab,
                        active: open_menu == Some(menu),
                    },
                )?;
            }
            tab_x = tab_x.saturating_add(requested_width);
        }

        let mut control_x = 12_u32;
        let button_height = geometry.toolbar_height.saturating_sub(20).min(34);
        let button_y = geometry.menu_height.saturating_add(18);
        let mut previous_group = None;
        for (label, requested_width, action, group) in TOOLBAR_ITEMS {
            if previous_group != Some(group) {
                groups
                    .try_push(ToolbarGroup {
                        x: i32::try_from(control_x)
                            .map_err(|_| anyhow!("native toolbar group x exceeds i32"))?,
                        label: group,
                    })
                    .map_err(|_| anyhow!("native toolbar group capacity exceeded"))?;
                previous_group = Some(group);
            }
            let control_width = requested_width.min(width.saturating_sub(control_x));
            if control_width == 0 || button_height == 0 {
                break;
            }
            let active = match action {
                WindowAction::SelectTool(tool) => app.active_tool == tool,
                WindowAction::ToggleCine => app.cine.enabled,
                WindowAction::OpenMenu(Menu::GridPicker) => {
                    workspace_layout.is_grid() || open_menu == Some(Menu::GridPicker)
                }
                _ => false,
            };
            push_control(
                &mut controls,
                ChromeControl {
                    rect: make_rect(control_x, button_y, control_width, button_height)?,
                    label,
                    action,
                    kind: ControlKind::Toolbar,
                    active,
                },
            )?;
            control_x = control_x
                .saturating_add(requested_width)
                .saturating_add(CONTROL_GAP);
        }

        let mut grid_popup = None;
        if open_menu == Some(Menu::GridPicker) {
            if let Some(anchor) = controls
                .iter()
                .find(|control| control.action == WindowAction::OpenMenu(Menu::GridPicker))
                .map(|control| control.rect)
            {
                if let Some(popup) = grid_popup_bounds(width, height, &geometry, anchor)? {
                    grid_popup = Some(popup);
                    let (selected_columns, selected_rows) = workspace_layout
                        .grid()
                        .map_or((0, 0), PanelGrid::dimensions);
                    for row in 1..=MAX_GRID_ROWS {
                        for column in 1..=MAX_GRID_COLUMNS {
                            let grid = PanelGrid::new(column, row).ok_or_else(|| {
                                anyhow!("panel grid picker generated an invalid grid")
                            })?;
                            let label = grid.menu_label().ok_or_else(|| {
                                anyhow!("panel grid picker label is outside its range")
                            })?;
                            let cell_x = GRID_POPUP_PADDING
                                .checked_add(
                                    (column - 1).saturating_mul(GRID_CELL_WIDTH + GRID_CELL_GAP),
                                )
                                .and_then(|offset| offset.checked_add(u32::try_from(popup.x).ok()?))
                                .ok_or_else(|| anyhow!("panel grid picker x overflows"))?;
                            let cell_y = u32::try_from(popup.y)
                                .map_err(|_| anyhow!("panel grid picker y is negative"))?
                                .checked_add(GRID_POPUP_PADDING)
                                .and_then(|value| value.checked_add(GRID_POPUP_HEADER_HEIGHT))
                                .and_then(|value| {
                                    value.checked_add(
                                        (row - 1).saturating_mul(GRID_CELL_HEIGHT + GRID_CELL_GAP),
                                    )
                                })
                                .ok_or_else(|| anyhow!("panel grid picker y overflows"))?;
                            push_control(
                                &mut controls,
                                ChromeControl {
                                    rect: make_rect(
                                        cell_x,
                                        cell_y,
                                        GRID_CELL_WIDTH,
                                        GRID_CELL_HEIGHT,
                                    )?,
                                    label,
                                    action: WindowAction::SetLayout(WorkspaceLayout::Panels(grid)),
                                    kind: ControlKind::Grid,
                                    active: workspace_layout.is_grid()
                                        && column <= selected_columns
                                        && row <= selected_rows,
                                },
                            )?;
                        }
                    }
                }
            }
        } else if let Some(menu) = open_menu {
            let popup_width = 210_u32.min(width);
            let menu_x = menu_tab_geometry(menu).min(width.saturating_sub(popup_width));
            let count = menu_item_count(menu);
            let available_height = height.saturating_sub(geometry.menu_height);
            let count_u32 =
                u32::try_from(count).map_err(|_| anyhow!("native menu item count exceeds u32"))?;
            let row_height = if count == 0 {
                0
            } else {
                MENU_ROW_HEIGHT.min(available_height / count_u32)
            };
            if popup_width > 0 && row_height > 0 {
                for index in 0..count {
                    let index_u32 =
                        u32::try_from(index).map_err(|_| anyhow!("native menu row exceeds u32"))?;
                    let row_y = geometry
                        .menu_height
                        .checked_add(index_u32.saturating_mul(row_height))
                        .ok_or_else(|| anyhow!("native menu row y overflows"))?;
                    let (label, action, active) = menu_item(menu, index, app, workspace_layout)?;
                    push_control(
                        &mut controls,
                        ChromeControl {
                            rect: make_rect(menu_x, row_y, popup_width, row_height)?,
                            label,
                            action,
                            kind: ControlKind::MenuItem,
                            active,
                        },
                    )?;
                }
            }
        }
        Ok(Self {
            geometry,
            controls,
            groups,
            grid_popup,
        })
    }

    pub(super) fn action_at(
        &self,
        x: f64,
        y: f64,
        browser: Option<&SeriesBrowser>,
    ) -> Option<WindowAction> {
        self.controls
            .iter()
            .rev()
            .find(|control| rect_contains(control.rect, x, y))
            .map(|control| control.action)
            .or_else(|| {
                let browser = browser?;
                let index = series::index_at(browser, self.geometry.series_preview, x, y)?;
                Some(WindowAction::SelectSeries(index))
            })
    }

    pub(super) fn series_index_at(
        &self,
        browser: Option<&SeriesBrowser>,
        x: f64,
        y: f64,
    ) -> Option<usize> {
        let browser = browser?;
        series::index_at(browser, self.geometry.series_preview, x, y)
    }

    #[cfg(test)]
    pub(super) const fn viewport_area(&self) -> ViewportArea {
        self.geometry.viewport_area
    }

    pub(super) fn visible_series(&self) -> usize {
        series::visible_count(self.geometry.series_preview)
    }

    pub(super) fn series_contains(&self, x: f64, y: f64) -> bool {
        rect_contains(self.geometry.series_preview, x, y)
    }

    pub(super) fn owns_pointer(&self, x: f64, y: f64) -> bool {
        rect_contains(self.geometry.menu_bar, x, y)
            || rect_contains(self.geometry.toolbar, x, y)
            || self.series_contains(x, y)
            || rect_contains(self.geometry.status_bar, x, y)
            || self.controls.iter().any(|control| {
                matches!(control.kind, ControlKind::MenuItem | ControlKind::Grid)
                    && rect_contains(control.rect, x, y)
            })
    }

    #[cfg(test)]
    pub(super) fn controls_fit_within(&self, width: u32, height: u32) -> bool {
        let width = i64::from(width);
        let height = i64::from(height);
        self.controls.iter().all(|control| {
            let left = i64::from(control.rect.x);
            let top = i64::from(control.rect.y);
            let right = left + i64::from(control.rect.width);
            let bottom = top + i64::from(control.rect.height);
            left >= 0 && top >= 0 && right <= width && bottom <= height
        })
    }
}

fn menu_tab_geometry(menu: Menu) -> u32 {
    let mut x = 8_u32;
    for (candidate, _, width) in MENU_TABS {
        if candidate == menu {
            return x;
        }
        x = x.saturating_add(width);
    }
    0
}

fn menu_item_count(menu: Menu) -> usize {
    match menu {
        Menu::File => 2,
        Menu::View => 4,
        Menu::Tools => ToolKind::all().len(),
        Menu::Window => 2,
        Menu::GridPicker => 0,
    }
}

fn menu_item(
    menu: Menu,
    index: usize,
    app: &SnapApp,
    workspace_layout: WorkspaceLayout,
) -> Result<(&'static str, WindowAction, bool)> {
    match menu {
        Menu::File => match index {
            0 => Ok(("Open Study...", WindowAction::OpenStudy, false)),
            1 => Ok(("Exit", WindowAction::Exit, false)),
            _ => Err(anyhow!("native File menu row is outside its items")),
        },
        Menu::View => match index {
            0 => Ok((
                "Toggle series preview",
                WindowAction::ToggleSeriesPreview,
                false,
            )),
            1 => Ok((
                "Toggle linked crosshair",
                WindowAction::ToggleCrosshair,
                app.show_crosshair,
            )),
            2 => Ok((
                "Toggle cine playback",
                WindowAction::ToggleCine,
                app.cine.enabled,
            )),
            3 => Ok(("Reset view", WindowAction::ResetView, false)),
            _ => Err(anyhow!("native View menu row is outside its items")),
        },
        Menu::Tools => {
            let tool = ToolKind::all()
                .get(index)
                .copied()
                .ok_or_else(|| anyhow!("native Tools menu row is outside its items"))?;
            Ok((
                tool.label(),
                WindowAction::SelectTool(tool),
                app.active_tool == tool,
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
            _ => Err(anyhow!("native Window menu row is outside its items")),
        },
        Menu::GridPicker => Err(anyhow!("panel grid picker has no menu rows")),
    }
}

fn grid_popup_bounds(
    width: u32,
    height: u32,
    geometry: &ChromeGeometry,
    anchor: Rect,
) -> Result<Option<Rect>> {
    if width < GRID_POPUP_WIDTH || height < GRID_POPUP_HEIGHT {
        return Ok(None);
    }
    let anchor_x =
        u32::try_from(anchor.x).map_err(|_| anyhow!("panel grid picker anchor x is negative"))?;
    let anchor_width = u32::try_from(anchor.width)
        .map_err(|_| anyhow!("panel grid picker anchor width is negative"))?;
    let center = anchor_x
        .checked_add(anchor_width / 2)
        .ok_or_else(|| anyhow!("panel grid picker anchor center overflows"))?;
    let x = center
        .saturating_sub(GRID_POPUP_WIDTH / 2)
        .min(width.saturating_sub(GRID_POPUP_WIDTH));
    let y = geometry
        .menu_height
        .checked_add(geometry.toolbar_height)
        .ok_or_else(|| anyhow!("panel grid picker y overflows"))?
        .min(height.saturating_sub(GRID_POPUP_HEIGHT));
    Ok(Some(make_rect(x, y, GRID_POPUP_WIDTH, GRID_POPUP_HEIGHT)?))
}

fn push_control(
    controls: &mut ArrayVec<ChromeControl, CONTROL_CAPACITY>,
    control: ChromeControl,
) -> Result<()> {
    controls
        .try_push(control)
        .map_err(|_| anyhow!("native control capacity exceeded"))
}

fn make_rect(x: u32, y: u32, width: u32, height: u32) -> Result<Rect> {
    Ok(Rect::new(
        i32::try_from(x).map_err(|_| anyhow!("native control x exceeds i32"))?,
        i32::try_from(y).map_err(|_| anyhow!("native control y exceeds i32"))?,
        i32::try_from(width).map_err(|_| anyhow!("native control width exceeds i32"))?,
        i32::try_from(height).map_err(|_| anyhow!("native control height exceeds i32"))?,
    ))
}

fn rect_contains(rect: Rect, x: f64, y: f64) -> bool {
    if rect.width <= 0 || rect.height <= 0 || !x.is_finite() || !y.is_finite() {
        return false;
    }
    let left = f64::from(rect.x);
    let top = f64::from(rect.y);
    x >= left && y >= top && x < left + f64::from(rect.width) && y < top + f64::from(rect.height)
}
