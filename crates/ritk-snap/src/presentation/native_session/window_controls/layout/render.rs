//! Drawing for the native viewer workspace.

use super::super::super::layout::text_style;
use super::super::super::layout::WorkspaceLayout;
use super::super::series;
use super::*;
use crate::app::SnapApp;
use crate::presentation::PresentationFrame;
use anyhow::{anyhow, Result};
use arrayvec::ArrayString;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::typeface::draw_text;
use metis_platform::{Color, Framebuffer, Rect};
use std::fmt::Write as _;

const SURFACE_BACKGROUND: Color = Color::rgb(27, 32, 39);
const TOOLBAR_BACKGROUND: Color = Color::rgb(40, 46, 54);
const STATUS_BACKGROUND: Color = Color::rgb(29, 34, 41);
const CONTROL_BACKGROUND: Color = Color::rgb(55, 64, 74);
const CONTROL_ACTIVE: Color = Color::rgb(42, 104, 139);
const CONTROL_TEXT: Color = Color::rgb(235, 240, 245);
const POPUP_BACKGROUND: Color = Color::rgb(31, 37, 45);
const ICON_COLOR: Color = Color::rgb(194, 209, 222);
const ICON_ACTIVE: Color = Color::rgb(180, 232, 255);
const GROUP_TEXT: Color = Color::rgb(161, 176, 190);
const DIVIDER: Color = Color::rgb(82, 92, 103);
const STATUS_TEXT: Color = Color::rgb(190, 203, 215);

impl ChromeLayout {
    pub(in crate::presentation::native_session::window_controls) fn render(
        &self,
        framebuffer: &mut Framebuffer,
        app: &SnapApp,
        series_previews: &[Option<&PresentationFrame>],
        browser: Option<&SeriesBrowser>,
        workspace_layout: WorkspaceLayout,
        active_panel: usize,
        displayed_series: &[Option<usize>],
    ) -> Result<()> {
        fill_rect(
            framebuffer,
            self.geometry.menu_bar,
            CornerRadius::SQUARE,
            SURFACE_BACKGROUND,
        );
        fill_rect(
            framebuffer,
            self.geometry.toolbar,
            CornerRadius::SQUARE,
            TOOLBAR_BACKGROUND,
        );
        if self.geometry.series_preview.width > 0 {
            series::render(
                framebuffer,
                self.geometry.series_preview,
                browser,
                app,
                series_previews,
                active_panel,
                displayed_series,
            )?;
        }
        fill_rect(
            framebuffer,
            self.geometry.status_bar,
            CornerRadius::SQUARE,
            STATUS_BACKGROUND,
        );
        if let Some(popup) = self.grid_popup {
            fill_rect(framebuffer, popup, CornerRadius::SQUARE, POPUP_BACKGROUND);
            let border = Color::rgb(86, 101, 115);
            fill_rect(
                framebuffer,
                Rect::new(popup.x, popup.y, popup.width, 1),
                CornerRadius::SQUARE,
                border,
            );
            fill_rect(
                framebuffer,
                Rect::new(
                    popup.x,
                    popup.y.saturating_add(popup.height).saturating_sub(1),
                    popup.width,
                    1,
                ),
                CornerRadius::SQUARE,
                border,
            );
            fill_rect(
                framebuffer,
                Rect::new(popup.x, popup.y, 1, popup.height),
                CornerRadius::SQUARE,
                border,
            );
            fill_rect(
                framebuffer,
                Rect::new(
                    popup.x.saturating_add(popup.width).saturating_sub(1),
                    popup.y,
                    1,
                    popup.height,
                ),
                CornerRadius::SQUARE,
                border,
            );
            let picker_style = text_style(CONTROL_TEXT, 12)?;
            draw_text(
                framebuffer,
                popup.x.saturating_add(10),
                popup.y.saturating_add(11),
                "PANEL LAYOUT  |  COLUMNS x ROWS",
                picker_style,
            );
        }

        let control_style = text_style(CONTROL_TEXT, 12)?;
        let menu_item_style = text_style(CONTROL_TEXT, 13)?;
        let group_style = text_style(GROUP_TEXT, 9)?;
        for group in &self.groups {
            let label_y = i32::try_from(self.geometry.menu_height)
                .map_err(|_| anyhow!("native toolbar y exceeds i32"))?
                .saturating_add(4);
            draw_text(framebuffer, group.x, label_y, group.label, group_style);
            if group.x > 10 {
                let divider_y = label_y.saturating_add(18);
                fill_rect(
                    framebuffer,
                    Rect::new(group.x - 8, divider_y, 1, 30),
                    CornerRadius::SQUARE,
                    DIVIDER,
                );
            }
        }
        for control in &self.controls {
            if !matches!(control.kind, ControlKind::MenuTab) || control.active {
                let background = if control.active {
                    CONTROL_ACTIVE
                } else {
                    CONTROL_BACKGROUND
                };
                fill_rect(framebuffer, control.rect, CornerRadius::SQUARE, background);
            }
            if matches!(control.kind, ControlKind::Toolbar) {
                draw_control_icon(framebuffer, control, app.cine.enabled);
            }
            let x = control
                .rect
                .x
                .checked_add(if matches!(control.kind, ControlKind::MenuTab) {
                    12
                } else if matches!(control.kind, ControlKind::Toolbar) {
                    30
                } else if matches!(control.kind, ControlKind::Grid) {
                    7
                } else {
                    10
                })
                .ok_or_else(|| anyhow!("native control text x overflows"))?;
            let y = control
                .rect
                .y
                .checked_add((control.rect.height - 14).max(0) / 2)
                .ok_or_else(|| anyhow!("native control text y overflows"))?;
            let style = if matches!(control.kind, ControlKind::MenuItem) {
                menu_item_style
            } else {
                control_style
            };
            draw_text(framebuffer, x, y, control.label, style);
        }

        let study_status = if app.loaded.is_some() || displayed_series.iter().any(Option::is_some) {
            "Study loaded"
        } else {
            "No study loaded  |  File > Open Study"
        };
        let right_status = match (app.show_crosshair, app.cine.enabled) {
            (true, true) => "Crosshair: On  |  Cine: Playing",
            (true, false) => "Crosshair: On  |  Cine: Paused",
            (false, true) => "Crosshair: Off  |  Cine: Playing",
            (false, false) => "Crosshair: Off  |  Cine: Paused",
        };
        let mut layout_status = ArrayString::<48>::new();
        if let Some(grid) = workspace_layout.grid() {
            write!(
                &mut layout_status,
                "Panel {} of {}",
                active_panel.saturating_add(1),
                grid.panel_count()
            )
            .map_err(|_| anyhow!("native panel status exceeds its display buffer"))?;
        } else {
            layout_status
                .try_push_str("MPR workspace")
                .map_err(|_| anyhow!("native layout status exceeds its display buffer"))?;
        }
        let status_y = self
            .geometry
            .status_bar
            .y
            .checked_add((self.geometry.status_bar.height - 14).max(0) / 2)
            .ok_or_else(|| anyhow!("native status text y overflows"))?;
        let status_style = text_style(STATUS_TEXT, 12)?;
        draw_text(framebuffer, 12, status_y, study_status, status_style);
        let width = i32::try_from(framebuffer.width())
            .map_err(|_| anyhow!("native status width exceeds i32"))?;
        let layout_width = status_style
            .extent(0, status_y, layout_status.as_str())
            .ok_or_else(|| anyhow!("native layout status has no visible extent"))?
            .width;
        draw_text(
            framebuffer,
            width
                .saturating_sub(layout_width)
                .checked_div(2)
                .ok_or_else(|| anyhow!("native layout status x calculation failed"))?,
            status_y,
            layout_status.as_str(),
            status_style,
        );
        let text_width = status_style
            .extent(0, status_y, right_status)
            .ok_or_else(|| anyhow!("native status text has no visible extent"))?
            .width;
        let x = width.saturating_sub(text_width).saturating_sub(12).max(12);
        draw_text(framebuffer, x, status_y, right_status, status_style);
        Ok(())
    }
}

fn draw_control_icon(framebuffer: &mut Framebuffer, control: &ChromeControl, cine_enabled: bool) {
    let x = control.rect.x.saturating_add(9);
    let y = control
        .rect
        .y
        .saturating_add((control.rect.height.saturating_sub(14)) / 2);
    let color = if control.active {
        ICON_ACTIVE
    } else {
        ICON_COLOR
    };
    match control.action {
        WindowAction::OpenStudy => {
            icon_bar(framebuffer, x, y + 2, 7, 2, color);
            icon_bar(framebuffer, x, y + 4, 14, 2, color);
            icon_bar(framebuffer, x, y + 6, 2, 6, color);
            icon_bar(framebuffer, x + 12, y + 6, 2, 6, color);
            icon_bar(framebuffer, x + 2, y + 10, 11, 2, color);
        }
        WindowAction::SelectTool(ToolKind::WindowLevel) => {
            icon_bar(framebuffer, x + 2, y + 2, 2, 10, color);
            icon_bar(framebuffer, x + 10, y + 2, 2, 10, color);
            icon_bar(framebuffer, x + 4, y + 2, 6, 2, color);
            icon_bar(framebuffer, x + 4, y + 6, 6, 2, color);
            icon_bar(framebuffer, x + 4, y + 10, 6, 2, color);
        }
        WindowAction::SelectTool(ToolKind::Pan) => {
            icon_bar(framebuffer, x + 6, y + 2, 2, 10, color);
            icon_bar(framebuffer, x + 2, y + 6, 10, 2, color);
            icon_bar(framebuffer, x + 5, y, 4, 2, color);
            icon_bar(framebuffer, x + 5, y + 12, 4, 2, color);
            icon_bar(framebuffer, x, y + 5, 2, 4, color);
            icon_bar(framebuffer, x + 12, y + 5, 2, 4, color);
        }
        WindowAction::SelectTool(ToolKind::Zoom) => {
            icon_bar(framebuffer, x + 1, y + 1, 9, 2, color);
            icon_bar(framebuffer, x + 1, y + 3, 2, 7, color);
            icon_bar(framebuffer, x + 8, y + 3, 2, 7, color);
            icon_bar(framebuffer, x + 3, y + 8, 7, 2, color);
            icon_bar(framebuffer, x + 9, y + 9, 5, 2, color);
            icon_bar(framebuffer, x + 12, y + 11, 2, 2, color);
        }
        WindowAction::SelectTool(ToolKind::MeasureLength) => {
            icon_bar(framebuffer, x + 2, y + 6, 11, 2, color);
            icon_bar(framebuffer, x + 1, y + 4, 2, 6, color);
            icon_bar(framebuffer, x + 12, y + 4, 2, 6, color);
        }
        WindowAction::SelectTool(ToolKind::MeasureAngle) => {
            icon_bar(framebuffer, x + 2, y + 10, 11, 2, color);
            icon_bar(framebuffer, x + 2, y + 3, 2, 7, color);
            icon_bar(framebuffer, x + 4, y + 8, 2, 2, color);
            icon_bar(framebuffer, x + 6, y + 6, 2, 2, color);
            icon_bar(framebuffer, x + 8, y + 5, 2, 2, color);
            icon_bar(framebuffer, x + 10, y + 4, 2, 2, color);
        }
        WindowAction::SelectTool(ToolKind::Crosshair) | WindowAction::ToggleCrosshair => {
            icon_bar(framebuffer, x + 6, y + 1, 2, 12, color);
            icon_bar(framebuffer, x + 1, y + 6, 12, 2, color);
            icon_bar(framebuffer, x + 4, y + 4, 6, 6, color);
        }
        WindowAction::ToggleCine => {
            if cine_enabled {
                icon_bar(framebuffer, x + 3, y + 2, 3, 10, color);
                icon_bar(framebuffer, x + 9, y + 2, 3, 10, color);
            } else {
                for (row, width) in [(0, 3), (2, 6), (4, 9), (6, 6), (8, 3)] {
                    icon_bar(framebuffer, x + 3, y + 2 + row, width, 2, color);
                }
            }
        }
        WindowAction::OpenMenu(super::super::Menu::GridPicker) => {
            icon_bar(framebuffer, x + 1, y + 1, 5, 12, color);
            icon_bar(framebuffer, x + 8, y + 1, 5, 12, color);
            icon_bar(framebuffer, x + 6, y + 1, 1, 12, color);
        }
        _ => {}
    }
}

fn icon_bar(framebuffer: &mut Framebuffer, x: i32, y: i32, width: i32, height: i32, color: Color) {
    fill_rect(
        framebuffer,
        Rect::new(x, y, width, height),
        CornerRadius::SQUARE,
        color,
    );
}
