//! Labels and controls rendered over native viewer panes.

use anyhow::{Result, anyhow};
use metis_platform::rasterizer::CornerRadius;
use metis_platform::{Color, Rect};
use metis_ui_lang::{DisplayCommand, DisplayList};

use super::super::frame::RenderedView;
use super::super::projection::RenderedProjection;
use super::geometry::NativeViewport;
use super::text::{PANEL_DETAIL_SIZE, PANEL_TITLE_SIZE, text_style};

pub(crate) const OVERLAY_BAR_HEIGHT: i32 = 20;
const OVERLAY_MARGIN: i32 = 6;
const OVERLAY_BACKGROUND: Color = Color::rgba(0, 0, 0, 224);
pub(crate) const OVERLAY_TEXT: Color = Color::rgba(255, 255, 160, 255);
pub(crate) fn application_overlay(
    views: &[RenderedView; 3],
    viewports: &[NativeViewport; 3],
    cine_enabled: bool,
    cine_fps: f32,
) -> Result<DisplayList> {
    let mut overlay = DisplayList::default();
    for (view, viewport) in views.iter().zip(viewports) {
        let panel_x =
            i32::try_from(viewport.panel_x).map_err(|_| anyhow!("native overlay x exceeds i32"))?;
        let panel_y =
            i32::try_from(viewport.panel_y).map_err(|_| anyhow!("native overlay y exceeds i32"))?;
        let panel_width = i32::try_from(viewport.panel_width)
            .map_err(|_| anyhow!("native overlay width exceeds i32"))?;
        let panel_height = i32::try_from(viewport.panel_height)
            .map_err(|_| anyhow!("native overlay height exceeds i32"))?;
        if panel_width <= OVERLAY_MARGIN * 2 {
            continue;
        }
        push_overlay_command(
            &mut overlay,
            DisplayCommand::FillRect {
                rect: Rect::new(panel_x, panel_y, panel_width, OVERLAY_BAR_HEIGHT),
                radius: CornerRadius::SQUARE,
                color: OVERLAY_BACKGROUND,
            },
        )?;
        push_overlay_command(
            &mut overlay,
            DisplayCommand::FillRect {
                rect: Rect::new(
                    panel_x,
                    panel_y + panel_height - OVERLAY_BAR_HEIGHT,
                    panel_width,
                    OVERLAY_BAR_HEIGHT,
                ),
                radius: CornerRadius::SQUARE,
                color: OVERLAY_BACKGROUND,
            },
        )?;
        let title = format!("METIS  RITK-SNAP  {}", view.plane_name);
        push_overlay_command(
            &mut overlay,
            DisplayCommand::DrawText {
                text: title,
                x: panel_x + OVERLAY_MARGIN,
                y: panel_y + 2,
                style: text_style(OVERLAY_TEXT, PANEL_TITLE_SIZE)?,
            },
        )?;
        let footer = format!(
            "Slice {}/{}  {}x{}  W:{:.0} C:{:.0}",
            view.slice_index.saturating_add(1),
            view.slice_count,
            view.frame.width(),
            view.frame.height(),
            view.window_level.width,
            view.window_level.center
        );
        let footer = if cine_enabled {
            format!("{footer}  Cine:{cine_fps:.0}fps  Space/-/+")
        } else {
            footer
        };
        push_overlay_command(
            &mut overlay,
            DisplayCommand::DrawText {
                text: footer,
                x: panel_x + OVERLAY_MARGIN,
                y: panel_y + panel_height - OVERLAY_BAR_HEIGHT + 2,
                style: text_style(OVERLAY_TEXT, PANEL_DETAIL_SIZE)?,
            },
        )?;
    }
    Ok(overlay)
}

pub(crate) fn projection_overlay(
    projection: &RenderedProjection,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
) -> Result<DisplayList> {
    fourth_panel_overlay(
        &format!("METIS  RITK-SNAP  3D {}", projection.statistic.label()),
        &format!(
            "Axial {}  {}x{}",
            projection.statistic.label(),
            projection.frame.width(),
            projection.frame.height()
        ),
        panel_x,
        panel_y,
        panel_width,
        panel_height,
    )
}

pub(super) fn fourth_panel_overlay(
    title: &str,
    footer: &str,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
) -> Result<DisplayList> {
    let panel_x =
        i32::try_from(panel_x).map_err(|_| anyhow!("native projection overlay x exceeds i32"))?;
    let panel_y =
        i32::try_from(panel_y).map_err(|_| anyhow!("native projection overlay y exceeds i32"))?;
    let panel_width = i32::try_from(panel_width)
        .map_err(|_| anyhow!("native projection overlay width exceeds i32"))?;
    let panel_height = i32::try_from(panel_height)
        .map_err(|_| anyhow!("native projection overlay height exceeds i32"))?;
    let mut overlay = DisplayList::default();
    if panel_width <= OVERLAY_MARGIN * 2 || panel_height <= OVERLAY_BAR_HEIGHT * 2 {
        return Ok(overlay);
    }
    push_overlay_command(
        &mut overlay,
        DisplayCommand::FillRect {
            rect: Rect::new(panel_x, panel_y, panel_width, OVERLAY_BAR_HEIGHT),
            radius: CornerRadius::SQUARE,
            color: OVERLAY_BACKGROUND,
        },
    )?;
    push_overlay_command(
        &mut overlay,
        DisplayCommand::FillRect {
            rect: Rect::new(
                panel_x,
                panel_y + panel_height - OVERLAY_BAR_HEIGHT,
                panel_width,
                OVERLAY_BAR_HEIGHT,
            ),
            radius: CornerRadius::SQUARE,
            color: OVERLAY_BACKGROUND,
        },
    )?;
    push_overlay_command(
        &mut overlay,
        DisplayCommand::DrawText {
            text: title.to_owned(),
            x: panel_x + OVERLAY_MARGIN,
            y: panel_y + 2,
            style: text_style(OVERLAY_TEXT, PANEL_TITLE_SIZE)?,
        },
    )?;
    push_overlay_command(
        &mut overlay,
        DisplayCommand::DrawText {
            text: footer.to_owned(),
            x: panel_x + OVERLAY_MARGIN,
            y: panel_y + panel_height - OVERLAY_BAR_HEIGHT + 2,
            style: text_style(OVERLAY_TEXT, PANEL_DETAIL_SIZE)?,
        },
    )?;
    Ok(overlay)
}

fn push_overlay_command(display: &mut DisplayList, command: DisplayCommand) -> Result<()> {
    display
        .commands
        .try_reserve(1)
        .map_err(|_| anyhow!("native viewer overlay command allocation failed"))?;
    display.commands.push(command);
    Ok(())
}

pub(super) fn append_overlay_list(target: &mut DisplayList, mut source: DisplayList) -> Result<()> {
    target
        .commands
        .try_reserve(source.commands.len())
        .map_err(|_| anyhow!("native viewer overlay command allocation failed"))?;
    target.commands.append(&mut source.commands);
    Ok(())
}
