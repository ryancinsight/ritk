//! Labels and controls rendered over native viewer panes.

use anyhow::{anyhow, Result};
use metis_platform::rasterizer::CornerRadius;
use metis_platform::{Color, Rect};
use metis_ui_lang::{DisplayCommand, DisplayList};

use super::super::frame::RenderedView;
use super::super::projection::RenderedProjection;
use super::geometry::NativeViewport;
use super::text::{text_command, text_style, PANEL_DETAIL_SIZE, PANEL_TITLE_SIZE};

pub(crate) const OVERLAY_BAR_HEIGHT: i32 = 20;
const OVERLAY_MARGIN: i32 = 6;
const OVERLAY_BACKGROUND: Color = Color::rgba(0, 0, 0, 224);
pub(crate) const OVERLAY_TEXT: Color = Color::rgba(255, 255, 160, 255);
pub(crate) fn application_overlay(
    views: &[RenderedView; 3],
    viewports: &[NativeViewport],
    cine_enabled: bool,
    cine_fps: f32,
) -> Result<OverlayCommands> {
    let mut overlay = OverlayCommands::with_capacity(views.len() * 4)?;
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
        overlay.push(DisplayCommand::FillRect {
            rect: Rect::new(panel_x, panel_y, panel_width, OVERLAY_BAR_HEIGHT),
            radius: CornerRadius::SQUARE,
            color: OVERLAY_BACKGROUND,
        });
        overlay.push(DisplayCommand::FillRect {
            rect: Rect::new(
                panel_x,
                panel_y + panel_height - OVERLAY_BAR_HEIGHT,
                panel_width,
                OVERLAY_BAR_HEIGHT,
            ),
            radius: CornerRadius::SQUARE,
            color: OVERLAY_BACKGROUND,
        });
        let title = format!("METIS  RITK-SNAP  {}", view.plane_name);
        overlay.push(text_command(
            title,
            panel_x + OVERLAY_MARGIN,
            panel_y + 2,
            text_style(OVERLAY_TEXT, PANEL_TITLE_SIZE)?,
        ));
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
        overlay.push(text_command(
            footer,
            panel_x + OVERLAY_MARGIN,
            panel_y + panel_height - OVERLAY_BAR_HEIGHT + 2,
            text_style(OVERLAY_TEXT, PANEL_DETAIL_SIZE)?,
        ));
    }
    Ok(overlay)
}

pub(crate) fn projection_overlay(
    projection: &RenderedProjection,
    panel_x: u32,
    panel_y: u32,
    panel_width: u32,
    panel_height: u32,
) -> Result<OverlayCommands> {
    let panel_x =
        i32::try_from(panel_x).map_err(|_| anyhow!("native projection overlay x exceeds i32"))?;
    let panel_y =
        i32::try_from(panel_y).map_err(|_| anyhow!("native projection overlay y exceeds i32"))?;
    let panel_width = i32::try_from(panel_width)
        .map_err(|_| anyhow!("native projection overlay width exceeds i32"))?;
    let panel_height = i32::try_from(panel_height)
        .map_err(|_| anyhow!("native projection overlay height exceeds i32"))?;
    let mut overlay = OverlayCommands::with_capacity(4)?;
    if panel_width <= OVERLAY_MARGIN * 2 || panel_height <= OVERLAY_BAR_HEIGHT * 2 {
        return Ok(overlay);
    }
    overlay.push(DisplayCommand::FillRect {
        rect: Rect::new(panel_x, panel_y, panel_width, OVERLAY_BAR_HEIGHT),
        radius: CornerRadius::SQUARE,
        color: OVERLAY_BACKGROUND,
    });
    overlay.push(DisplayCommand::FillRect {
        rect: Rect::new(
            panel_x,
            panel_y + panel_height - OVERLAY_BAR_HEIGHT,
            panel_width,
            OVERLAY_BAR_HEIGHT,
        ),
        radius: CornerRadius::SQUARE,
        color: OVERLAY_BACKGROUND,
    });
    overlay.push(text_command(
        format!("METIS  RITK-SNAP  3D {}", projection.statistic.label()),
        panel_x + OVERLAY_MARGIN,
        panel_y + 2,
        text_style(OVERLAY_TEXT, PANEL_TITLE_SIZE)?,
    ));
    overlay.push(text_command(
        format!(
            "Axial {}  {}x{}",
            projection.statistic.label(),
            projection.frame.width(),
            projection.frame.height()
        ),
        panel_x + OVERLAY_MARGIN,
        panel_y + panel_height - OVERLAY_BAR_HEIGHT + 2,
        text_style(OVERLAY_TEXT, PANEL_DETAIL_SIZE)?,
    ));
    Ok(overlay)
}

/// The painter-ordered commands an overlay contributes.
///
/// The display list owns its command storage, so an overlay builds its
/// commands here and converts once, at the boundary the renderer reads.
pub(crate) struct OverlayCommands {
    commands: Vec<DisplayCommand>,
}

impl OverlayCommands {
    /// Reserves room for `capacity` commands before any is written.
    fn with_capacity(capacity: usize) -> Result<Self> {
        let mut commands = Vec::new();
        commands
            .try_reserve(capacity)
            .map_err(|_| anyhow!("native viewer overlay command allocation failed"))?;
        Ok(Self { commands })
    }

    /// Appends one command; the reserved capacity covers it.
    fn push(&mut self, command: DisplayCommand) {
        self.commands.push(command);
    }

    /// Appends another overlay's commands after this one's.
    pub(crate) fn extend(&mut self, other: Self) -> Result<()> {
        self.commands
            .try_reserve(other.commands.len())
            .map_err(|_| anyhow!("native viewer overlay command allocation failed"))?;
        self.commands.extend(other.commands);
        Ok(())
    }

    /// Hands the commands to the display list that renders them.
    pub(crate) fn into_display_list(self) -> DisplayList {
        DisplayList::from_commands(self.commands)
    }
}
