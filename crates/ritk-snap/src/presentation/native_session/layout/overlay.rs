//! Labels and controls rendered over native viewer panes.

use super::overlay_clinical::{append_clinical_corners, append_scale_bar};

use anyhow::{anyhow, Result};
use metis_platform::rasterizer::CornerRadius;
use metis_platform::{Color, Rect};
use metis_ui_lang::{DisplayCommand, DisplayList};

use super::super::frame::RenderedView;
use super::super::projection::RenderedProjection;
use super::geometry::NativeViewport;
use super::text::{
    fit_text, text_command, text_style, TextStyle, PANEL_DETAIL_SIZE, PANEL_TITLE_SIZE,
};
use crate::ui::coordinate_system::OrientationLabels;
use crate::ui::coordinate_system::{edge_slot, orientation_labels, remap_labels_by_slots};
use crate::ui::ViewTransform;

pub(crate) const OVERLAY_BAR_HEIGHT: i32 = 20;
pub(crate) const OVERLAY_MARGIN: i32 = 6;
const OVERLAY_BACKGROUND: Color = Color::rgba(0, 0, 0, 224);
pub(crate) const OVERLAY_TEXT: Color = Color::rgba(255, 255, 160, 255);
/// Clinical content for one native overlay composition.
///
/// The session builds this from the loaded volume beside the rendered views
/// so the overlay builder stays pure over plain data.
pub(crate) struct ClinicalOverlay<'a> {
    /// Volume direction cosines in row-major order.
    pub(crate) direction: [f64; 9],
    /// Patient name for the top-left corner; blanked when anonymized.
    pub(crate) patient_name: Option<&'a str>,
    /// Patient identifier for the top-left corner; blanked when anonymized.
    pub(crate) patient_id: Option<&'a str>,
    /// Series description for the top-right corner; blanked when anonymized.
    pub(crate) series_description: Option<&'a str>,
    /// Modality for the top-right corner; shown even when anonymized.
    pub(crate) modality: Option<&'a str>,
    /// Study date for the top-right corner; blanked when anonymized.
    pub(crate) study_date: Option<&'a str>,
    /// Voxel spacing in volume order `[dz, dy, dx]`, in millimetres.
    pub(crate) spacing: [f64; 3],
    /// Display zoom factor for the bottom-left corner.
    pub(crate) zoom: f32,
    /// Cursor voxel intensity for the bottom-right readout.
    pub(crate) cursor_value: Option<f32>,
    /// Blank patient corners while keeping modality and measurements.
    pub(crate) anonymize: bool,
}

/// Orientation letters for one pane after the view transform.
///
/// Returns the base [`orientation_labels`] remapped through the transformed
/// source edges so flipped or rotated panes label the displayed edges.
/// Degenerate sizes or unmappable coordinates keep the untransformed labels.
pub(crate) fn clinical_labels_for_view(
    axis: usize,
    direction: &[f64; 9],
    source_size: [usize; 2],
    transform: ViewTransform,
) -> OrientationLabels {
    let labels = orientation_labels(axis, direction);
    if transform.is_identity() || source_size.contains(&0) {
        return labels;
    }
    let Some(width) = u32::try_from(source_size[0]).ok().map(f64::from) else {
        return labels;
    };
    let Some(height) = u32::try_from(source_size[1]).ok().map(f64::from) else {
        return labels;
    };
    let output_size = transform.output_size(source_size);
    let source_edges = [
        [0.0, height * 0.5],
        [width, height * 0.5],
        [width * 0.5, 0.0],
        [width * 0.5, height],
    ];
    let mut slots = [0; 4];
    for (slot, point) in slots.iter_mut().zip(source_edges) {
        let output = transform.source_to_output_coordinates(point, source_size);
        *slot = edge_slot(output, output_size);
    }
    remap_labels_by_slots(labels, slots)
}

pub(crate) fn application_overlay(
    views: &[RenderedView; 3],
    viewports: &[NativeViewport],
    cine_enabled: bool,
    cine_fps: f32,
    clinical: Option<&ClinicalOverlay<'_>>,
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
        if panel_width <= OVERLAY_MARGIN * 2 || panel_height < 2 * OVERLAY_BAR_HEIGHT {
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
        let title_style = text_style(OVERLAY_TEXT, PANEL_TITLE_SIZE)?;
        let detail_style = text_style(OVERLAY_TEXT, PANEL_DETAIL_SIZE)?;
        let text_width =
            panel_width - 2 * OVERLAY_MARGIN - super::super::slice_scrollbar::RIGHT_GUTTER;
        let title = fit_text(&title, text_width, title_style);
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
        let footer = fit_text(&footer, text_width, detail_style);
        for (text, style, bar_top) in [
            (title, title_style, panel_y),
            (
                footer,
                detail_style,
                panel_y + panel_height - OVERLAY_BAR_HEIGHT,
            ),
        ] {
            if let Some(bounds) = style.extent(0, 0, &text) {
                let bottom = panel_y + panel_height - bounds.height;
                if bottom < panel_y {
                    continue;
                }
                let top =
                    (bar_top + (OVERLAY_BAR_HEIGHT - bounds.height) / 2).clamp(panel_y, bottom);
                overlay.push(text_command(
                    text,
                    panel_x + OVERLAY_MARGIN - bounds.x,
                    top - bounds.y,
                    style,
                ));
            }
        }
        if let Some(clinical) = clinical {
            append_edge_letters(
                &mut overlay,
                view,
                panel_x,
                panel_y,
                panel_width,
                panel_height,
                &clinical.direction,
            )?;
            append_clinical_corners(
                &mut overlay,
                panel_x,
                panel_y,
                panel_width,
                panel_height,
                clinical,
            )?;
            append_scale_bar(&mut overlay, view, viewport, panel_y, panel_height)?;
        }
    }
    Ok(overlay)
}

/// Orientation letters on the four panel edges for one composed view.
///
/// Letters sit inside the panel chrome: below the title bar, above the
/// footer bar, and clear of the left and right panel edges. Panes too short
/// to separate the edges keep their bars without letters.
fn append_edge_letters(
    overlay: &mut OverlayCommands,
    view: &RenderedView,
    panel_x: i32,
    panel_y: i32,
    panel_width: i32,
    panel_height: i32,
    direction: &[f64; 9],
) -> Result<()> {
    let style = text_style(OVERLAY_TEXT, PANEL_TITLE_SIZE)?;
    let line = style.extent(0, 0, "R").map_or(0, |bounds| bounds.height);
    let chrome = line.max(OVERLAY_BAR_HEIGHT);
    let top = panel_y + chrome;
    let bottom = panel_y + panel_height - chrome;
    if panel_height < 2 * chrome + 2 * (line + OVERLAY_MARGIN) + line {
        return Ok(());
    }
    let labels = clinical_labels_for_view(view.axis, direction, view.source_size, view.transform);
    let left = panel_x + OVERLAY_MARGIN;
    let right =
        panel_x + panel_width - OVERLAY_MARGIN - super::super::slice_scrollbar::RIGHT_GUTTER;
    let center_x = left + (right - left) / 2;
    let center_y = panel_y + panel_height / 2;
    for letter in [labels.top, labels.bottom, labels.left, labels.right]
        .into_iter()
        .enumerate()
    {
        let (edge, letter) = letter;
        let Some(bounds) = style.extent(0, 0, letter) else {
            continue;
        };
        if bounds.width > (right - left) / 2 {
            continue;
        }
        let (x, y) = match edge {
            0 => (center_x - bounds.width / 2, top),
            1 => (center_x - bounds.width / 2, bottom - line),
            2 => (left, center_y - line / 2),
            _ => (right - bounds.width, center_y - line / 2),
        };
        overlay.push(text_command(letter, x - bounds.x, y - bounds.y, style));
    }
    Ok(())
}

/// Longest readable scale bar for `px_per_mm` screen pixels per millimetre.
///
/// Picks the largest 1/2/5-step length whose bar fits `max_px` (capped at
/// [`MAX_SCALE_BAR_PX`]) and stays legible ([`MIN_SCALE_BAR_PX`]). Returns
/// the physical length with its exact pixel width.
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
        format!("METIS  RITK-SNAP  3D {}", projection.kind.label()),
        panel_x + OVERLAY_MARGIN,
        panel_y + 2,
        text_style(OVERLAY_TEXT, PANEL_TITLE_SIZE)?,
    ));
    overlay.push(text_command(
        format!(
            "Axial {}  {}x{}",
            projection.kind.label(),
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
    pub(crate) fn with_capacity(capacity: usize) -> Result<Self> {
        let mut commands = Vec::new();
        commands
            .try_reserve(capacity)
            .map_err(|_| anyhow!("native viewer overlay command allocation failed"))?;
        Ok(Self { commands })
    }

    /// Appends one command; the reserved capacity covers it.
    pub(crate) fn push(&mut self, command: DisplayCommand) {
        self.commands.push(command);
    }

    pub(super) fn overlaps_text(&self, region: Rect) -> bool {
        self.text_overlapping(region).next().is_some()
    }

    fn text_overlapping(&self, region: Rect) -> impl Iterator<Item = Rect> + '_ {
        self.commands.iter().filter_map(move |command| {
            let DisplayCommand::DrawText { text, x, y, style } = command else {
                return None;
            };
            style.extent(*x, *y, text).filter(|text| {
                text.x < region.x + region.width
                    && region.x < text.x + text.width
                    && text.y < region.y + region.height
                    && region.y < text.y + text.height
            })
        })
    }

    pub(super) fn measurement_label_bounds(panel: Rect, margin: i32) -> Rect {
        Rect::new(
            panel.x + margin,
            panel.y + OVERLAY_BAR_HEIGHT,
            panel.width
                - super::super::slice_scrollbar::RIGHT_GUTTER
                - super::overlay_display::COLORBAR_WIDTH
                - 2 * margin,
            panel.height - 2 * OVERLAY_BAR_HEIGHT,
        )
    }

    pub(super) fn push_text_inside(
        &mut self,
        text: String,
        style: TextStyle,
        origin: (i32, i32),
        bounds: Rect,
    ) {
        let Some(glyph) = style.extent(0, 0, &text) else {
            return;
        };
        if glyph.width > bounds.width || glyph.height > bounds.height {
            return;
        }
        let right = bounds.x + bounds.width - glyph.width;
        let bottom = bounds.y + bounds.height - glyph.height;
        let x = origin.0.saturating_add(glyph.x).clamp(bounds.x, right);
        let y = origin.1.saturating_add(glyph.y).clamp(bounds.y, bottom);
        for row in (y..=bottom).chain((bounds.y..y).rev()) {
            let mut region = Rect::new(x, row, glyph.width, glyph.height);
            while region.x <= right {
                let edge = self
                    .text_overlapping(region)
                    .map(|rect| rect.x + rect.width)
                    .max();
                let Some(edge) = edge else {
                    break;
                };
                region.x = edge;
            }
            if region.x > right {
                region.x = x;
                while region.x >= bounds.x {
                    let edge = self.text_overlapping(region).map(|rect| rect.x).min();
                    let Some(edge) = edge else {
                        break;
                    };
                    region.x = edge - glyph.width;
                }
            }
            if region.x >= bounds.x {
                self.push(text_command(text, region.x - glyph.x, row - glyph.y, style));
                return;
            }
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurement_labels_fit_before_the_colorbar() {
        let mut overlay = OverlayCommands::with_capacity(6).expect("overlay storage");
        let style = text_style(OVERLAY_TEXT, PANEL_DETAIL_SIZE).expect("detail typeface");
        let pane = Rect::new(0, 0, 336, 260);
        let bounds = OverlayCommands::measurement_label_bounds(pane, 8);
        let colorbar_x = bounds.x + bounds.width - 12;
        overlay.push(text_command("1000", colorbar_x, 86, style));
        let labels = ["90.0 deg", "80.0 deg", "70.0 deg", "60.0 deg", "12.0 mm"];
        for label in labels {
            overlay.push_text_inside(label.into(), style, (colorbar_x, 86), bounds);
        }
        assert_eq!(overlay.commands.len(), labels.len() + 1);
        let mut placed = OverlayCommands::with_capacity(6).expect("overlay storage");
        for (command, expected) in overlay
            .commands
            .into_iter()
            .zip(["1000"].into_iter().chain(labels))
        {
            let DisplayCommand::DrawText { text, x, y, style } = &command else {
                panic!("all commands are text");
            };
            assert_eq!(text.as_ref(), expected);
            let glyph = style.extent(*x, *y, text).expect("glyph bounds");
            assert!(
                !placed.overlaps_text(glyph),
                "label {text} overlaps prior text"
            );
            if expected != "1000" {
                assert!(bounds.contains(glyph.x, glyph.y));
                assert!(bounds.contains(glyph.x + glyph.width - 1, glyph.y + glyph.height - 1));
            }
            placed.push(command);
        }
    }
}
