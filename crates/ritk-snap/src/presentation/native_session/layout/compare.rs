//! Grid layouts for independently navigable DICOM series.

mod annotations;
mod panel_chrome;
use self::annotations::draw_dicom_corner_values;
use self::panel_chrome::draw_panel_actions;
use super::super::frame::RenderedView;
use super::composition::blit_frame;
use super::geometry::{
    placement_with_bounds, NativeViewport, ScreenRect, ViewportArea, VIEW_GAP_PIXELS,
};
use super::text::{draw_text, text_style};
use crate::tools::interaction::ViewportOffset;
use anyhow::{anyhow, bail, Result};
use arrayvec::ArrayVec;
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Framebuffer, Rect};
use std::num::NonZeroU8;

pub(in crate::presentation::native_session) const MAX_GRID_PANELS: usize = 20;
pub(in crate::presentation::native_session) const MAX_COMPARISON_PANELS: usize =
    MAX_GRID_PANELS - 1;
pub(in crate::presentation::native_session) const MAX_GRID_COLUMNS: u32 = 5;
pub(in crate::presentation::native_session) const MAX_GRID_ROWS: u32 = 4;
pub(super) const PANEL_HEADER_HEIGHT: u32 = 28;
const PANEL_ACTION_WIDTH: i32 = 16;
const PANEL_ACTION_GAP: i32 = 2;
const WORKSPACE_BACKGROUND: Color = Color::rgb(12, 14, 18);
const PANEL_HEADER: Color = Color::rgb(40, 47, 56);
const ACTIVE_PANEL: Color = Color::rgb(38, 111, 145);
const INACTIVE_PANEL: Color = Color::rgb(73, 83, 94);
const HEADER_TEXT: Color = Color::rgb(225, 233, 240);
const PLACEHOLDER_TEXT: Color = Color::rgb(157, 171, 184);
const PANEL_ACTION_BACKGROUND: Color = Color::rgb(57, 66, 77);
const PANEL_ACTION_FOREGROUND: Color = Color::rgb(220, 228, 236);
const DICOM_ANNOTATION_TEXT: Color = Color::rgb(238, 244, 250);
const DICOM_ANNOTATION_SHADOW: Color = Color::rgb(0, 0, 0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::presentation::native_session) enum PanelHeaderAction {
    Maximize,
    Close,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::presentation::native_session) enum WorkspaceLayout {
    Orthogonal,
    Panels(PanelGrid),
}

impl WorkspaceLayout {
    pub(in crate::presentation::native_session) const fn grid(self) -> Option<PanelGrid> {
        match self {
            Self::Orthogonal => None,
            Self::Panels(grid) => Some(grid),
        }
    }

    pub(in crate::presentation::native_session) const fn is_grid(self) -> bool {
        matches!(self, Self::Panels(_))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::presentation::native_session) struct PanelGrid {
    columns: NonZeroU8,
    rows: NonZeroU8,
}

impl PanelGrid {
    pub(in crate::presentation::native_session) fn new(columns: u32, rows: u32) -> Option<Self> {
        if columns == 0 || columns > MAX_GRID_COLUMNS || rows == 0 || rows > MAX_GRID_ROWS {
            return None;
        }
        let columns = u8::try_from(columns).ok()?;
        let rows = u8::try_from(rows).ok()?;
        let columns = NonZeroU8::new(columns)?;
        let rows = NonZeroU8::new(rows)?;
        Some(Self { columns, rows })
    }

    pub(in crate::presentation::native_session) fn dimensions(self) -> (u32, u32) {
        (u32::from(self.columns.get()), u32::from(self.rows.get()))
    }

    pub(in crate::presentation::native_session) fn panel_count(self) -> usize {
        usize::from(self.columns.get()) * usize::from(self.rows.get())
    }

    pub(in crate::presentation::native_session) fn containing_panel(
        panel_index: usize,
    ) -> Option<Self> {
        let required_panels = panel_index.checked_add(1)?;
        let mut selected = None;
        for rows in 1..=MAX_GRID_ROWS {
            for columns in 1..=MAX_GRID_COLUMNS {
                let grid = Self::new(columns, rows)?;
                if grid.panel_count() < required_panels {
                    continue;
                }
                let rank = (grid.panel_count(), rows, columns);
                if selected.is_none_or(|(best_rank, _)| rank < best_rank) {
                    selected = Some((rank, grid));
                }
            }
        }
        selected.map(|(_, grid)| grid)
    }

    pub(in crate::presentation::native_session) fn menu_label(self) -> Option<&'static str> {
        let columns_per_row = usize::try_from(MAX_GRID_COLUMNS).ok()?;
        let row_offset =
            usize::from(self.rows.get().saturating_sub(1)).checked_mul(columns_per_row)?;
        let index = row_offset.checked_add(usize::from(self.columns.get().saturating_sub(1)))?;
        GRID_LABELS.get(index).copied()
    }
}

const GRID_LABELS: [&str; MAX_GRID_PANELS] = [
    "1 x 1", "2 x 1", "3 x 1", "4 x 1", "5 x 1", "1 x 2", "2 x 2", "3 x 2", "4 x 2", "5 x 2",
    "1 x 3", "2 x 3", "3 x 3", "4 x 3", "5 x 3", "1 x 4", "2 x 4", "3 x 4", "4 x 4", "5 x 4",
];

#[derive(Clone, Copy)]
pub(in crate::presentation::native_session) struct GridPanel<'a> {
    pub(in crate::presentation::native_session) view: Option<&'a RenderedView>,
    pub(in crate::presentation::native_session) label: &'a str,
    pub(in crate::presentation::native_session) navigation: (f32, ViewportOffset),
    pub(in crate::presentation::native_session) maximized: bool,
}

pub(in crate::presentation::native_session) fn panel_header_action_at(
    viewports: &[NativeViewport],
    x: f64,
    y: f64,
    maximized: bool,
) -> Option<(usize, PanelHeaderAction)> {
    for (index, viewport) in viewports.iter().enumerate().rev() {
        if (maximized || viewports.len() > 1)
            && panel_action_rect(*viewport, PanelHeaderAction::Maximize)
                .is_some_and(|rect| rect_contains(rect, x, y))
        {
            return Some((index, PanelHeaderAction::Maximize));
        }
        if panel_action_rect(*viewport, PanelHeaderAction::Close)
            .is_some_and(|rect| rect_contains(rect, x, y))
        {
            return Some((index, PanelHeaderAction::Close));
        }
    }
    None
}

fn panel_action_rect(viewport: NativeViewport, action: PanelHeaderAction) -> Option<Rect> {
    let panel_x = i32::try_from(viewport.panel_x).ok()?;
    let panel_y = i32::try_from(viewport.panel_y).ok()?;
    let panel_width = i32::try_from(viewport.panel_width).ok()?;
    if panel_width < PANEL_ACTION_WIDTH * 2 + PANEL_ACTION_GAP {
        return None;
    }
    let right = panel_x.checked_add(panel_width)?;
    let offset = match action {
        PanelHeaderAction::Maximize => PANEL_ACTION_WIDTH + PANEL_ACTION_GAP,
        PanelHeaderAction::Close => 0,
    };
    let x = right.checked_sub(PANEL_ACTION_WIDTH)?.checked_sub(offset)?;
    let header_height = i32::try_from(PANEL_HEADER_HEIGHT).ok()?;
    let y = panel_y.checked_sub(header_height)?.checked_add(4)?;
    Some(Rect::new(
        x,
        y,
        PANEL_ACTION_WIDTH,
        header_height.saturating_sub(8),
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

pub(in crate::presentation::native_session) struct SeriesGrid {
    pub(in crate::presentation::native_session) framebuffer: Framebuffer,
    pub(in crate::presentation::native_session) viewports:
        ArrayVec<NativeViewport, MAX_GRID_PANELS>,
}

pub(in crate::presentation::native_session) fn surface_frames_grid(
    panels: &[GridPanel<'_>],
    grid: PanelGrid,
    active_panel: usize,
    primary_placeholder: &RenderedView,
    surface_size: [u32; 2],
    viewport_area: ViewportArea,
) -> Result<SeriesGrid> {
    let [surface_width, surface_height] = surface_size;
    if surface_width == 0 || surface_height == 0 {
        bail!("native series-grid surface dimensions must be nonzero");
    }
    if panels.len() != grid.panel_count() {
        bail!(
            "native series grid has {} panel states for a {}-panel layout",
            panels.len(),
            grid.panel_count()
        );
    }
    if active_panel >= panels.len() {
        bail!("native series-grid active panel is outside its layout");
    }

    let (columns, rows) = grid.dimensions();
    let horizontal_gaps = (columns - 1)
        .checked_mul(VIEW_GAP_PIXELS)
        .ok_or_else(|| anyhow!("native series-grid horizontal gaps overflow"))?;
    let vertical_gaps = (rows - 1)
        .checked_mul(VIEW_GAP_PIXELS)
        .ok_or_else(|| anyhow!("native series-grid vertical gaps overflow"))?;
    let available_width = viewport_area
        .width
        .checked_sub(horizontal_gaps)
        .ok_or_else(|| anyhow!("native series-grid area is narrower than its separators"))?;
    let available_height = viewport_area
        .height
        .checked_sub(vertical_gaps)
        .ok_or_else(|| anyhow!("native series-grid area is shorter than its separators"))?;
    let column_width = available_width / columns;
    let row_height = available_height / rows;
    let extra_columns = available_width % columns;
    let extra_rows = available_height % rows;
    if column_width == 0 || row_height <= PANEL_HEADER_HEIGHT {
        bail!("native surface cannot allocate visible series-grid panels");
    }

    let mut framebuffer = Framebuffer::new(surface_width, surface_height)
        .map_err(|error| anyhow!("allocate native series-grid framebuffer: {error}"))?;
    framebuffer.clear(WORKSPACE_BACKGROUND);
    let header_style = text_style(HEADER_TEXT, 12)?;
    let placeholder_style = text_style(PLACEHOLDER_TEXT, 13)?;
    let annotation_style = text_style(DICOM_ANNOTATION_TEXT, 11)?;
    let annotation_shadow = text_style(DICOM_ANNOTATION_SHADOW, 11)?;
    let mut viewports = ArrayVec::new();

    for (index, panel) in panels.iter().enumerate() {
        let panel_index = u32::try_from(index)
            .map_err(|_| anyhow!("native series-grid panel index exceeds u32"))?;
        let row = panel_index / columns;
        let column = panel_index % columns;
        let panel_x = grid_axis_origin(
            viewport_area.x,
            column,
            column_width,
            extra_columns,
            VIEW_GAP_PIXELS,
        )?;
        let panel_y = grid_axis_origin(
            viewport_area.y,
            row,
            row_height,
            extra_rows,
            VIEW_GAP_PIXELS,
        )?;
        let panel_width = column_width + u32::from(column < extra_columns);
        let panel_height = row_height + u32::from(row < extra_rows);
        let content_height = panel_height
            .checked_sub(PANEL_HEADER_HEIGHT)
            .ok_or_else(|| anyhow!("native series-grid panel is shorter than its header"))?;
        let header = make_rect(panel_x, panel_y, panel_width, PANEL_HEADER_HEIGHT)?;
        fill_rect(&mut framebuffer, header, CornerRadius::SQUARE, PANEL_HEADER);
        draw_text(
            &mut framebuffer,
            header.x.saturating_add(8),
            header.y.saturating_add(7),
            panel.label,
            header_style,
        );
        draw_panel_actions(
            &mut framebuffer,
            panel_x,
            panel_y,
            panel_width,
            panel.maximized,
            panels.len() > 1 || panel.maximized,
        )?;

        let image_y = panel_y
            .checked_add(PANEL_HEADER_HEIGHT)
            .ok_or_else(|| anyhow!("native series-grid image y overflows"))?;
        let view = panel.view.unwrap_or(primary_placeholder);
        let (zoom, pan_offset) = panel.navigation;
        if !zoom.is_finite() || zoom <= 0.0 {
            bail!("native series-grid zoom must be finite and positive");
        }
        let viewport = placement_with_bounds(
            view,
            panel_x,
            image_y,
            panel_width,
            content_height,
            zoom,
            pan_offset,
        )?;
        if let Some(frame) = panel.view {
            blit_frame(
                &mut framebuffer,
                frame,
                viewport,
                surface_width,
                surface_height,
            )?;
            draw_dicom_corner_values(
                &mut framebuffer,
                frame,
                panel_x,
                image_y,
                panel_width,
                content_height,
                &annotation_style,
                &annotation_shadow,
            )?;
        } else {
            let placeholder = make_rect(panel_x, image_y, panel_width, content_height)?;
            fill_rect(
                &mut framebuffer,
                placeholder,
                CornerRadius::SQUARE,
                WORKSPACE_BACKGROUND,
            );
            draw_text(
                &mut framebuffer,
                placeholder.x.saturating_add(10),
                placeholder.y.saturating_add(22),
                "Select a series",
                placeholder_style,
            );
        }

        let panel_bounds = ScreenRect {
            x: f64::from(panel_x),
            y: f64::from(image_y),
            width: f64::from(panel_width),
            height: f64::from(content_height),
        };
        let viewport = NativeViewport {
            panel: panel_bounds,
            panel_x,
            panel_y: image_y,
            panel_width,
            panel_height: content_height,
            ..viewport
        };
        draw_panel_border(
            &mut framebuffer,
            panel_x,
            panel_y,
            panel_width,
            panel_height,
            if active_panel == index {
                ACTIVE_PANEL
            } else {
                INACTIVE_PANEL
            },
        )?;
        viewports
            .try_push(viewport)
            .map_err(|_| anyhow!("native series-grid exceeds its 20-panel limit"))?;
    }

    Ok(SeriesGrid {
        framebuffer,
        viewports,
    })
}

fn grid_axis_origin(
    origin: u32,
    index: u32,
    cell_extent: u32,
    remainder: u32,
    gap: u32,
) -> Result<u32> {
    let cell_offset = index
        .checked_mul(cell_extent)
        .and_then(|offset| offset.checked_add(index.min(remainder)))
        .ok_or_else(|| anyhow!("native series-grid cell offset overflows"))?;
    let gap_offset = index
        .checked_mul(gap)
        .ok_or_else(|| anyhow!("native series-grid separator offset overflows"))?;
    origin
        .checked_add(cell_offset)
        .and_then(|value| value.checked_add(gap_offset))
        .ok_or_else(|| anyhow!("native series-grid cell origin overflows"))
}

fn make_rect(x: u32, y: u32, width: u32, height: u32) -> Result<Rect> {
    Ok(Rect::new(
        i32::try_from(x).map_err(|_| anyhow!("native series-grid x exceeds i32"))?,
        i32::try_from(y).map_err(|_| anyhow!("native series-grid y exceeds i32"))?,
        i32::try_from(width).map_err(|_| anyhow!("native series-grid width exceeds i32"))?,
        i32::try_from(height).map_err(|_| anyhow!("native series-grid height exceeds i32"))?,
    ))
}

fn draw_panel_border(
    framebuffer: &mut Framebuffer,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    color: Color,
) -> Result<()> {
    let x = i32::try_from(x).map_err(|_| anyhow!("series-grid border x exceeds i32"))?;
    let y = i32::try_from(y).map_err(|_| anyhow!("series-grid border y exceeds i32"))?;
    let width =
        i32::try_from(width).map_err(|_| anyhow!("series-grid border width exceeds i32"))?;
    let height =
        i32::try_from(height).map_err(|_| anyhow!("series-grid border height exceeds i32"))?;
    let bottom = y
        .checked_add(height)
        .and_then(|edge| edge.checked_sub(1))
        .ok_or_else(|| anyhow!("series-grid border bottom overflows"))?;
    let right = x
        .checked_add(width)
        .and_then(|edge| edge.checked_sub(1))
        .ok_or_else(|| anyhow!("series-grid border right edge overflows"))?;
    for border in [
        Rect::new(x, y, width, 1),
        Rect::new(x, bottom, width, 1),
        Rect::new(x, y, 1, height),
        Rect::new(right, y, 1, height),
    ] {
        fill_rect(framebuffer, border, CornerRadius::SQUARE, color);
    }
    Ok(())
}

#[cfg(test)]
#[path = "compare/tests.rs"]
mod tests;
