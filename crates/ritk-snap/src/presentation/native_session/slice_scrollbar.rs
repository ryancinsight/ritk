//! Per-pane slice navigation using Métis scrollbar geometry.

use anyhow::{anyhow, Result};
use metis_platform::rasterizer::{fill_rect, CornerRadius};
use metis_platform::{Color, Rect};
use metis_ui_lang::{ScrollbarAxis, ScrollbarGeometry};

use super::layout::NativeViewport;
use super::NativeViewerSession;

const TRACK_WIDTH: i32 = 8;
const EDGE_INSET: i32 = 2;
pub(super) const RIGHT_GUTTER: i32 = TRACK_WIDTH + EDGE_INSET;
const VERTICAL_INSET: i32 = 28;
const TRACK_COLOR: Color = Color::rgb(47, 55, 64);
const THUMB_COLOR: Color = Color::rgb(138, 152, 168);

#[derive(Debug, Clone, Copy)]
pub(super) struct SliceScrollbar {
    pub(super) geometry: ScrollbarGeometry,
    pub(super) offset: u64,
}

impl NativeViewerSession {
    pub(super) fn render_slice_scrollbars(&mut self) -> Result<()> {
        for (index, viewport) in self.viewports.iter_mut().enumerate() {
            let app = if self.workspace_layout.is_grid() && index > 0 {
                &self
                    .compare_panels
                    .get(index - 1)
                    .ok_or_else(|| anyhow!("series panel state is missing"))?
                    .app
            } else {
                &self.app
            };
            let (slice, total) = app.axis_slice_info(viewport.axis());
            viewport.scrollbar = geometry(*viewport, slice, total)?;
            if let Some(scrollbar) = viewport.scrollbar {
                for (rect, color) in [
                    (scrollbar.geometry.track(), TRACK_COLOR),
                    (scrollbar.geometry.thumb(), THUMB_COLOR),
                ] {
                    fill_rect(&mut self.framebuffer, rect, CornerRadius::SQUARE, color);
                }
            }
        }
        Ok(())
    }
}

fn geometry(
    viewport: NativeViewport,
    slice: usize,
    total: usize,
) -> Result<Option<SliceScrollbar>> {
    let width = i32::try_from(viewport.panel_width)?;
    let height = i32::try_from(viewport.panel_height)?;
    if width < 3 * TRACK_WIDTH || height <= 2 * VERTICAL_INSET {
        return Ok(None);
    }
    let track = Rect::new(
        i32::try_from(viewport.panel_x)? + width - RIGHT_GUTTER,
        i32::try_from(viewport.panel_y)? + VERTICAL_INSET,
        TRACK_WIDTH,
        height - 2 * VERTICAL_INSET,
    );
    let offset = u64::try_from(slice)?;
    Ok(ScrollbarGeometry::new(
        ScrollbarAxis::Vertical,
        track,
        u64::try_from(total)?,
        1,
        offset,
    )
    .map(|geometry| SliceScrollbar { geometry, offset }))
}
