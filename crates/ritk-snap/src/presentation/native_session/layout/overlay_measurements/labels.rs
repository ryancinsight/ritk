use super::super::super::slice_scrollbar::RIGHT_GUTTER;
use super::super::overlay::OVERLAY_BAR_HEIGHT;
use super::super::overlay_display::COLORBAR_WIDTH;
use super::super::text::{text_command, text_style, PANEL_DETAIL_SIZE};
use super::{screen_coordinate, OverlayCommands, Pane, MEASUREMENT_COLOR};
use anyhow::Result;

const LABEL_OFFSET: i32 = 8;
const LENGTH_LABEL_OFFSET: f64 = 10.0;

impl Pane<'_> {
    pub(super) fn label(
        &self,
        overlay: &mut OverlayCommands,
        anchor: [f64; 2],
        text: String,
    ) -> Result<()> {
        let (x, y) = self.point(anchor)?;
        self.draw_label(
            overlay,
            x.saturating_add(LABEL_OFFSET),
            y.saturating_add(LABEL_OFFSET),
            text,
        )
    }

    pub(super) fn length_label(
        &self,
        overlay: &mut OverlayCommands,
        a: [f64; 2],
        b: [f64; 2],
        text: String,
    ) -> Result<()> {
        let (start, end) = (self.point(a)?, self.point(b)?);
        let (x0, y0) = (f64::from(start.0), f64::from(start.1));
        let (x1, y1) = (f64::from(end.0), f64::from(end.1));
        let (dx, dy) = (x1 - x0, y1 - y0);
        let norm = dx.hypot(dy);
        let (ox, oy) = if norm == 0.0 {
            (0.0, -LENGTH_LABEL_OFFSET)
        } else {
            (
                -dy / norm * LENGTH_LABEL_OFFSET,
                dx / norm * LENGTH_LABEL_OFFSET,
            )
        };
        self.draw_label(
            overlay,
            screen_coordinate((x0 + x1) * 0.5 + ox, "length label x")?,
            screen_coordinate((y0 + y1) * 0.5 + oy, "length label y")?,
            text,
        )
    }

    fn draw_label(
        &self,
        overlay: &mut OverlayCommands,
        x: i32,
        y: i32,
        text: String,
    ) -> Result<()> {
        let panel = self.viewport.panel_rect()?;
        let left = panel.x + LABEL_OFFSET;
        let right = panel.x + panel.width - RIGHT_GUTTER - COLORBAR_WIDTH - LABEL_OFFSET;
        let top = panel.y + OVERLAY_BAR_HEIGHT;
        let bottom = panel.y + panel.height - OVERLAY_BAR_HEIGHT;
        let style = text_style(MEASUREMENT_COLOR, PANEL_DETAIL_SIZE)?;
        if let Some(glyph) = style.extent(0, 0, &text)
            && glyph.width <= right - left
            && glyph.height <= bottom - top
        {
            let x = x.saturating_add(glyph.x).clamp(left, right - glyph.width) - glyph.x;
            let y = y.saturating_add(glyph.y).clamp(top, bottom - glyph.height) - glyph.y;
            overlay.push(text_command(text, x, y, style));
        }
        Ok(())
    }
}
