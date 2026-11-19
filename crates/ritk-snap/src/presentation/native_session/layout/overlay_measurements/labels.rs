use super::{
    screen_coordinate, text_style, Pane, LABEL_OFFSET, LENGTH_LABEL_OFFSET, MEASUREMENT_COLOR,
    PANEL_DETAIL_SIZE,
};
use crate::presentation::native_session::layout::OverlayCommands;
use anyhow::Result;
use metis_platform::Rect;
use metis_ui_lang::PlacementObstacle;

impl Pane<'_> {
    pub(super) fn label(
        &self,
        labels: &mut Vec<(String, (i32, i32))>,
        anchor: [f64; 2],
        text: String,
    ) -> Result<()> {
        let (x, y) = self.point(anchor)?;
        labels.push((
            text,
            (
                x.saturating_add(LABEL_OFFSET),
                y.saturating_add(LABEL_OFFSET),
            ),
        ));
        Ok(())
    }

    pub(super) fn length_label(
        &self,
        labels: &mut Vec<(String, (i32, i32))>,
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
        let x = screen_coordinate((x0 + x1) * 0.5 + ox, "length label x")?;
        let y = screen_coordinate((y0 + y1) * 0.5 + oy, "length label y")?;
        labels.push((text, (x, y)));
        Ok(())
    }

    pub(super) fn draw_label(
        &self,
        overlay: &mut OverlayCommands,
        text: String,
        origin: (i32, i32),
        frame: Rect,
        reserved: &[PlacementObstacle],
    ) -> Result<()> {
        let panel = self.viewport.panel_rect()?;
        let bounds = OverlayCommands::measurement_label_bounds(panel, LABEL_OFFSET);
        overlay.push_text_inside(
            text,
            text_style(MEASUREMENT_COLOR, PANEL_DETAIL_SIZE)?,
            origin,
            bounds,
            frame,
            reserved,
        )
    }
}
