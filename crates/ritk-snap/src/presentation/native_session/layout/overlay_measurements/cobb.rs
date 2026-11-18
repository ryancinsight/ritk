use super::{image_point, OverlayCommands, Pane, ToolState, MEASUREMENT_COLOR};
use anyhow::Result;

impl Pane<'_> {
    pub(super) fn cobb(
        &self,
        overlay: &mut OverlayCommands,
        [p1, p2, p3, p4]: [[f32; 2]; 4],
        angle_deg: f32,
    ) -> Result<()> {
        self.line(overlay, image_point(p1), image_point(p2), MEASUREMENT_COLOR)?;
        self.line(overlay, image_point(p3), image_point(p4), MEASUREMENT_COLOR)?;
        self.label(overlay, image_point(p2), format!("{angle_deg:.1} deg"))
    }

    pub(super) fn cobb_preview(
        &self,
        overlay: &mut OverlayCommands,
        state: &ToolState,
    ) -> Result<()> {
        match state {
            ToolState::CobbDrag {
                first,
                start,
                current,
            } => {
                if let Some(first) = first {
                    self.segment(overlay, *first, MEASUREMENT_COLOR)?;
                }
                self.segment(overlay, [*start, *current], MEASUREMENT_COLOR)?;
            }
            ToolState::CobbPending { segment } => {
                self.segment(overlay, *segment, MEASUREMENT_COLOR)?;
            }
            _ => {}
        }
        Ok(())
    }
}
