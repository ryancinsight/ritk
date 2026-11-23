//! The active acquisition supplies the plane for every inactive comparison pane.
use super::{layout::reference_segment, NativeViewerSession};
use anyhow::Result;
use metis_platform::Color;
use metis_ui_lang::DisplayList;

impl NativeViewerSession {
    pub(super) fn render_reference_lines(&mut self) -> Result<()> {
        if !self.reference_lines {
            return Ok(());
        }
        let source = self.active_app();
        let Some(stack) = &source.patient_stack else {
            return Ok(());
        };
        let index = source.axis_slice_info(0).0;
        let mut overlay = DisplayList::default();
        for (i, viewport) in self.viewports.iter().enumerate() {
            if i == self.active_panel {
                continue;
            }
            let (app, view) = if i == 0 {
                (&self.app, &self.views[0])
            } else {
                let panel = &self.compare_panels[i - 1];
                (&panel.app, panel.axial_view())
            };
            let Some(trace) = app
                .patient_stack
                .as_ref()
                .and_then(|target| stack.trace_on(index, target, app.axis_slice_info(0).0))
            else {
                continue;
            };
            if let Some([start, end]) = reference_segment(&trace, view, *viewport)? {
                overlay.append_line(start, end, Color::rgba(255, 255, 0, 255))?;
            }
        }
        overlay.render_to(&mut self.framebuffer);
        Ok(())
    }
}
