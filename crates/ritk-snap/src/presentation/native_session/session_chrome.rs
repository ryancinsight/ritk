//! Native chrome composition for the viewer session.

use super::layout::overlay_display::{append_colorbar_overlay, append_histogram_overlay};
use super::layout::{
    append_measurement_overlay, append_measurement_preview, crosshair_overlay, OverlayCommands,
    MAX_GRID_PANELS,
};
use super::session::{volume_affine, NativeViewerSession};
use anyhow::{anyhow, Result};
use arrayvec::ArrayVec;

impl NativeViewerSession {
    pub(super) fn render_measurements(&mut self) -> Result<()> {
        for (index, viewport) in self.viewports.iter().enumerate() {
            let (app, view) = if self.workspace_layout.is_grid() && index > 0 {
                let panel = self
                    .compare_panels
                    .get(index - 1)
                    .ok_or_else(|| anyhow!("series panel state is missing"))?;
                (&panel.app, panel.axial_view())
            } else {
                let view = self
                    .views
                    .get(index)
                    .ok_or_else(|| anyhow!("native viewport has no slice frame"))?;
                (&self.app, view)
            };
            let preview = matches!(
                app.tool_state,
                crate::tools::interaction::ToolState::CobbDrag { .. }
                    | crate::tools::interaction::ToolState::CobbPending { .. }
            );
            if (app.annotations.is_empty() && !preview)
                || viewport.panel_width == 0
                || viewport.panel_height == 0
            {
                continue;
            }
            let mut overlay = OverlayCommands::with_capacity(app.annotations.len())?;
            append_measurement_overlay(
                &mut overlay,
                std::slice::from_ref(view),
                std::slice::from_ref(viewport),
                &app.annotations,
                volume_affine(app),
            )?;
            if preview {
                append_measurement_preview(
                    &mut overlay,
                    view,
                    viewport,
                    &app.tool_state,
                    app.pointer
                        .drag_measurement_location
                        .or(app.pointer.pending_measurement_location),
                )?;
            }
            self.framebuffer
                .render_clipped(viewport.panel_rect()?, |framebuffer| {
                    overlay.into_display_list().render_to(framebuffer);
                });
        }
        Ok(())
    }

    /// Renders the Display menu colorbars (every pane) and the histogram
    /// window panel (active pane) once a study is loaded.
    pub(super) fn render_display_overlays(&mut self) -> Result<()> {
        for (index, viewport) in self.viewports.iter().enumerate() {
            let app = if self.workspace_layout.is_grid() && index > 0 {
                &self
                    .compare_panels
                    .get(index - 1)
                    .ok_or_else(|| anyhow!("series panel state is missing"))?
                    .app
            } else {
                &self.app
            };
            if app.loaded.is_none() || viewport.panel_width == 0 || viewport.panel_height == 0 {
                continue;
            }
            let mut overlay = OverlayCommands::with_capacity(384)?;
            let (center, width) = app.browser_window_level_values();
            append_colorbar_overlay(&mut overlay, viewport, app.colormap, center, width)?;
            if index == self.active_panel
                && let Some(histogram) = &app.cached_histogram
            {
                append_histogram_overlay(&mut overlay, viewport, histogram, center, width)?;
            }
            self.framebuffer
                .render_clipped(viewport.panel_rect()?, |framebuffer| {
                    overlay.into_display_list().render_to(framebuffer);
                });
        }
        Ok(())
    }

    pub(super) fn render_crosshair_overlay(&mut self) -> Result<()> {
        if self.workspace_layout.is_grid() {
            return Ok(());
        }
        let shape = self.app.loaded.as_ref().map(|volume| volume.shape);
        let cursor = self.app.linked_cursor.map(|cursor| cursor.voxel());
        crosshair_overlay(
            &self.views,
            self.viewports.as_slice(),
            shape,
            cursor,
            self.app.show_crosshair,
        )?
        .render_to(&mut self.framebuffer);
        Ok(())
    }

    pub(super) fn render_chrome(&mut self) -> Result<()> {
        let (
            framebuffer,
            window_chrome,
            primary_app,
            views,
            compare_panels,
            series_browser,
            workspace_layout,
            active_panel,
            primary_series_index,
        ) = (
            &mut self.framebuffer,
            &mut self.window_chrome,
            &self.app,
            &self.views,
            &self.compare_panels,
            self.series_browser.as_mut(),
            self.workspace_layout,
            self.active_panel,
            self.primary_series_index,
        );
        let app = if workspace_layout.is_grid() && active_panel > 0 {
            compare_panels
                .get(active_panel - 1)
                .map_or(primary_app, |panel| &panel.app)
        } else {
            primary_app
        };
        let visible_panels = workspace_layout.grid().map_or(1, |grid| grid.panel_count());
        let mut series_previews: ArrayVec<
            Option<&crate::presentation::PresentationFrame>,
            MAX_GRID_PANELS,
        > = ArrayVec::new();
        let mut displayed_series: ArrayVec<Option<usize>, MAX_GRID_PANELS> = ArrayVec::new();
        series_previews
            .try_push(Some(views[0].frame()))
            .map_err(|_| anyhow::anyhow!("native series preview capacity exceeded"))?;
        displayed_series
            .try_push(primary_series_index)
            .map_err(|_| anyhow::anyhow!("native displayed-series capacity exceeded"))?;
        for panel in compare_panels.iter().take(visible_panels.saturating_sub(1)) {
            series_previews
                .try_push(panel.series_index.map(|_| panel.axial_view().frame()))
                .map_err(|_| anyhow::anyhow!("native series preview capacity exceeded"))?;
            displayed_series
                .try_push(panel.series_index)
                .map_err(|_| anyhow::anyhow!("native displayed-series capacity exceeded"))?;
        }
        window_chrome.render(
            framebuffer,
            app,
            series_previews.as_slice(),
            series_browser,
            workspace_layout,
            active_panel,
            displayed_series.as_slice(),
        )
    }
}
