//! Native chrome composition for the viewer session.

use super::layout::overlay_display::{
    append_colorbar_overlay, append_histogram_overlay, histogram_bounds,
};
use super::layout::{
    append_measurement_overlay, application_overlay, crosshair_overlay, OverlayCommands,
    MAX_GRID_PANELS,
};
use super::session::{clinical_overlay, volume_affine, NativeViewerSession};
use anyhow::{anyhow, Result};
use arrayvec::ArrayVec;
use metis_platform::Rect;
use metis_ui_lang::PlacementObstacle;

impl NativeViewerSession {
    pub(super) fn render_pane_overlays(&mut self) -> Result<()> {
        let frame = Rect::new(
            0,
            0,
            i32::try_from(self.framebuffer.width())?,
            i32::try_from(self.framebuffer.height())?,
        );
        let mut reserved = if !self.workspace_layout.is_grid()
            && (self.app.show_overlay || self.capture_application)
        {
            let clinical = clinical_overlay(&self.app, self.anonymize_overlay);
            application_overlay(
                &self.views,
                &self.viewports,
                self.app.cine.enabled,
                self.app.cine.fps,
                clinical.as_ref(),
            )?
            .placement_obstacles(frame, &[])?
        } else {
            Vec::new()
        };
        reserved
            .try_reserve(1)
            .map_err(|_| anyhow!("native pane obstacle allocation failed"))?;
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
            if app.loaded.is_none() || viewport.panel_width == 0 || viewport.panel_height == 0 {
                continue;
            }
            let mut overlay = OverlayCommands::with_capacity(384 + app.annotations.len())?;
            let (center, width) = app.browser_window_level_values();
            append_colorbar_overlay(&mut overlay, viewport, app.colormap, center, width)?;
            let histogram = if index == self.active_panel {
                app.cached_histogram.as_ref()
            } else {
                None
            };
            let reserved_len = reserved.len();
            if let Some(bounds) = histogram
                .map(|histogram| histogram_bounds(viewport, histogram))
                .transpose()?
                .flatten()
            {
                reserved.push(PlacementObstacle::Rect(bounds));
            }
            let preview = matches!(
                app.tool_state,
                crate::tools::interaction::ToolState::LengthDrag { .. }
                    | crate::tools::interaction::ToolState::ArrowDrag { .. }
                    | crate::tools::interaction::ToolState::PencilDrag { .. }
                    | crate::tools::interaction::ToolState::CobbDrag { .. }
                    | crate::tools::interaction::ToolState::CobbPending { .. }
            );
            append_measurement_overlay(
                &mut overlay,
                std::slice::from_ref(view),
                std::slice::from_ref(viewport),
                &app.annotations,
                app.pointer.selected_annotation,
                app.pointer
                    .pending_annotation_move
                    .as_ref()
                    .map(|pending| (pending.annotation_index, &pending.candidate)),
                volume_affine(app),
                |location| app.length_scale(location),
                preview.then_some((
                    &app.tool_state,
                    app.cobb_location()
                        .or(app.pointer.drag_measurement_location),
                )),
                frame,
                &reserved,
            )?;
            reserved.truncate(reserved_len);
            if let Some(histogram) = histogram {
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
