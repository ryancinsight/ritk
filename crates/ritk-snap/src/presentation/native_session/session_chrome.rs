//! Native chrome composition for the viewer session.

use super::layout::MAX_GRID_PANELS;
use super::session::NativeViewerSession;
use anyhow::Result;
use arrayvec::ArrayVec;

impl NativeViewerSession {
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
            &self.window_chrome,
            &self.app,
            &self.views,
            &self.compare_panels,
            self.series_browser.as_ref(),
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
