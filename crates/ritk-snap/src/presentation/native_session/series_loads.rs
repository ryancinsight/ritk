//! Asynchronous multi-series loading for the native session.
//!
//! Study discovery stays synchronous, but volume decoding runs on Moirai
//! blocking workers. Each panel owns its decode through its own [`SnapApp`]
//! load slot, so panels publish independently as their workers finish: images
//! already read can be browsed while the remaining series still decode.
//! [`NativeViewerSession::pump_series_loads`] polls every panel once per
//! frame and maintains the loading progress line; cancellation drops every
//! in-flight task through the existing generation guard, so a late worker
//! can never replace newer state.

use super::compare::ComparePanel;
use super::frame::empty_orthogonal_views;
use super::layout::{PanelGrid, WorkspaceLayout, MAX_COMPARISON_PANELS, MAX_GRID_PANELS};
use super::projection::ProjectionRenderScratch;
use super::session::NativeViewerSession;
use crate::app::volume_input::VolumeInput;
use crate::app::SnapApp;
use crate::render::FrameRenderScratch;
use anyhow::{anyhow, Result};
use arrayvec::ArrayVec;
use std::sync::Arc;

impl NativeViewerSession {
    /// Open every selected series without blocking the UI thread.
    ///
    /// Panels and the grid layout are built immediately; each panel's volume
    /// decodes on a worker and publishes through the per-frame pump. Panels
    /// whose catalog entry is missing still fail fast here, before any panel
    /// is replaced.
    pub(super) fn open_selected_series(&mut self, selected: &[usize]) -> Result<bool> {
        if selected.is_empty() {
            return Ok(false);
        }
        if selected.len() > MAX_GRID_PANELS {
            return Err(anyhow!(
                "selected {} series but the viewer supports at most {MAX_GRID_PANELS} panels",
                selected.len()
            ));
        }
        let browser = self
            .series_browser
            .as_ref()
            .ok_or_else(|| anyhow!("opening multiple series requires a study catalog"))?;
        let mut seen = ArrayVec::<usize, MAX_GRID_PANELS>::new();
        let mut acquisitions = Vec::new();
        acquisitions
            .try_reserve_exact(selected.len())
            .map_err(|_| anyhow!("selected series allocation failed"))?;
        for index in selected.iter().copied() {
            if seen.contains(&index) {
                return Err(anyhow!("series selection contains duplicate index {index}"));
            }
            seen.try_push(index)
                .map_err(|_| anyhow!("series selection exceeds panel capacity"))?;
            let choice = browser
                .choice(index)
                .ok_or_else(|| anyhow!("selected series index {index} is outside the catalog"))?;
            acquisitions.push((
                index,
                Arc::clone(&choice.acquisition),
                choice.modality.to_string(),
                choice.image_count,
            ));
        }

        let mut acquisitions = acquisitions.into_iter();
        let (primary_index, primary_acquisition, _, _) = acquisitions
            .next()
            .ok_or_else(|| anyhow!("series selection lost its primary item"))?;
        let mut primary_app = SnapApp::default();
        primary_app.pending_load = Some(VolumeInput::Series(primary_acquisition));
        primary_app.process_pending_loads();
        self.app = primary_app;
        self.views = empty_orthogonal_views()?;
        self.render_scratch = std::array::from_fn(|_| FrameRenderScratch::default());
        self.projection_scratch = ProjectionRenderScratch::default();
        self.projection = None;

        let mut compare_panels = ArrayVec::<ComparePanel, MAX_COMPARISON_PANELS>::new();
        for (index, acquisition, _, _) in acquisitions {
            let mut panel = ComparePanel::empty()?;
            panel.series_index = Some(index);
            panel.app.pending_load = Some(VolumeInput::Series(acquisition));
            panel.app.process_pending_loads();
            compare_panels
                .try_push(panel)
                .map_err(|_| anyhow!("selected series exceed comparison panel capacity"))?;
        }
        let layout = if selected.len() == 1 {
            WorkspaceLayout::Orthogonal
        } else {
            let grid = PanelGrid::containing_panel(selected.len().saturating_sub(1))
                .ok_or_else(|| anyhow!("selected series exceed the supported panel range"))?;
            while compare_panels.len() < grid.panel_count().saturating_sub(1) {
                compare_panels
                    .try_push(ComparePanel::empty()?)
                    .map_err(|_| anyhow!("selected series exceed comparison panel capacity"))?;
            }
            WorkspaceLayout::Panels(grid)
        };

        self.compare_panels = compare_panels;
        self.workspace_layout = layout;
        self.active_panel = 0;
        self.active_view = None;
        self.primary_series_index = Some(primary_index);
        self.series_browser
            .as_mut()
            .ok_or_else(|| anyhow!("study catalog closed during series selection"))?
            .set_active(primary_index);
        self.series_load_total = selected.len();
        self.series_load_reported = (0, selected.len());
        self.app.status_message = format!("Loading series 0/{}\u{2026}", selected.len());
        Ok(true)
    }

    /// Poll every panel's decode task and maintain the progress line.
    ///
    /// Publication itself happens inside each panel's generation-guarded
    /// task join; this only drives polling, per-panel series messages, and
    /// the aggregate progress state. Panels publish independently, so a
    /// session with a finished first series and pending later series is a
    /// normal observable state, not a transient.
    pub(super) fn pump_series_loads(&mut self) {
        let primary_was_loaded = self.app.loaded.is_some();
        let mut panels_were_loaded = ArrayVec::<bool, MAX_COMPARISON_PANELS>::new();
        for panel in &self.compare_panels {
            panels_were_loaded
                .try_push(panel.app.loaded.is_some())
                .expect("panel count is bounded by panel capacity");
        }
        self.app.poll_load_tasks();
        for panel in &mut self.compare_panels {
            panel.app.poll_load_tasks();
        }
        if !primary_was_loaded
            && self.app.loaded.is_some()
            && let Some(message) = self.primary_series_message()
        {
            self.app.status_message = message;
        }
        for (index, was_loaded) in panels_were_loaded.iter().copied().enumerate() {
            if was_loaded {
                continue;
            }
            if let Some(message) = self.compare_series_message(index)
                && let Some(panel) = self.compare_panels.get_mut(index)
                && panel.app.loaded.is_some()
            {
                panel.app.status_message = message;
            }
        }
        let total = self.series_load_total;
        if total == 0 {
            return;
        }
        let done = self.series_load_done();
        if done < total {
            if self.series_load_reported != (done, total) {
                self.series_load_reported = (done, total);
                self.app.status_message = format!("Loading series {done}/{total}\u{2026}");
            }
        } else {
            self.series_load_total = 0;
            self.series_load_reported = (usize::MAX, usize::MAX);
            if let Some(message) = self.primary_series_message() {
                self.app.status_message = message;
            }
        }
    }

    /// Cancel every in-flight series decode.
    ///
    /// Returns whether any load was active. Late workers are discarded by
    /// the bumped generations, so nothing publishes after this returns.
    pub(super) fn cancel_series_loads(&mut self) -> bool {
        let mut active = self.app.primary_load_active();
        self.app.cancel_load_tasks();
        for panel in &mut self.compare_panels {
            active |= panel.app.primary_load_active();
            panel.app.cancel_load_tasks();
        }
        if active {
            self.series_load_total = 0;
            self.series_load_reported = (usize::MAX, usize::MAX);
            self.app.status_message = "Series loading cancelled.".to_owned();
        }
        active
    }

    /// Join every in-flight decode; test-only deterministic drain.
    ///
    /// Production code never blocks on workers: the per-frame pump observes
    /// partial progress instead. Tests use this to reach a chosen
    /// intermediate state deterministically, because publication only
    /// happens through an explicit poll or join.
    #[cfg(test)]
    pub(super) fn wait_for_series_loads(&mut self) {
        self.app.wait_for_load_tasks();
        for panel in &mut self.compare_panels {
            panel.app.wait_for_load_tasks();
        }
        self.pump_series_loads();
    }

    /// Count panels whose volume has published.
    fn series_load_done(&self) -> usize {
        usize::from(self.app.loaded.is_some())
            + self
                .compare_panels
                .iter()
                .filter(|panel| panel.series_index.is_some() && panel.app.loaded.is_some())
                .count()
    }

    /// Series message for the primary panel, if the catalog still knows it.
    fn primary_series_message(&self) -> Option<String> {
        let browser = self.series_browser.as_ref()?;
        let index = self.primary_series_index?;
        let choice = browser.choice(index)?;
        Some(format!(
            "Loaded {} series ({} images).",
            choice.modality, choice.image_count
        ))
    }

    /// Series message for one comparison panel, if it shows catalog content.
    fn compare_series_message(&self, panel: usize) -> Option<String> {
        let browser = self.series_browser.as_ref()?;
        let index = self.compare_panels.get(panel)?.series_index?;
        let choice = browser.choice(index)?;
        Some(format!(
            "Loaded {} series ({} images).",
            choice.modality, choice.image_count
        ))
    }
}
