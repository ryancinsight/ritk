//! Study opening and multi-series workspace transitions.

use super::compare::ComparePanel;
use super::layout::{PanelGrid, WorkspaceLayout, MAX_COMPARISON_PANELS, MAX_GRID_PANELS};
use super::series_browser::SeriesBrowser;
use super::session::NativeViewerSession;
use super::{frame, projection};
use crate::app::SnapApp;
use crate::dicom::loader::{
    load_volume_from_path, load_volume_from_series_info, scan_folder_for_series,
};
use crate::render::FrameRenderScratch;
use anyhow::{anyhow, Context, Result};
use arrayvec::ArrayVec;
use metis_platform::native::{pick, DialogSelection};
use std::path::Path;
use std::sync::Arc;

impl NativeViewerSession {
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
        let loaded = {
            let browser = self
                .series_browser
                .as_ref()
                .ok_or_else(|| anyhow!("opening multiple series requires a study catalog"))?;
            let mut seen = ArrayVec::<usize, MAX_GRID_PANELS>::new();
            let mut loaded = Vec::new();
            loaded
                .try_reserve_exact(selected.len())
                .map_err(|_| anyhow!("selected series allocation failed"))?;
            for index in selected.iter().copied() {
                if seen.contains(&index) {
                    return Err(anyhow!("series selection contains duplicate index {index}"));
                }
                seen.try_push(index)
                    .map_err(|_| anyhow!("series selection exceeds panel capacity"))?;
                let choice = browser.choice(index).ok_or_else(|| {
                    anyhow!("selected series index {index} is outside the catalog")
                })?;
                let status = format!(
                    "Loaded {} series ({} images).",
                    choice.modality, choice.image_count
                );
                let volume = load_volume_from_series_info(&choice.acquisition)
                    .with_context(|| format!("open selected series {index}"))?;
                loaded.push((index, status, volume));
            }
            loaded
        };

        let mut loaded = loaded.into_iter();
        let (primary_index, primary_status, primary_volume) = loaded
            .next()
            .ok_or_else(|| anyhow!("series selection lost its primary item"))?;
        let mut primary_app = SnapApp::default();
        primary_app.load_volume(primary_volume, primary_status);
        let mut render_scratch = std::array::from_fn(|_| FrameRenderScratch::default());
        let views = frame::render_orthogonal_views(&primary_app, &mut render_scratch)?;
        let mut projection_scratch = projection::ProjectionRenderScratch::default();
        let projection = match (
            selected.len() == 1,
            self.presentation_mode.projection_statistic(),
        ) {
            (false, _) | (_, None) => None,
            (true, Some(statistic)) => {
                let mut rendered = projection::empty_projection(statistic)?;
                projection::render_projection_into(
                    &primary_app,
                    statistic,
                    &mut rendered,
                    &mut projection_scratch,
                )?;
                Some(rendered)
            }
        };

        let mut compare_panels = ArrayVec::<ComparePanel, MAX_COMPARISON_PANELS>::new();
        for (index, status, volume) in loaded {
            let mut panel = ComparePanel::empty()?;
            panel.replace(volume, index, status)?;
            compare_panels
                .try_push(panel)
                .map_err(|_| anyhow!("selected series exceed comparison panel capacity"))?;
        }
        let layout = if selected.len() == 1 {
            WorkspaceLayout::Orthogonal
        } else {
            let grid = PanelGrid::containing_panel(selected.len().saturating_sub(1))
                .ok_or_else(|| anyhow!("selected series exceed supported panel layouts"))?;
            while compare_panels.len() < grid.panel_count().saturating_sub(1) {
                compare_panels
                    .try_push(ComparePanel::empty()?)
                    .map_err(|_| anyhow!("selected series exceed comparison panel capacity"))?;
            }
            WorkspaceLayout::Panels(grid)
        };

        self.app = primary_app;
        self.views = views;
        self.render_scratch = render_scratch;
        self.projection_scratch = projection_scratch;
        self.projection = projection;
        self.compare_panels = compare_panels;
        self.workspace_layout = layout;
        self.active_panel = 0;
        self.active_view = None;
        self.primary_series_index = Some(primary_index);
        self.series_browser
            .as_mut()
            .ok_or_else(|| anyhow!("study catalog closed during series selection"))?
            .set_active(primary_index);
        Ok(true)
    }

    pub(super) fn assign_series_to_next_panel(&mut self, index: usize) -> Result<bool> {
        let browser = self
            .series_browser
            .as_ref()
            .ok_or_else(|| anyhow!("series assignment requires a study catalog"))?;
        if browser.choice(index).is_none() {
            return Err(anyhow!("selected series row is outside the study"));
        }
        self.restore_maximized_panel()?;
        let target = self
            .compare_panels
            .iter()
            .position(|panel| panel.series_index.is_none())
            .map_or_else(
                || self.compare_panels.len().saturating_add(1),
                |index| index + 1,
            );
        if target >= MAX_GRID_PANELS {
            return Err(anyhow!(
                "the viewer has reached its {MAX_GRID_PANELS}-panel limit"
            ));
        }
        if self
            .workspace_layout
            .grid()
            .is_none_or(|grid| grid.panel_count() <= target)
        {
            let grid = PanelGrid::containing_panel(target)
                .ok_or_else(|| anyhow!("selected series exceeds the supported panel range"))?;
            self.set_workspace_layout(WorkspaceLayout::Panels(grid))?;
        }
        self.assign_series_to_panel(index, target)
    }

    pub(super) fn open_study_path(&mut self, path: &Path) -> Result<()> {
        if path.is_dir() {
            let tree = scan_folder_for_series(path).context("discover selected RITK study")?;
            if tree.total_series() == 0 {
                let volume = load_volume_from_path(path).context("open selected RITK study")?;
                self.app
                    .load_volume(volume, "Loaded DICOM study.".to_owned());
                self.series_browser = None;
                self.primary_series_index = None;
                self.reset_comparison();
                return Ok(());
            }

            let browser = SeriesBrowser::from_tree(&tree, None)?;
            let choice = browser
                .choice(browser.active_index())
                .expect("invariant: a non-empty study browser has an active series");
            let modality = choice.modality.to_string();
            let image_count = choice.image_count;
            let volume = match load_volume_from_series_info(&choice.acquisition) {
                Ok(volume) => volume,
                Err(error) => {
                    self.series_browser = Some(browser);
                    self.primary_series_index = None;
                    self.reset_comparison();
                    tracing::warn!(
                        error_chain_depth = error.chain().count(),
                        "first discovered series could not be opened; study catalog remains available"
                    );
                    self.app.status_message =
                        "First series could not be opened; select another series from the preview bar."
                            .to_owned();
                    return Ok(());
                }
            };
            self.app.load_volume(
                volume,
                format!("Loaded {modality} series ({image_count} images)."),
            );
            self.primary_series_index = Some(browser.active_index());
            self.series_browser = Some(browser);
            self.reset_comparison();
        } else {
            let volume = load_volume_from_path(path).context("open selected RITK study")?;
            self.app
                .load_volume(volume, "Loaded DICOM study.".to_owned());
            self.series_browser = None;
            self.primary_series_index = None;
            self.reset_comparison();
        }
        Ok(())
    }

    pub(super) fn select_series(&mut self, index: usize) -> Result<bool> {
        let Some(browser) = self.series_browser.as_ref() else {
            return Ok(false);
        };
        if browser.choice(index).is_none() {
            return Err(anyhow!("selected series row is outside the study"));
        }
        if self.maximized_panel.is_some() && self.primary_series_index != Some(index) {
            self.restore_maximized_panel()?;
        }
        let browser = self
            .series_browser
            .as_ref()
            .ok_or_else(|| anyhow!("selected series lost its study catalog"))?;
        let browser_active_index = browser.active_index();
        if self.primary_series_index == Some(index) {
            let changed = self.active_panel != 0 || browser_active_index != index;
            self.active_panel = 0;
            self.series_browser
                .as_mut()
                .expect("invariant: selected series retains its study browser")
                .set_active(index);
            return Ok(changed);
        }
        if let Some(panel_index) = self
            .compare_panels
            .iter()
            .position(|panel| panel.series_index == Some(index))
            .map(|index| index.saturating_add(1))
        {
            let panel_is_visible = self
                .workspace_layout
                .grid()
                .is_some_and(|grid| panel_index < grid.panel_count());
            let layout_changed = if panel_is_visible {
                false
            } else {
                let grid = PanelGrid::containing_panel(panel_index)
                    .ok_or_else(|| anyhow!("assigned series exceeds the supported panel range"))?;
                self.set_workspace_layout(WorkspaceLayout::Panels(grid))?
            };
            let changed =
                layout_changed || self.active_panel != panel_index || browser_active_index != index;
            self.active_panel = panel_index;
            self.series_browser
                .as_mut()
                .expect("invariant: selected series retains its study browser")
                .set_active(index);
            return Ok(changed);
        }
        self.assign_series_to_panel(index, self.active_panel)
    }

    pub(super) fn browse_series(&mut self, index: usize, panel_index: usize) -> Result<bool> {
        let Some(browser) = self.series_browser.as_ref() else {
            return Ok(false);
        };
        if browser.choice(index).is_none() {
            return Err(anyhow!("selected series row is outside the study"));
        }
        let layout = self
            .maximized_panel
            .map_or(self.workspace_layout, |maximized| maximized.layout);
        let panel_count = layout.grid().map_or(1, PanelGrid::panel_count);
        if panel_index >= panel_count {
            return Err(anyhow!("browsed series target is outside the study layout"));
        }
        let hidden_panel = self.maximized_panel.is_some() && panel_index != 0;
        let current_series = if panel_index == 0 {
            self.primary_series_index
        } else {
            self.compare_panels
                .get(panel_index - 1)
                .and_then(|panel| panel.series_index)
        };
        if current_series == Some(index) {
            if hidden_panel {
                return Ok(false);
            }
            let changed = self.active_panel != panel_index
                || self
                    .series_browser
                    .as_ref()
                    .is_some_and(|browser| browser.active_index() != index);
            self.active_panel = panel_index;
            self.series_browser
                .as_mut()
                .expect("invariant: browsed series retains its study browser")
                .set_active(index);
            return Ok(changed);
        }

        self.replace_series_at(index, panel_index)?;
        if !hidden_panel {
            self.active_panel = panel_index;
            self.series_browser
                .as_mut()
                .expect("invariant: browsed series retains its study browser")
                .set_active(index);
        }
        Ok(true)
    }

    pub(super) fn assign_series_to_panel(
        &mut self,
        index: usize,
        panel_index: usize,
    ) -> Result<bool> {
        let browser = self
            .series_browser
            .as_ref()
            .ok_or_else(|| anyhow!("series assignment requires a study catalog"))?;
        if browser.choice(index).is_none() {
            return Err(anyhow!("selected series row is outside the study"));
        }
        let panel_count = self
            .workspace_layout
            .grid()
            .map_or(1, |grid| grid.panel_count());
        if panel_index >= panel_count {
            return Err(anyhow!(
                "series assignment target is outside the active layout"
            ));
        }
        let previous_panel = self.active_panel;
        let previous_series = if panel_index == 0 {
            self.primary_series_index
        } else {
            self.compare_panels
                .get(panel_index - 1)
                .ok_or_else(|| anyhow!("series assignment panel is not initialized"))?
                .series_index
        };
        let changed = previous_panel != panel_index || previous_series != Some(index);
        if previous_series == Some(index) {
            self.active_panel = panel_index;
            self.series_browser
                .as_mut()
                .expect("invariant: selected series retains its study browser")
                .set_active(index);
            return Ok(changed);
        }

        self.replace_series_at(index, panel_index)?;
        self.active_panel = panel_index;
        self.series_browser
            .as_mut()
            .expect("invariant: selected series retains its study browser")
            .set_active(index);
        Ok(true)
    }

    fn replace_series_at(&mut self, index: usize, panel_index: usize) -> Result<()> {
        let (acquisition, modality, image_count) = {
            let browser = self
                .series_browser
                .as_ref()
                .ok_or_else(|| anyhow!("series replacement requires a study catalog"))?;
            let choice = browser
                .choice(index)
                .ok_or_else(|| anyhow!("selected series row is outside the study"))?;
            (
                Arc::clone(&choice.acquisition),
                choice.modality.to_string(),
                choice.image_count,
            )
        };
        let volume = if self.primary_series_index == Some(index) {
            self.app
                .loaded
                .as_ref()
                .ok_or_else(|| anyhow!("primary series {index} has no decoded volume"))?
                .clone()
        } else if let Some(panel) = self
            .compare_panels
            .iter()
            .find(|panel| panel.series_index == Some(index))
        {
            panel
                .app
                .loaded
                .as_ref()
                .ok_or_else(|| anyhow!("comparison series {index} has no decoded volume"))?
                .clone()
        } else {
            load_volume_from_series_info(&acquisition)
                .with_context(|| "open the selected DICOM series")?
        };
        let status = format!("Loaded {modality} series ({image_count} images).");
        if panel_index == 0 {
            self.app.load_volume(volume, status);
            self.primary_series_index = Some(index);
        } else {
            self.compare_panels
                .get_mut(panel_index - 1)
                .ok_or_else(|| anyhow!("series replacement panel is not initialized"))?
                .replace(volume, index, status)?;
        }
        Ok(())
    }

    pub(super) fn open_study_from_dialog(&mut self) -> Result<bool> {
        let selected = pick(DialogSelection::Folder).context("show native study picker")?;
        let Some(path) = selected else {
            return Ok(false);
        };
        self.open_study_path(&path)?;
        Ok(true)
    }
}
