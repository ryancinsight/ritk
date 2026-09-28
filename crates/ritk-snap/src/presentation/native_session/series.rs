//! Study opening and multi-series workspace transitions.

use super::layout::{PanelGrid, WorkspaceLayout};
use super::series_browser::SeriesBrowser;
use super::session::NativeViewerSession;
use crate::dicom::loader::{
    load_volume_from_path, load_volume_from_series_info, scan_folder_for_series,
};
use anyhow::{anyhow, Context, Result};
use metis_platform::native::{pick, DialogSelection};
use std::path::Path;
use std::sync::Arc;

impl NativeViewerSession {
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
            let instance_count = choice.instance_count;
            let volume = load_volume_from_series_info(&choice.acquisition)
                .with_context(|| "open the first discovered DICOM series")?;
            self.app.load_volume(
                volume,
                format!("Loaded {modality} series ({instance_count} instances)."),
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

    pub(super) fn assign_series_to_panel(
        &mut self,
        index: usize,
        panel_index: usize,
    ) -> Result<bool> {
        let browser = self
            .series_browser
            .as_ref()
            .ok_or_else(|| anyhow!("series assignment requires a study catalog"))?;
        let choice = browser
            .choice(index)
            .ok_or_else(|| anyhow!("selected series row is outside the study"))?;
        let (acquisition, modality, instance_count) = (
            Arc::clone(&choice.acquisition),
            choice.modality.to_string(),
            choice.instance_count,
        );
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
        self.active_panel = panel_index;
        let changed = previous_panel != panel_index || previous_series != Some(index);
        if previous_series == Some(index) {
            self.series_browser
                .as_mut()
                .expect("invariant: selected series retains its study browser")
                .set_active(index);
            return Ok(changed);
        }

        let volume = load_volume_from_series_info(&acquisition)
            .with_context(|| "open the selected DICOM series")?;
        let status = format!("Loaded {modality} series ({instance_count} instances).");
        if panel_index == 0 {
            self.app.load_volume(volume, status);
            self.primary_series_index = Some(index);
        } else {
            self.compare_panels
                .get_mut(panel_index - 1)
                .ok_or_else(|| anyhow!("series assignment panel is not initialized"))?
                .replace(volume, index, status)?;
        }
        self.series_browser
            .as_mut()
            .expect("invariant: selected series retains its study browser")
            .set_active(index);
        Ok(true)
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
