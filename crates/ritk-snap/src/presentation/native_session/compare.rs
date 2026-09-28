//! Independent state for additional series in the native grid layout.

use super::frame::{empty_axial_view, render_axial_view_into, RenderedView};
use super::layout::{PanelGrid, WorkspaceLayout};
use super::session::NativeViewerSession;
use crate::app::SnapApp;
use crate::dicom::loader::load_volume_from_series_info;
use crate::render::FrameRenderScratch;
use crate::LoadedVolume;
use anyhow::{anyhow, Context, Result};
use arrayvec::ArrayString;
use std::fmt::Write as _;

pub(super) struct ComparePanel {
    pub(super) app: SnapApp,
    pub(super) view: RenderedView,
    scratch: FrameRenderScratch,
    pub(super) series_index: Option<usize>,
}

impl ComparePanel {
    pub(super) fn empty() -> Result<Self> {
        Ok(Self {
            app: SnapApp::default(),
            view: empty_axial_view()?,
            scratch: FrameRenderScratch::default(),
            series_index: None,
        })
    }

    pub(super) fn replace(
        &mut self,
        volume: LoadedVolume,
        series_index: usize,
        status: String,
    ) -> Result<()> {
        let mut app = SnapApp::default();
        app.load_volume(volume, status);
        let mut view = empty_axial_view()?;
        let mut scratch = FrameRenderScratch::default();
        render_axial_view_into(&app, &mut view, &mut scratch)?;
        self.app = app;
        self.view = view;
        self.scratch = scratch;
        self.series_index = Some(series_index);
        Ok(())
    }

    pub(super) fn refresh(&mut self) -> Result<()> {
        if self.app.loaded.is_some() {
            render_axial_view_into(&self.app, &mut self.view, &mut self.scratch)?;
        }
        Ok(())
    }

    pub(super) const fn axial_view(&self) -> &RenderedView {
        &self.view
    }
}

impl NativeViewerSession {
    pub(super) fn initialize_comparison_uid(&mut self, uid: &str) -> Result<()> {
        let browser = self
            .series_browser
            .as_ref()
            .ok_or_else(|| anyhow!("series comparison requires a DICOM study catalog"))?;
        let index = browser
            .index_for_uid(uid)
            .ok_or_else(|| anyhow!("comparison SeriesInstanceUID is absent from the study"))?;
        if self.primary_series_index == Some(index) {
            return Err(anyhow!(
                "comparison SeriesInstanceUID must differ from the primary series"
            ));
        }
        let choice = browser
            .choice(index)
            .ok_or_else(|| anyhow!("comparison series index is outside the study catalog"))?;
        let modality = choice.modality.to_string();
        let instance_count = choice.instance_count;
        let volume = load_volume_from_series_info(&choice.acquisition)
            .with_context(|| "open the comparison DICOM series")?;
        let mut panel = ComparePanel::empty()?;
        panel.replace(
            volume,
            index,
            format!("Loaded {modality} series ({instance_count} instances)."),
        )?;
        let grid = PanelGrid::new(2, 1).ok_or_else(|| {
            anyhow!("side-by-side comparison grid is outside the supported range")
        })?;
        self.set_workspace_layout(WorkspaceLayout::Panels(grid))?;
        let target = self
            .compare_panels
            .get_mut(0)
            .ok_or_else(|| anyhow!("side-by-side layout has no second panel"))?;
        *target = panel;
        self.active_panel = 0;
        Ok(())
    }

    pub(super) fn series_panel_label(&self, panel: usize) -> Result<ArrayString<96>> {
        let index = if panel == 0 {
            self.primary_series_index
        } else {
            self.compare_panels
                .get(panel - 1)
                .and_then(|compare| compare.series_index)
        };
        let mut label = ArrayString::<96>::new();
        if let (Some(browser), Some(index)) = (self.series_browser.as_ref(), index) {
            let choice = browser
                .choice(index)
                .ok_or_else(|| anyhow!("displayed series is absent from its study catalog"))?;
            write!(&mut label, "P{}  |  {}  |  ", panel + 1, choice.modality)
                .map_err(|_| anyhow!("comparison modality label exceeds its buffer"))?;
            let remaining = 96_usize.saturating_sub(label.len());
            for character in choice.description.chars().take(remaining) {
                label
                    .try_push(character)
                    .map_err(|_| anyhow!("comparison series label exceeds its buffer"))?;
            }
        } else if panel == 0 && self.app.loaded.is_some() {
            write!(&mut label, "P1  |  RITK volume")
                .map_err(|_| anyhow!("primary comparison label exceeds its buffer"))?;
        } else {
            write!(&mut label, "P{}  |  Select a series", panel + 1)
                .map_err(|_| anyhow!("empty comparison label exceeds its buffer"))?;
        }
        Ok(label)
    }

    pub(super) fn reset_comparison(&mut self) {
        self.compare_panels.clear();
        self.workspace_layout = WorkspaceLayout::Orthogonal;
        self.active_panel = 0;
        self.active_view = None;
    }
}
