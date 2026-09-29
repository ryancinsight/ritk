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

    pub(super) fn clear(&mut self) -> Result<()> {
        self.app = SnapApp::default();
        self.view = empty_axial_view()?;
        self.scratch = FrameRenderScratch::default();
        self.series_index = None;
        Ok(())
    }

    pub(super) fn swap_primary(
        &mut self,
        app: &mut SnapApp,
        view: &mut RenderedView,
        scratch: &mut FrameRenderScratch,
        series_index: &mut Option<usize>,
    ) {
        std::mem::swap(&mut self.app, app);
        std::mem::swap(&mut self.view, view);
        std::mem::swap(&mut self.scratch, scratch);
        std::mem::swap(&mut self.series_index, series_index);
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
        let image_count = choice.image_count;
        let volume = load_volume_from_series_info(&choice.acquisition)
            .with_context(|| "open the comparison DICOM series")?;
        let mut panel = ComparePanel::empty()?;
        panel.replace(
            volume,
            index,
            format!("Loaded {modality} series ({image_count} images)."),
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
            append_series_panel_label(
                &mut label,
                panel + 1,
                &choice.modality,
                &choice.description,
            )?;
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
        self.maximized_panel = None;
        self.workspace_layout = WorkspaceLayout::Orthogonal;
        self.active_panel = 0;
        self.active_view = None;
    }
}

fn append_series_panel_label(
    label: &mut ArrayString<96>,
    panel: usize,
    modality: &str,
    description: &str,
) -> Result<()> {
    write!(label, "P{panel}  |  {modality}  |  ")
        .map_err(|_| anyhow!("comparison modality label exceeds its buffer"))?;
    let mut remaining = label.capacity().saturating_sub(label.len());
    for character in description.chars() {
        let bytes = character.len_utf8();
        if bytes > remaining {
            break;
        }
        label
            .try_push(character)
            .map_err(|_| anyhow!("comparison series label exceeds its buffer"))?;
        remaining = remaining.saturating_sub(bytes);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::append_series_panel_label;
    use arrayvec::ArrayString;

    #[test]
    fn multibyte_series_descriptions_fit_the_panel_label_buffer() {
        let description = "頭部画像".repeat(20);
        let mut label = ArrayString::<96>::new();

        append_series_panel_label(&mut label, 1, "MR", &description)
            .expect("truncate the label at a UTF-8 character boundary");

        assert_eq!(label.len(), 95);
        assert!(label.as_str().starts_with("P1  |  MR  |  "));
        assert!(label.as_str().ends_with("頭部画"));
    }
}
