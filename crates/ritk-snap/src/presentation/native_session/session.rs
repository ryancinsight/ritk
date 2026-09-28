//! Interactive native viewer session state and per-event transitions.
//!
//! [`NativeViewerSession`] owns the composed framebuffer, render scratch and
//! DICOM study and series state for one Métis native host loop; `native_session`'s
//! sibling `events` and `routing` modules drive it through
//! [`NativeApplication`](metis_platform::native::NativeApplication) and the
//! RITK presentation action reducer.

use super::compare::ComparePanel;
use super::composition::compose_frames;
use super::layout::{
    surface_frames_grid, GridPanel, NativeViewport, WorkspaceLayout, MAX_COMPARISON_PANELS,
    MAX_GRID_PANELS,
};
use super::observation::{record_state, NativeViewerObservation};
use super::projection::{
    empty_projection, render_projection_into, ProjectionRenderScratch, RenderedProjection,
};
use super::series_browser::SeriesBrowser;
use super::{frame, WindowChrome, INITIAL_HEIGHT, INITIAL_WIDTH};
use crate::app::SnapApp;
use crate::dicom::loader::{
    load_volume_from_path, load_volume_from_series_info, scan_folder_for_series,
};
use crate::launch::NativePresentationSelection;
use crate::render::FrameRenderScratch;
use crate::tools::interaction::ViewportOffset;
use anyhow::{anyhow, Context, Result};
use arrayvec::{ArrayString, ArrayVec};
use frame::{render_orthogonal_views, render_orthogonal_views_into, RenderedView};
use metis_platform::native::{pick, DialogSelection};
use metis_platform::Framebuffer;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

fn viewport_offset(app: &SnapApp) -> ViewportOffset {
    app.pan_offset
}

pub(super) struct NativeViewerSession {
    pub(super) app: SnapApp,
    pub(super) views: [RenderedView; 3],
    pub(super) render_scratch: [FrameRenderScratch; 3],
    pub(super) projection: Option<RenderedProjection>,
    pub(super) projection_scratch: ProjectionRenderScratch,
    pub(super) presentation_mode: NativePresentationSelection,
    pub(super) framebuffer: Framebuffer,
    pub(super) viewports: ArrayVec<NativeViewport, MAX_GRID_PANELS>,
    pub(super) active_view: Option<usize>,
    pub(super) surface_width: u32,
    pub(super) surface_height: u32,
    pub(super) dpi: u32,
    pub(super) minimized: bool,
    pub(super) capture_after_idle: bool,
    pub(super) capture_application: bool,
    pub(super) observation: Arc<NativeViewerObservation>,
    pub(super) clock_start: Instant,
    pub(super) series_browser: Option<SeriesBrowser>,
    pub(super) primary_series_index: Option<usize>,
    pub(super) compare_panels: ArrayVec<ComparePanel, MAX_COMPARISON_PANELS>,
    pub(super) workspace_layout: WorkspaceLayout,
    pub(super) active_panel: usize,
    pub(super) window_chrome: WindowChrome,
}

impl NativeViewerSession {
    pub(super) fn new_with_browser(
        app: SnapApp,
        observation: Arc<NativeViewerObservation>,
        capture_after_idle: bool,
        presentation_mode: NativePresentationSelection,
        capture_application: bool,
        series_browser: Option<SeriesBrowser>,
        comparison_series_uid: Option<&str>,
    ) -> Result<Self> {
        let mut render_scratch = std::array::from_fn(|_| FrameRenderScratch::default());
        let views = if app.loaded.is_some() {
            render_orthogonal_views(&app, &mut render_scratch)?
        } else {
            frame::empty_orthogonal_views()?
        };
        let mut projection_scratch = ProjectionRenderScratch::default();
        let projection = match presentation_mode.projection_statistic() {
            None => None,
            Some(statistic) => {
                let mut projection = empty_projection(statistic)?;
                if app.loaded.is_some() {
                    render_projection_into(
                        &app,
                        statistic,
                        &mut projection,
                        &mut projection_scratch,
                    )?;
                }
                Some(projection)
            }
        };
        let window_chrome = WindowChrome::new(!capture_after_idle);
        let viewport_area = window_chrome.viewport_area(INITIAL_WIDTH, INITIAL_HEIGHT)?;
        let (framebuffer, initial_viewports) = compose_frames(
            &views,
            projection.as_ref(),
            presentation_mode,
            INITIAL_WIDTH,
            INITIAL_HEIGHT,
            viewport_area,
            app.zoom,
            viewport_offset(&app),
            app.cine.enabled,
            app.cine.fps,
            capture_application,
        )?;
        let mut viewports = ArrayVec::new();
        for viewport in initial_viewports {
            viewports
                .try_push(viewport)
                .map_err(|_| anyhow!("initial orthogonal layout exceeds viewport capacity"))?;
        }
        observation
            .initial_frame_width
            .store(views[0].frame().width(), Ordering::Relaxed);
        observation
            .initial_frame_height
            .store(views[0].frame().height(), Ordering::Relaxed);
        observation
            .surface_width
            .store(INITIAL_WIDTH, Ordering::Relaxed);
        observation
            .surface_height
            .store(INITIAL_HEIGHT, Ordering::Relaxed);
        observation.frame_generations.store(1, Ordering::Relaxed);
        let primary_series_index = series_browser.as_ref().map(SeriesBrowser::active_index);
        let mut session = Self {
            app,
            views,
            render_scratch,
            projection,
            projection_scratch,
            presentation_mode,
            framebuffer,
            viewports,
            active_view: None,
            surface_width: INITIAL_WIDTH,
            surface_height: INITIAL_HEIGHT,
            dpi: 96,
            minimized: false,
            capture_after_idle,
            capture_application,
            observation,
            clock_start: Instant::now(),
            series_browser,
            primary_series_index,
            compare_panels: ArrayVec::new(),
            workspace_layout: WorkspaceLayout::Orthogonal,
            active_panel: 0,
            window_chrome,
        };
        if let Some(uid) = comparison_series_uid {
            session.initialize_comparison_uid(uid)?;
            session.refresh_frame()?;
        } else {
            session.render_crosshair_overlay()?;
            session.render_chrome()?;
            record_state(&session.observation, &session.app, 96, false)?;
        }
        Ok(session)
    }

    pub(super) fn elapsed_seconds(&self) -> f64 {
        self.clock_start.elapsed().as_secs_f64()
    }

    pub(super) fn active_app(&self) -> &SnapApp {
        if self.workspace_layout.is_grid() && self.active_panel > 0 {
            if let Some(panel) = self.compare_panels.get(self.active_panel - 1) {
                return &panel.app;
            }
        }
        &self.app
    }

    pub(super) fn active_app_mut(&mut self) -> &mut SnapApp {
        if self.workspace_layout.is_grid() && self.active_panel > 0 {
            if let Some(panel) = self.compare_panels.get_mut(self.active_panel - 1) {
                return &mut panel.app;
            }
        }
        &mut self.app
    }

    pub(super) fn set_workspace_layout(&mut self, layout: WorkspaceLayout) -> Result<bool> {
        if self.workspace_layout == layout {
            return Ok(false);
        }
        let was_grid = self.workspace_layout.is_grid();
        if let Some(grid) = layout.grid() {
            let needed = grid.panel_count().saturating_sub(1);
            while self.compare_panels.len() < needed {
                self.compare_panels
                    .try_push(ComparePanel::empty()?)
                    .map_err(|_| anyhow!("native series layout exceeds its 20-panel limit"))?;
            }
            self.active_panel = if was_grid {
                self.active_panel.min(grid.panel_count().saturating_sub(1))
            } else {
                1.min(grid.panel_count().saturating_sub(1))
            };
        } else {
            self.active_panel = 0;
        }
        self.workspace_layout = layout;
        self.active_view = None;
        Ok(true)
    }

    pub(super) fn refresh_frame(&mut self) -> Result<()> {
        if self.app.loaded.is_some() {
            render_orthogonal_views_into(&self.app, &mut self.views, &mut self.render_scratch)?;
            match self.presentation_mode.projection_statistic() {
                None => self.projection = None,
                Some(statistic) => {
                    if let Some(projection) = self.projection.as_mut() {
                        render_projection_into(
                            &self.app,
                            statistic,
                            projection,
                            &mut self.projection_scratch,
                        )?;
                    } else {
                        let mut projection = empty_projection(statistic)?;
                        render_projection_into(
                            &self.app,
                            statistic,
                            &mut projection,
                            &mut self.projection_scratch,
                        )?;
                        self.projection = Some(projection);
                    }
                }
            }
        }
        let visible_comparison_panels = self
            .workspace_layout
            .grid()
            .map_or(0, |grid| grid.panel_count().saturating_sub(1));
        for panel in self
            .compare_panels
            .iter_mut()
            .take(visible_comparison_panels)
        {
            panel.refresh()?;
        }
        let viewport_area = self
            .window_chrome
            .viewport_area(self.surface_width, self.surface_height)?;
        let (framebuffer, viewports) = if let Some(grid) = self.workspace_layout.grid() {
            let mut labels: ArrayVec<ArrayString<96>, MAX_GRID_PANELS> = ArrayVec::new();
            for index in 0..grid.panel_count() {
                labels
                    .try_push(self.series_panel_label(index)?)
                    .map_err(|_| anyhow!("native series-grid labels exceed panel capacity"))?;
            }
            let mut panels: ArrayVec<GridPanel<'_>, MAX_GRID_PANELS> = ArrayVec::new();
            for index in 0..grid.panel_count() {
                let (view, navigation) = if index == 0 {
                    (
                        self.app.loaded.as_ref().map(|_| &self.views[0]),
                        (self.app.zoom, viewport_offset(&self.app)),
                    )
                } else {
                    let panel = self
                        .compare_panels
                        .get(index - 1)
                        .ok_or_else(|| anyhow!("native series-grid panel state is missing"))?;
                    (
                        panel.app.loaded.as_ref().map(|_| panel.axial_view()),
                        (panel.app.zoom, viewport_offset(&panel.app)),
                    )
                };
                panels
                    .try_push(GridPanel {
                        view,
                        label: labels[index].as_str(),
                        navigation,
                    })
                    .map_err(|_| anyhow!("native series grid exceeds panel capacity"))?;
            }
            let composition = surface_frames_grid(
                panels.as_slice(),
                grid,
                self.active_panel,
                &self.views[0],
                [self.surface_width, self.surface_height],
                viewport_area,
            )?;
            (composition.framebuffer, composition.viewports)
        } else {
            let (framebuffer, composition_viewports) = compose_frames(
                &self.views,
                self.projection.as_ref(),
                self.presentation_mode,
                self.surface_width,
                self.surface_height,
                viewport_area,
                self.app.zoom,
                viewport_offset(&self.app),
                self.app.cine.enabled,
                self.app.cine.fps,
                self.capture_application,
            )?;
            let mut viewports = ArrayVec::new();
            for viewport in composition_viewports {
                viewports
                    .try_push(viewport)
                    .map_err(|_| anyhow!("orthogonal viewports exceed panel capacity"))?;
            }
            (framebuffer, viewports)
        };
        self.framebuffer = framebuffer;
        self.viewports = viewports;
        self.render_crosshair_overlay()?;
        self.render_chrome()?;
        self.observation
            .frame_generations
            .fetch_add(1, Ordering::Relaxed);
        record_state(
            &self.observation,
            self.active_app(),
            self.dpi,
            self.minimized,
        )?;
        Ok(())
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
        if self.primary_series_index == Some(index) {
            let changed = self.active_panel != 0 || browser.active_index() != index;
            self.active_panel = 0;
            self.series_browser
                .as_mut()
                .expect("invariant: selected series retains its study browser")
                .set_active(index);
            return Ok(changed);
        }
        if self.workspace_layout.is_grid() {
            if let Some(panel_index) = self
                .compare_panels
                .iter()
                .position(|panel| panel.series_index == Some(index))
            {
                let panel_index = panel_index.saturating_add(1);
                let changed = self.active_panel != panel_index || browser.active_index() != index;
                self.active_panel = panel_index;
                self.series_browser
                    .as_mut()
                    .expect("invariant: selected series retains its study browser")
                    .set_active(index);
                return Ok(changed);
            }
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

    pub(super) fn record_terminal_frame(&self, destroyed: bool) -> Result<()> {
        self.observation
            .destroyed
            .store(destroyed, Ordering::Relaxed);
        let mut final_frame = self
            .observation
            .final_frame
            .lock()
            .map_err(|_| anyhow!("native viewer observation lock was poisoned"))?;
        if final_frame.is_none() {
            *final_frame = Some(self.framebuffer.clone());
        }
        Ok(())
    }
}
