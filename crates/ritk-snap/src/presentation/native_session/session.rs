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
    GridPanel, MAX_COMPARISON_PANELS, MAX_GRID_PANELS, NativeViewport, WorkspaceLayout,
    surface_frames_grid,
};
use super::observation::{NativeViewerObservation, record_state};
use super::panels::MaximizedPanel;
use super::projection::{
    ProjectionRenderScratch, RenderedProjection, empty_projection, render_projection_into,
};
use super::series_browser::SeriesBrowser;
use super::{INITIAL_HEIGHT, INITIAL_WIDTH, WindowChrome, frame};
use crate::app::SnapApp;
use crate::launch::NativePresentationSelection;
use crate::render::FrameRenderScratch;
use crate::tools::interaction::{ToolState, ViewportOffset};
use crate::tools::kind::ToolKind;
use anyhow::{Result, anyhow};
use arrayvec::{ArrayString, ArrayVec};
use frame::{RenderedView, render_orthogonal_views, render_orthogonal_views_into};
use metis_platform::Framebuffer;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Instant;

#[path = "annotations.rs"]
mod annotations;

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
    pub(super) maximized_panel: Option<MaximizedPanel>,
    pub(super) window_chrome: WindowChrome,
    pub(super) suppress_cancelled_pointer_release: Option<crate::presentation::PointerButton>,
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
            maximized_panel: None,
            window_chrome,
            suppress_cancelled_pointer_release: None,
        };
        if let Some(uid) = comparison_series_uid {
            session.initialize_comparison_uid(uid)?;
            session.refresh_frame()?;
        } else {
            session.render_crosshair_overlay()?;
            session.render_measurement_overlays()?;
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
        let restored = self.restore_maximized_panel()?;
        if self.workspace_layout == layout {
            return Ok(restored);
        }
        let was_grid = self.workspace_layout.is_grid();
        if let Some(grid) = layout.grid() {
            for app in std::iter::once(&mut self.app)
                .chain(self.compare_panels.iter_mut().map(|panel| &mut panel.app))
            {
                app.show_crosshair = false;
                if app.active_tool == ToolKind::Crosshair {
                    app.active_tool = ToolKind::Pan;
                    app.tool_state = ToolState::Idle;
                }
            }
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
                        maximized: self.maximized_panel.is_some() && index == 0,
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
        self.render_measurement_overlays()?;
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

    fn render_measurement_overlays(&mut self) -> Result<()> {
        if let Some(grid) = self.workspace_layout.grid() {
            for panel_index in 0..grid.panel_count().min(self.viewports.len()) {
                let viewport = self.viewports[panel_index];
                if panel_index == 0 {
                    annotations::render_measurements(
                        &mut self.framebuffer,
                        &self.app,
                        &self.views[0],
                        viewport,
                    )?;
                } else {
                    let panel = self
                        .compare_panels
                        .get(panel_index - 1)
                        .ok_or_else(|| anyhow!("native measurement panel state is missing"))?;
                    annotations::render_measurements(
                        &mut self.framebuffer,
                        &panel.app,
                        panel.axial_view(),
                        viewport,
                    )?;
                }
            }
            return Ok(());
        }

        let Some((view, viewport)) = self
            .views
            .iter()
            .zip(self.viewports.iter().copied())
            .find(|(view, _)| view.axis == self.app.axis)
        else {
            return Ok(());
        };
        annotations::render_measurements(&mut self.framebuffer, &self.app, view, viewport)
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
