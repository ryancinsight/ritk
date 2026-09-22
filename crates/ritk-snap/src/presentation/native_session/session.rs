//! Interactive native viewer session state and per-event transitions.
//!
//! [`NativeViewerSession`] owns the composed framebuffer, render scratch and
//! DICOM selection state for one Métis native host loop; `native_session`'s
//! sibling `events` and `routing` modules drive it through
//! [`NativeApplication`](metis_platform::native::NativeApplication) and the
//! RITK presentation action reducer.

use super::composition::compose_frames;
use super::layout::{crosshair_overlay, NativeViewport};
use super::observation::{record_state, NativeViewerObservation};
use super::projection::{
    empty_projection, render_projection_into, ProjectionRenderScratch, RenderedProjection,
};
use super::selection::{SelectionAction, SeriesSelection};
use super::{frame, INITIAL_HEIGHT, INITIAL_WIDTH};
use crate::app::SnapApp;
use crate::dicom::loader::{
    load_volume_from_path, load_volume_from_series_uid, scan_folder_for_series,
};
use crate::dicom::series_tree::SeriesEntryView;
use crate::launch::NativePresentationSelection;
use crate::render::FrameRenderScratch;
use crate::tools::interaction::ViewportOffset;
use anyhow::{anyhow, Context, Result};
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
    pub(super) viewports: [NativeViewport; 3],
    pub(super) active_view: Option<usize>,
    pub(super) surface_width: u32,
    pub(super) surface_height: u32,
    pub(super) dpi: u32,
    pub(super) minimized: bool,
    pub(super) capture_after_idle: bool,
    pub(super) capture_application: bool,
    pub(super) observation: Arc<NativeViewerObservation>,
    pub(super) clock_start: Instant,
    pub(super) selection: Option<SeriesSelection>,
}

impl NativeViewerSession {
    pub(super) fn new_with_selection(
        app: SnapApp,
        observation: Arc<NativeViewerObservation>,
        capture_after_idle: bool,
        presentation_mode: NativePresentationSelection,
        capture_application: bool,
        selection: Option<SeriesSelection>,
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
        let (framebuffer, viewports) = compose_frames(
            &views,
            projection.as_ref(),
            presentation_mode,
            INITIAL_WIDTH,
            INITIAL_HEIGHT,
            app.zoom,
            viewport_offset(&app),
            app.cine.enabled,
            app.cine.fps,
            capture_application,
        )?;
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
            selection,
        };
        session.render_crosshair_overlay()?;
        if session.selection.is_some() {
            session.render_selection_overlay()?;
        }
        record_state(&session.observation, &session.app, 96, false)?;
        Ok(session)
    }

    pub(super) fn elapsed_seconds(&self) -> f64 {
        self.clock_start.elapsed().as_secs_f64()
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
        let (framebuffer, viewports) = compose_frames(
            &self.views,
            self.projection.as_ref(),
            self.presentation_mode,
            self.surface_width,
            self.surface_height,
            self.app.zoom,
            viewport_offset(&self.app),
            self.app.cine.enabled,
            self.app.cine.fps,
            self.capture_application,
        )?;
        self.framebuffer = framebuffer;
        self.viewports = viewports;
        self.render_crosshair_overlay()?;
        if self.selection.is_some() {
            self.render_selection_overlay()?;
        }
        self.observation
            .frame_generations
            .fetch_add(1, Ordering::Relaxed);
        record_state(&self.observation, &self.app, self.dpi, self.minimized)?;
        Ok(())
    }

    pub(super) fn open_study_path(&mut self, path: &Path) -> Result<()> {
        if path.is_dir() {
            let tree = scan_folder_for_series(path).context("discover selected RITK study")?;
            match tree.total_series() {
                0 => {
                    let volume = load_volume_from_path(path).context("open selected RITK study")?;
                    self.app.load_volume(
                        volume,
                        format!("Loaded native Métis study: {}", path.display()),
                    );
                    self.selection = None;
                }
                1 => {
                    let series = tree
                        .iter_series()
                        .next()
                        .expect("invariant: one discovered series has one entry");
                    let uid = series.series_uid();
                    let volume = load_volume_from_series_uid(path, uid)
                        .with_context(|| format!("open selected RITK series {uid}"))?;
                    self.app.load_volume(
                        volume,
                        format!("Loaded native Métis series {}: {}", uid, path.display()),
                    );
                    self.selection = None;
                }
                _ => {
                    self.selection = Some(SeriesSelection::from_tree(path, &tree)?);
                    self.app.status_message = format!(
                        "Select one of {} DICOM series before loading {}",
                        self.selection.as_ref().map_or(0, SeriesSelection::len),
                        path.display()
                    );
                }
            }
        } else {
            let volume = load_volume_from_path(path).context("open selected RITK study")?;
            self.app.load_volume(
                volume,
                format!("Loaded native Métis study: {}", path.display()),
            );
            self.selection = None;
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

    fn render_selection_overlay(&mut self) -> Result<()> {
        if let Some(selection) = &self.selection {
            selection.render_to(&mut self.framebuffer)?;
        }
        Ok(())
    }

    fn render_crosshair_overlay(&mut self) -> Result<()> {
        let shape = self.app.loaded.as_ref().map(|volume| volume.shape);
        let cursor = self.app.linked_cursor.map(|cursor| cursor.voxel());
        crosshair_overlay(
            &self.views,
            &self.viewports,
            shape,
            cursor,
            self.app.show_crosshair,
        )?
        .render_to(&mut self.framebuffer);
        Ok(())
    }

    pub(super) fn reduce_selection_key(
        &mut self,
        virtual_key: u32,
        repeated: bool,
    ) -> Result<(bool, bool)> {
        let Some(selection) = self.selection.as_mut() else {
            return Ok((false, false));
        };
        let action = selection.handle_key(virtual_key, repeated);
        match action {
            SelectionAction::Changed => Ok((true, false)),
            SelectionAction::Canceled => {
                self.selection = None;
                self.app.status_message =
                    "DICOM series selection canceled; current study remains displayed.".to_owned();
                Ok((true, false))
            }
            SelectionAction::Confirmed => {
                let (path, uid) = self
                    .selection
                    .as_ref()
                    .expect("invariant: confirmed selection remains present")
                    .selected_request();
                match load_volume_from_series_uid(&path, &uid) {
                    Ok(volume) => {
                        self.selection = None;
                        self.app.load_volume(
                            volume,
                            format!("Loaded native Métis series {}: {}", uid, path.display()),
                        );
                        Ok((true, true))
                    }
                    Err(error) => {
                        let message = format!(
                            "DICOM series {} could not be opened: {error:#}; choose another series.",
                            uid
                        );
                        self.app.status_message = message.clone();
                        if let Some(selection) = self.selection.as_mut() {
                            selection.set_notice(message.into_boxed_str());
                        }
                        Ok((true, false))
                    }
                }
            }
            SelectionAction::Ignored => Ok((false, false)),
        }
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
