//! Interactive Windows presentation session backed by the Métis host loop.
//!
//! RITK owns the loaded volume, display policy and input transitions. Métis
//! receives only the retained framebuffer and reports bounded native events.
//! The session composes RITK views into one bounded framebuffer. RITK owns
//! DICOM decoding, display policy and input state; Métis owns the native
//! surface and receives only that framebuffer.

use crate::app::SnapApp;
use crate::dicom::loader::{
    load_volume_from_path, load_volume_from_series_uid, scan_folder_for_series,
};
use crate::dicom::series_tree::SeriesEntryView;
use crate::launch::NativePresentationMode;
use crate::tools::interaction::ViewportOffset;
use anyhow::{anyhow, Context, Result};
use metis_platform::native::{
    pick, run_native_application, DialogSelection, WindowConfig, WindowVisibility,
};
use metis_platform::Framebuffer;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

mod frame;
mod layout;
mod projection;
use frame::{render_orthogonal_views, RenderedView};
use layout::NativeViewport;
use projection::{render_projection, RenderedProjection};
mod composition;
mod events;
mod routing;
mod selection;
mod startup;
use composition::{compose_frames, save_capture};
use selection::{SelectionAction, SeriesSelection};
use startup::prepare_initial_study;

const INITIAL_WIDTH: u32 = 1_280;
const INITIAL_HEIGHT: u32 = 800;
const EVENT_WAIT: Duration = Duration::from_millis(16);
const NATIVE_TITLE: &str = "RITK-SNAP — Métis native";
const VIRTUAL_KEY_OPEN_STUDY: u32 = 0x4f;

mod outcome;
pub use outcome::NativeViewerOutcome;

/// Run one loaded DICOM study through the interactive Métis native host.
///
/// RITK opens and decodes `initial_path`, optionally selecting
/// `initial_series_uid` after discovery, applies its existing hanging protocol
/// and window/level rules, and renders the three orthogonal slices. The
/// Projection layouts add a display-only fourth panel using the selected RITK
/// scalar statistic (`maximum`, `minimum`, or `average`).
/// The Métis host owns the visible window, finite event wait, framebuffer
/// presentation and terminal cleanup. When `capture` is supplied, the window
/// is hidden and the session closes after its first idle event batch, then
/// writes the final composed RITK framebuffer.
///
/// # Errors
/// Returns a DICOM load, frame conversion, native-host, or capture error. A
/// missing path and a host destruction before a requested capture are errors.
#[must_use = "the session outcome records host and viewer transitions"]
pub fn run_native_viewer(
    initial_path: impl AsRef<Path>,
    initial_series_uid: Option<&str>,
    capture: Option<&Path>,
    presentation_mode: NativePresentationMode,
    capture_application: bool,
) -> Result<NativeViewerOutcome> {
    let initial_path = initial_path.as_ref();
    let mut app = SnapApp::default();
    let selection = match initial_series_uid {
        Some(series_uid) => {
            let volume =
                load_volume_from_series_uid(initial_path, series_uid).with_context(|| {
                    format!("open selected RITK series from {}", initial_path.display())
                })?;
            app.load_volume(
                volume,
                format!(
                    "Loaded native Métis series {}: {}",
                    series_uid,
                    initial_path.display()
                ),
            );
            None
        }
        None => prepare_initial_study(&mut app, initial_path, capture.is_some())?,
    };

    let observation = Arc::new(NativeViewerObservation::default());
    let session = NativeViewerSession::new_with_selection(
        app,
        Arc::clone(&observation),
        capture.is_some(),
        presentation_mode,
        capture_application,
        selection,
    )?;
    let config = WindowConfig::with_visibility(
        NATIVE_TITLE,
        INITIAL_WIDTH,
        INITIAL_HEIGHT,
        if capture.is_some() {
            WindowVisibility::Hidden
        } else {
            WindowVisibility::Visible
        },
    )?;

    run_native_application(&config, session, EVENT_WAIT)
        .map_err(|error| anyhow!("Métis native viewer host failed: {error}"))?;

    let final_frame = observation
        .final_frame
        .lock()
        .map_err(|_| anyhow!("native viewer observation lock was poisoned"))?
        .take();
    if let Some(output) = capture {
        if observation.destroyed.load(Ordering::Relaxed) {
            return Err(anyhow!(
                "native viewer was destroyed before capture completed"
            ));
        }
        let frame = final_frame
            .as_ref()
            .ok_or_else(|| anyhow!("native viewer closed without a final frame"))?;
        save_capture(frame, output)?;
    }

    let zoom = f32::from_bits(observation.zoom_bits.load(Ordering::Relaxed));
    Ok(NativeViewerOutcome {
        surface_width: observation.surface_width.load(Ordering::Relaxed),
        surface_height: observation.surface_height.load(Ordering::Relaxed),
        initial_frame_width: observation.initial_frame_width.load(Ordering::Relaxed),
        initial_frame_height: observation.initial_frame_height.load(Ordering::Relaxed),
        view_count: 3,
        presented_frames: observation.presented_frames.load(Ordering::Relaxed),
        event_batches: observation.event_batches.load(Ordering::Relaxed),
        translated_events: observation.translated_events.load(Ordering::Relaxed),
        frame_generations: observation.frame_generations.load(Ordering::Relaxed),
        last_slice: observation.last_slice.load(Ordering::Relaxed),
        zoom,
        dpi: observation.dpi.load(Ordering::Relaxed),
        minimized: observation.minimized.load(Ordering::Relaxed),
        destroyed: observation.destroyed.load(Ordering::Relaxed),
    })
}

fn viewport_offset(app: &SnapApp) -> ViewportOffset {
    app.pan_offset
}

struct NativeViewerSession {
    app: SnapApp,
    views: [RenderedView; 3],
    projection: Option<RenderedProjection>,
    presentation_mode: NativePresentationMode,
    framebuffer: Framebuffer,
    viewports: [NativeViewport; 3],
    active_view: Option<usize>,
    surface_width: u32,
    surface_height: u32,
    dpi: u32,
    minimized: bool,
    capture_after_idle: bool,
    capture_application: bool,
    observation: Arc<NativeViewerObservation>,
    clock_start: Instant,
    selection: Option<SeriesSelection>,
}

impl NativeViewerSession {
    fn new_with_selection(
        app: SnapApp,
        observation: Arc<NativeViewerObservation>,
        capture_after_idle: bool,
        presentation_mode: NativePresentationMode,
        capture_application: bool,
        selection: Option<SeriesSelection>,
    ) -> Result<Self> {
        let views = if app.loaded.is_some() {
            render_orthogonal_views(&app)?
        } else {
            frame::empty_orthogonal_views()?
        };
        let projection = match presentation_mode.projection_statistic() {
            None => None,
            Some(statistic) => Some(if app.loaded.is_some() {
                render_projection(&app, statistic)?
            } else {
                projection::empty_projection(statistic)?
            }),
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
            projection,
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
        if session.selection.is_some() {
            session.render_selection_overlay()?;
        }
        record_state(&session.observation, &session.app, 96, false);
        Ok(session)
    }

    fn elapsed_seconds(&self) -> f64 {
        self.clock_start.elapsed().as_secs_f64()
    }

    fn refresh_frame(&mut self) -> Result<()> {
        let (views, projection) = if self.app.loaded.is_some() {
            let views = render_orthogonal_views(&self.app)?;
            let projection = match self.presentation_mode.projection_statistic() {
                None => None,
                Some(statistic) => Some(render_projection(&self.app, statistic)?),
            };
            (views, projection)
        } else {
            (self.views.clone(), self.projection.clone())
        };
        let (framebuffer, viewports) = compose_frames(
            &views,
            projection.as_ref(),
            self.presentation_mode,
            self.surface_width,
            self.surface_height,
            self.app.zoom,
            viewport_offset(&self.app),
            self.app.cine.enabled,
            self.app.cine.fps,
            self.capture_application,
        )?;
        self.views = views;
        self.projection = projection;
        self.framebuffer = framebuffer;
        self.viewports = viewports;
        if self.selection.is_some() {
            self.render_selection_overlay()?;
        }
        self.observation
            .frame_generations
            .fetch_add(1, Ordering::Relaxed);
        record_state(&self.observation, &self.app, self.dpi, self.minimized);
        Ok(())
    }

    fn open_study_path(&mut self, path: &Path) -> Result<()> {
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

    fn open_study_from_dialog(&mut self) -> Result<bool> {
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

    fn reduce_selection_key(&mut self, virtual_key: u32, repeated: bool) -> Result<(bool, bool)> {
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

    fn record_terminal_frame(&self, destroyed: bool) -> Result<()> {
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

#[derive(Debug)]
struct NativeViewerError {
    message: Box<str>,
}

impl NativeViewerError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into().into_boxed_str(),
        }
    }
}

impl fmt::Display for NativeViewerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for NativeViewerError {}

impl From<anyhow::Error> for NativeViewerError {
    fn from(error: anyhow::Error) -> Self {
        Self::new(error.to_string())
    }
}

#[derive(Default)]
struct NativeViewerObservation {
    surface_width: AtomicU32,
    surface_height: AtomicU32,
    initial_frame_width: AtomicU32,
    initial_frame_height: AtomicU32,
    presented_frames: AtomicUsize,
    event_batches: AtomicUsize,
    translated_events: AtomicUsize,
    frame_generations: AtomicUsize,
    last_slice: AtomicUsize,
    zoom_bits: AtomicU32,
    dpi: AtomicU32,
    minimized: AtomicBool,
    destroyed: AtomicBool,
    final_frame: Mutex<Option<Framebuffer>>,
}

fn record_state(observation: &NativeViewerObservation, app: &SnapApp, dpi: u32, minimized: bool) {
    let (slice, _) = app.axis_slice_info(app.axis);
    observation.last_slice.store(slice, Ordering::Relaxed);
    observation
        .zoom_bits
        .store(app.zoom.to_bits(), Ordering::Relaxed);
    observation.dpi.store(dpi, Ordering::Relaxed);
    observation.minimized.store(minimized, Ordering::Relaxed);
}

#[cfg(test)]
mod tests;
