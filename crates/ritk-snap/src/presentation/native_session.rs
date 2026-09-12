//! Interactive Windows presentation session backed by the Métis host loop.
//!
//! RITK owns the loaded volume, display policy and input transitions. Métis
//! receives only the retained framebuffer and reports bounded native events.
//! The session composes all three orthogonal RITK planes into one bounded
//! framebuffer. RITK owns DICOM decoding, display policy and input state;
//! Métis owns the native surface and receives only that framebuffer.

use super::{translate_native_events, PresentationEvent};
use crate::app::SnapApp;
use crate::dicom::loader::{load_volume_from_path, load_volume_from_series_uid};
use anyhow::{anyhow, Context, Result};
use metis_platform::native::{
    run_native_application, NativeApplication, NativeFlow, WindowConfig, WindowEvent,
    WindowVisibility,
};
use metis_platform::Framebuffer;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

mod frame;
use frame::{render_orthogonal_views, surface_frames, NativeViewport, RenderedView};
mod routing;

const INITIAL_WIDTH: u32 = 1_280;
const INITIAL_HEIGHT: u32 = 800;
const EVENT_WAIT: Duration = Duration::from_millis(16);
const NATIVE_TITLE: &str = "RITK-SNAP — Métis native";

mod outcome;
pub use outcome::NativeViewerOutcome;

/// Run one loaded DICOM study through the interactive Métis native host.
///
/// RITK opens and decodes `initial_path`, optionally selecting
/// `initial_series_uid` after discovery, applies its existing hanging protocol
/// and window/level rules, and renders the three orthogonal slices.
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
) -> Result<NativeViewerOutcome> {
    let initial_path = initial_path.as_ref();
    let mut app = SnapApp::default();
    let volume = match initial_series_uid {
        Some(series_uid) => {
            load_volume_from_series_uid(initial_path, series_uid).with_context(|| {
                format!("open selected RITK series from {}", initial_path.display())
            })?
        }
        None => load_volume_from_path(initial_path)
            .with_context(|| format!("open initial RITK study at {}", initial_path.display()))?,
    };
    let status = match initial_series_uid {
        Some(series_uid) => format!(
            "Loaded native Métis series {}: {}",
            series_uid,
            initial_path.display()
        ),
        None => format!("Loaded native Métis study: {}", initial_path.display()),
    };
    app.load_volume(volume, status);

    let observation = Arc::new(NativeViewerObservation::default());
    let session = NativeViewerSession::new(app, Arc::clone(&observation), capture.is_some())?;
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

fn save_capture(framebuffer: &Framebuffer, output: &Path) -> Result<()> {
    let pixel_count =
        usize::try_from(u64::from(framebuffer.width()) * u64::from(framebuffer.height()))
            .map_err(|_| anyhow!("native capture pixel count exceeds usize"))?;
    if framebuffer.pixels().len() != pixel_count {
        return Err(anyhow!(
            "native capture framebuffer storage is inconsistent"
        ));
    }
    let byte_count = pixel_count
        .checked_mul(4)
        .ok_or_else(|| anyhow!("native capture byte count overflows usize"))?;
    let mut rgba = Vec::new();
    rgba.try_reserve_exact(byte_count)
        .map_err(|_| anyhow!("unable to reserve native capture bytes"))?;
    for packed in framebuffer.pixels() {
        let [alpha, red, green, blue] = packed.to_be_bytes();
        rgba.extend_from_slice(&[red, green, blue, alpha]);
    }
    let pixels = image::RgbaImage::from_raw(framebuffer.width(), framebuffer.height(), rgba)
        .ok_or_else(|| anyhow!("native capture RGBA dimensions do not match the framebuffer"))?;
    pixels
        .save_with_format(output, image::ImageFormat::Png)
        .with_context(|| format!("write native Métis capture to {}", output.display()))
}

struct NativeViewerSession {
    app: SnapApp,
    views: [RenderedView; 3],
    framebuffer: Framebuffer,
    viewports: [NativeViewport; 3],
    active_view: Option<usize>,
    surface_width: u32,
    surface_height: u32,
    dpi: u32,
    minimized: bool,
    capture_after_idle: bool,
    observation: Arc<NativeViewerObservation>,
}

impl NativeViewerSession {
    fn new(
        app: SnapApp,
        observation: Arc<NativeViewerObservation>,
        capture_after_idle: bool,
    ) -> Result<Self> {
        let views = render_orthogonal_views(&app)?;
        let (framebuffer, viewports) = surface_frames(
            &views,
            INITIAL_WIDTH,
            INITIAL_HEIGHT,
            app.zoom,
            app.pan_offset,
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
        record_state(&observation, &app, 96, false);
        Ok(Self {
            app,
            views,
            framebuffer,
            viewports,
            active_view: None,
            surface_width: INITIAL_WIDTH,
            surface_height: INITIAL_HEIGHT,
            dpi: 96,
            minimized: false,
            capture_after_idle,
            observation,
        })
    }

    fn refresh_frame(&mut self) -> Result<()> {
        let views = render_orthogonal_views(&self.app)?;
        let (framebuffer, viewports) = surface_frames(
            &views,
            self.surface_width,
            self.surface_height,
            self.app.zoom,
            self.app.pan_offset,
        )?;
        self.views = views;
        self.framebuffer = framebuffer;
        self.viewports = viewports;
        self.observation
            .frame_generations
            .fetch_add(1, Ordering::Relaxed);
        record_state(&self.observation, &self.app, self.dpi, self.minimized);
        Ok(())
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

impl NativeApplication for NativeViewerSession {
    type Error = NativeViewerError;

    fn framebuffer(&self) -> &Framebuffer {
        self.observation
            .presented_frames
            .fetch_add(1, Ordering::Relaxed);
        &self.framebuffer
    }

    fn handle_events(
        &mut self,
        events: &[WindowEvent],
    ) -> std::result::Result<NativeFlow, NativeViewerError> {
        let translated = translate_native_events(events).map_err(|error| {
            NativeViewerError::new(format!("translate Métis native events: {error}"))
        })?;
        self.observation
            .event_batches
            .fetch_add(1, Ordering::Relaxed);
        self.observation
            .translated_events
            .fetch_add(translated.len(), Ordering::Relaxed);
        if translated.is_empty() && self.capture_after_idle {
            self.record_terminal_frame(false)
                .map_err(NativeViewerError::from)?;
            return Ok(NativeFlow::Exit);
        }

        let mut resize = None;
        let mut dpi = None;
        let mut terminal = false;
        let mut destroyed = false;
        for event in translated.iter() {
            match event {
                PresentationEvent::Resized { width, height } => resize = Some((*width, *height)),
                PresentationEvent::DpiChanged { dpi: value } => dpi = Some(*value),
                PresentationEvent::CloseRequested => terminal = true,
                PresentationEvent::Destroyed => {
                    terminal = true;
                    destroyed = true;
                }
                _ => {}
            }
        }
        if dpi == Some(0) {
            return Err(NativeViewerError::new("native display DPI must be nonzero"));
        }

        let resized = resize.is_some_and(|(width, height)| width > 0 && height > 0);
        if let Some((width, height)) = resize {
            self.surface_width = width;
            self.surface_height = height;
            self.minimized = width == 0 || height == 0;
            self.observation
                .surface_width
                .store(width, Ordering::Relaxed);
            self.observation
                .surface_height
                .store(height, Ordering::Relaxed);
            self.observation
                .minimized
                .store(self.minimized, Ordering::Relaxed);
        }
        if let Some(value) = dpi {
            self.dpi = value;
            self.observation.dpi.store(value, Ordering::Relaxed);
        }

        // Establish the new viewport before reducing pointer events in the
        // same provider batch. Métis can coalesce a resize with input, and
        // those coordinates must use the new client rectangle.
        let geometry_refreshed = if resized {
            self.refresh_frame().map_err(NativeViewerError::from)?;
            true
        } else {
            false
        };
        let disposition = self.apply_events(&translated)?;

        let repaint = matches!(
            disposition,
            crate::app::action_adapter::ViewerActionDisposition::Continue { repaint: true }
        );
        if terminal
            || matches!(
                disposition,
                crate::app::action_adapter::ViewerActionDisposition::Exit
            )
        {
            if !self.minimized && (repaint || (resized && !geometry_refreshed)) {
                self.refresh_frame().map_err(NativeViewerError::from)?;
            } else if !geometry_refreshed {
                record_state(&self.observation, &self.app, self.dpi, self.minimized);
            }
            self.record_terminal_frame(destroyed)
                .map_err(NativeViewerError::from)?;
            return Ok(NativeFlow::Exit);
        }

        if !self.minimized && (repaint || (resized && !geometry_refreshed)) {
            self.refresh_frame().map_err(NativeViewerError::from)?;
        } else if !geometry_refreshed {
            record_state(&self.observation, &self.app, self.dpi, self.minimized);
        }
        Ok(NativeFlow::Continue { repaint: false })
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
