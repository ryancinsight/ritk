//! Interactive Windows presentation session backed by the Métis host loop.
//!
//! RITK owns the loaded volume, display policy and input transitions. Métis
//! receives only the retained framebuffer and reports bounded native events.
//! This module is the first interactive migration slice; it deliberately
//! presents one active orthogonal plane while the three-view composition is
//! completed in a later increment.

use super::{translate_native_events, PresentationEvent, PresentationFrame};
use crate::app::action_adapter::ViewerViewport;
use crate::app::SnapApp;
use crate::dicom::loader::load_volume_from_path;
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
use frame::{render_current_slice, surface_frame};

const INITIAL_WIDTH: u32 = 1_280;
const INITIAL_HEIGHT: u32 = 800;
const EVENT_WAIT: Duration = Duration::from_millis(16);
const NATIVE_TITLE: &str = "RITK-SNAP — Métis native";

/// Observable result of an interactive native viewer session.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NativeViewerOutcome {
    surface_width: u32,
    surface_height: u32,
    initial_frame_width: u32,
    initial_frame_height: u32,
    presented_frames: usize,
    event_batches: usize,
    translated_events: usize,
    frame_generations: usize,
    last_slice: usize,
    zoom: f32,
    dpi: u32,
    minimized: bool,
    destroyed: bool,
}

impl NativeViewerOutcome {
    /// Final native surface width in client pixels.
    #[must_use]
    pub const fn surface_width(self) -> u32 {
        self.surface_width
    }

    /// Final native surface height in client pixels.
    #[must_use]
    pub const fn surface_height(self) -> u32 {
        self.surface_height
    }

    /// Width of the decoded RITK slice before host scaling.
    #[must_use]
    pub const fn initial_frame_width(self) -> u32 {
        self.initial_frame_width
    }

    /// Height of the decoded RITK slice before host scaling.
    #[must_use]
    pub const fn initial_frame_height(self) -> u32 {
        self.initial_frame_height
    }

    /// Number of framebuffer presentations requested by the host loop.
    #[must_use]
    pub const fn presented_frames(self) -> usize {
        self.presented_frames
    }

    /// Number of bounded event batches consumed by the viewer.
    #[must_use]
    pub const fn event_batches(self) -> usize {
        self.event_batches
    }

    /// Number of native events translated at the RITK boundary.
    #[must_use]
    pub const fn translated_events(self) -> usize {
        self.translated_events
    }

    /// Number of decoded frames rendered after state transitions.
    #[must_use]
    pub const fn frame_generations(self) -> usize {
        self.frame_generations
    }

    /// Final slice index along the active orthogonal axis.
    #[must_use]
    pub const fn last_slice(self) -> usize {
        self.last_slice
    }

    /// Final RITK zoom value.
    #[must_use]
    pub const fn zoom(self) -> f32 {
        self.zoom
    }

    /// Final display DPI reported by the host.
    #[must_use]
    pub const fn dpi(self) -> u32 {
        self.dpi
    }

    /// Whether the last host event left the surface minimized.
    #[must_use]
    pub const fn minimized(self) -> bool {
        self.minimized
    }

    /// Whether the host reported destruction rather than an orderly close.
    #[must_use]
    pub const fn destroyed(self) -> bool {
        self.destroyed
    }
}

/// Run one loaded DICOM study through the interactive Métis native host.
///
/// RITK opens and decodes `initial_path`, applies its existing hanging
/// protocol and window/level rules, and renders the active orthogonal slice.
/// The Métis host owns the visible window, finite event wait, framebuffer
/// presentation and terminal cleanup. When `capture` is supplied, the window
/// is hidden and the session closes after its first idle event batch, then
/// writes the final RITK frame.
///
/// # Errors
/// Returns a DICOM load, frame conversion, native-host, or capture error. A
/// missing path and a host destruction before a requested capture are errors.
#[must_use = "the session outcome records host and viewer transitions"]
pub fn run_native_viewer(
    initial_path: impl AsRef<Path>,
    capture: Option<&Path>,
) -> Result<NativeViewerOutcome> {
    let initial_path = initial_path.as_ref();
    let mut app = SnapApp::default();
    let volume = load_volume_from_path(initial_path)
        .with_context(|| format!("open initial RITK study at {}", initial_path.display()))?;
    app.load_volume(
        volume,
        format!("Loaded native Métis study: {}", initial_path.display()),
    );

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

fn save_capture(frame: &PresentationFrame, output: &Path) -> Result<()> {
    let pixels = image::RgbaImage::from_raw(frame.width(), frame.height(), frame.rgba().to_vec())
        .ok_or_else(|| anyhow!("native capture RGBA dimensions do not match the frame"))?;
    pixels
        .save_with_format(output, image::ImageFormat::Png)
        .with_context(|| format!("write native Métis capture to {}", output.display()))
}

struct NativeViewerSession {
    app: SnapApp,
    source_frame: PresentationFrame,
    framebuffer: Framebuffer,
    viewport: ViewerViewport,
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
        let (source_frame, source_size, transform) = render_current_slice(&app)?;
        let (framebuffer, viewport) = surface_frame(
            &source_frame,
            source_size,
            transform,
            app.axis,
            INITIAL_WIDTH,
            INITIAL_HEIGHT,
            app.zoom,
        )?;
        observation
            .initial_frame_width
            .store(source_frame.width(), Ordering::Relaxed);
        observation
            .initial_frame_height
            .store(source_frame.height(), Ordering::Relaxed);
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
            source_frame,
            framebuffer,
            viewport,
            surface_width: INITIAL_WIDTH,
            surface_height: INITIAL_HEIGHT,
            dpi: 96,
            minimized: false,
            capture_after_idle,
            observation,
        })
    }

    fn refresh_frame(&mut self) -> Result<()> {
        let (source_frame, source_size, transform) = render_current_slice(&self.app)?;
        let (framebuffer, viewport) = surface_frame(
            &source_frame,
            source_size,
            transform,
            self.app.axis,
            self.surface_width,
            self.surface_height,
            self.app.zoom,
        )?;
        self.source_frame = source_frame;
        self.framebuffer = framebuffer;
        self.viewport = viewport;
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
            *final_frame = Some(self.source_frame.clone());
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
        let viewport = (!self.minimized).then_some(&self.viewport);
        let disposition = self
            .app
            .apply_presentation_events(&translated, viewport)
            .map_err(|error| {
                NativeViewerError::new(format!("apply RITK presentation events: {error}"))
            })?;

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
    final_frame: Mutex<Option<PresentationFrame>>,
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
