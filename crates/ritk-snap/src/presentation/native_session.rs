//! Interactive Windows presentation session backed by the Métis host loop.
//!
//! RITK owns the loaded volume, display policy and input transitions. Métis
//! receives only the retained framebuffer and reports bounded native events.
//! The session composes RITK views into one bounded framebuffer. RITK owns
//! DICOM decoding, display policy and input state; Métis owns the native
//! surface and receives only that framebuffer.

use crate::app::SnapApp;
use crate::dicom::loader::load_volume_from_series_uid;
use crate::launch::{NativePresentationMode, NativePresentationSelection};
use anyhow::{anyhow, Context, Result};
use metis_platform::native::{run_native_application, WindowConfig, WindowVisibility};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

mod frame;
mod layout;
mod projection;
use frame::RenderedView;
use layout::NativeViewport;
use projection::RenderedProjection;
mod composition;
mod events;
mod routing;
mod selection;
mod startup;
use composition::save_capture;
use selection::SeriesSelection;
use startup::prepare_initial_study;

mod observation;
mod session;
use observation::{record_state, NativeViewerError, NativeViewerObservation};
use session::NativeViewerSession;

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
    run_native_viewer_with_selection(
        initial_path,
        initial_series_uid,
        capture,
        NativePresentationSelection::Fixed(presentation_mode),
        capture_application,
    )
}

/// Run a loaded DICOM study with the responsive one-, two- or four-pane layout.
///
/// This additive entrypoint keeps [`NativePresentationMode`] exhaustive for
/// existing callers while exposing the new responsive host workflow without
/// changing that public enum's variants.
///
/// # Errors
/// Returns the same DICOM load, frame conversion, native-host, and capture
/// errors as [`run_native_viewer`].
#[must_use = "the session outcome records host and viewer transitions"]
pub fn run_native_responsive_viewer(
    initial_path: impl AsRef<Path>,
    initial_series_uid: Option<&str>,
    capture: Option<&Path>,
    capture_application: bool,
) -> Result<NativeViewerOutcome> {
    run_native_viewer_with_selection(
        initial_path,
        initial_series_uid,
        capture,
        NativePresentationSelection::Responsive,
        capture_application,
    )
}

fn run_native_viewer_with_selection(
    initial_path: impl AsRef<Path>,
    initial_series_uid: Option<&str>,
    capture: Option<&Path>,
    presentation_mode: NativePresentationSelection,
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

    let snapshot = observation
        .snapshot
        .lock()
        .map_err(|_| anyhow!("native viewer snapshot lock was poisoned"))?
        .take()
        .ok_or_else(|| anyhow!("native viewer completed without a presentation snapshot"))?;
    let zoom = f32::from_bits(observation.zoom_bits.load(Ordering::Relaxed));
    Ok(NativeViewerOutcome {
        snapshot,
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

#[cfg(test)]
mod tests;
