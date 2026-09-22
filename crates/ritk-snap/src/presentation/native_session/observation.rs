//! Atomic session telemetry shared across native host event callbacks.
//!
//! [`NativeViewerObservation`] is the cross-thread counter set the native
//! host callbacks and the outer driver in `native_session` both read after
//! every session mutation; [`record_state`] is the single site that updates
//! its app-derived fields. `NativeViewerError` is the boxed error type the
//! Métis `NativeApplication` trait requires from `NativeViewerSession`.

use crate::app::SnapApp;
use crate::presentation::PresentationSnapshot;
use anyhow::{anyhow, Result};
use metis_platform::Framebuffer;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Debug)]
pub(super) struct NativeViewerError {
    message: Box<str>,
}

impl NativeViewerError {
    pub(super) fn new(message: impl Into<String>) -> Self {
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
pub(super) struct NativeViewerObservation {
    pub(super) surface_width: AtomicU32,
    pub(super) surface_height: AtomicU32,
    pub(super) initial_frame_width: AtomicU32,
    pub(super) initial_frame_height: AtomicU32,
    pub(super) presented_frames: AtomicUsize,
    pub(super) event_batches: AtomicUsize,
    pub(super) translated_events: AtomicUsize,
    pub(super) frame_generations: AtomicUsize,
    pub(super) last_slice: AtomicUsize,
    pub(super) zoom_bits: AtomicU32,
    pub(super) dpi: AtomicU32,
    pub(super) minimized: AtomicBool,
    pub(super) destroyed: AtomicBool,
    pub(super) final_frame: Mutex<Option<Framebuffer>>,
    pub(super) snapshot: Mutex<Option<PresentationSnapshot>>,
}

pub(super) fn record_state(
    observation: &NativeViewerObservation,
    app: &SnapApp,
    dpi: u32,
    minimized: bool,
) -> Result<()> {
    let (slice, _) = app.axis_slice_info(app.axis);
    observation.last_slice.store(slice, Ordering::Relaxed);
    observation
        .zoom_bits
        .store(app.zoom.to_bits(), Ordering::Relaxed);
    observation.dpi.store(dpi, Ordering::Relaxed);
    observation.minimized.store(minimized, Ordering::Relaxed);
    *observation
        .snapshot
        .lock()
        .map_err(|_| anyhow!("native viewer snapshot lock was poisoned"))? =
        Some(app.presentation_snapshot());
    Ok(())
}
