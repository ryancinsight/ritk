//! AsterSlice workstation, shared by the Windows and browser hosts.
//! Both hosts submit the same client-coordinate events and present the same
//! composed framebuffer. File access and window lifecycle belong to the host.

#[cfg(windows)]
use crate::app::{SnapApp, PRODUCT_NAME};
#[cfg(windows)]
use crate::launch::{NativePresentationMode, NativePresentationSelection};
#[cfg(windows)]
use anyhow::{anyhow, Result};
#[cfg(windows)]
use metis_platform::native::{run_native_application, WindowConfig, WindowVisibility};
#[cfg(windows)]
use std::path::Path;
#[cfg(windows)]
use std::sync::atomic::Ordering;
#[cfg(windows)]
use std::sync::Arc;
#[cfg(windows)]
use std::time::Duration;

mod compare;
mod composition;
#[cfg(windows)]
mod deidentify;
mod events;
mod frame;
mod layout;
mod pressed_buttons;
mod projection;
#[cfg(windows)]
mod repair;
mod routing;
mod series;
mod series_browser;
#[cfg(windows)]
mod series_loads;
mod slice_scrollbar;
#[cfg(windows)]
mod startup;
mod window_actions;
mod window_controls;
#[cfg(windows)]
use composition::save_capture;
use series_browser::SeriesBrowser;
#[cfg(windows)]
use startup::prepare_initial_study;
use window_controls::{WindowAction, WindowChrome};

mod observation;
mod panels;
mod session;
mod session_chrome;
use observation::{record_state, NativeViewerError, NativeViewerObservation};
use session::NativeViewerSession;

const INITIAL_WIDTH: u32 = 1_280;
const INITIAL_HEIGHT: u32 = 800;
#[cfg(windows)]
const EVENT_WAIT: Duration = Duration::from_millis(16);
const VIRTUAL_KEY_OPEN_STUDY: u32 = 0x4f;

#[cfg(windows)]
mod outcome;
#[cfg(windows)]
pub use outcome::NativeViewerOutcome;

#[cfg(windows)]
mod desktop;
#[cfg(windows)]
pub use desktop::{run_native_comparison_viewer, run_native_responsive_viewer, run_native_viewer};

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(all(test, windows))]
mod tests;
