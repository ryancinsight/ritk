use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
mod capture;

/// Native Métis framebuffer layout selected at application startup.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum NativePresentationMode {
    /// Three spacing-aware orthogonal panels.
    #[default]
    #[value(name = "orthogonal")]
    Orthogonal,
    /// Orthogonal panels plus the RITK axial maximum-intensity projection.
    #[value(name = "orthogonal-with-mip")]
    OrthogonalWithMip,
}

/// Startup configuration for the native `ritk-snap` application.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppLaunchOptions {
    /// Optional DICOM folder or medical image file to load at startup.
    pub initial_path: Option<PathBuf>,
    /// Optional SeriesInstanceUID selected from the startup DICOM input.
    ///
    /// This is required when a directory contains more than one acquisition
    /// and keeps native and eframe startup on the same explicit-selection path.
    #[serde(default)]
    pub initial_series_uid: Option<String>,
    /// Save the rendered application frame as PNG and exit.
    ///
    /// A supplied initial study must load successfully. Capture failure is
    /// returned to the caller; closing the window early does not report success.
    #[serde(default)]
    pub capture: Option<PathBuf>,
    /// Include the bounded RITK application overlay in a Métis capture.
    ///
    /// This affects only the Métis native capture content. Operating-system
    /// decorations are outside the framebuffer contract.
    #[serde(default)]
    pub capture_application: bool,
    /// Use the Métis native host instead of the eframe shell. This requires a
    /// startup path and is currently available on Windows.
    #[serde(default)]
    pub metis_native: bool,
    /// Native Métis layout. Non-default values require `metis_native`.
    #[serde(default)]
    pub native_presentation_mode: NativePresentationMode,
}

/// Launch the `ritk-snap` native GUI application.
///
/// Initialises the selected desktop shell with a 1280×800 viewport, constructs
/// the default viewer state, and enters the platform event loop. This function
/// blocks until the window is closed.
///
/// # Errors
/// Returns an error if `eframe` cannot create a window or encounters a fatal
/// platform error during the event loop.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_app() -> anyhow::Result<()> {
    run_app_with_options(AppLaunchOptions::default())
}

/// Launch the `ritk-snap` native GUI application with startup options.
///
/// With `metis_native`, `initial_path` is opened by RITK before the interactive
/// Métis host starts. `initial_series_uid` selects one acquisition after RITK
/// discovery when the path contains several series. With the default eframe
/// shell, `initial_path` is queued for loading on the first UI update. Directory
/// paths are also scanned for the DICOM series browser before the first frame;
/// a requested capture waits for that load to publish before taking its frame.
///
/// # Errors
/// Returns a host creation/event-loop error. With capture requested, also
/// returns an error if the initial study fails, the PNG cannot be saved, the
/// screenshot response exceeds its deadline, or the window closes early.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_app_with_options(options: AppLaunchOptions) -> anyhow::Result<()> {
    if options.metis_native {
        let path = options
            .initial_path
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("--metis-native requires an initial DICOM path"))?;
        #[cfg(windows)]
        {
            crate::presentation::run_native_viewer(
                path,
                options.initial_series_uid.as_deref(),
                options.capture.as_deref(),
                options.native_presentation_mode,
                options.capture_application,
            )?;
            return Ok(());
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            let _ = options.capture;
            let _ = options.capture_application;
            let _ = options.native_presentation_mode;
            anyhow::bail!("--metis-native requires a Windows Métis native host");
        }
    }
    if options.native_presentation_mode != NativePresentationMode::Orthogonal {
        anyhow::bail!("native presentation layout requires the Métis native host");
    }
    use std::cell::Cell;
    use std::rc::Rc;

    let completion = Rc::new(Cell::new(None));
    let capture_requested = options.capture.is_some();
    let app_completion = Rc::clone(&completion);
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ritk-snap — DICOM Viewer")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "ritk-snap",
        native_options,
        Box::new(move |_cc| {
            let requirement = if options.initial_path.is_some() {
                capture::Requirement::Study
            } else {
                capture::Requirement::Application
            };
            let app = match options.initial_path {
                Some(path) => {
                    crate::app::SnapApp::with_initial_path(path, options.initial_series_uid.clone())
                }
                None => crate::app::SnapApp::default(),
            };
            Ok(Box::new(capture::CaptureApp::new(
                app,
                options.capture,
                requirement,
                app_completion,
            )))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))?;
    if capture_requested {
        completion
            .take()
            .ok_or_else(|| anyhow::anyhow!("window closed before screenshot completion"))??;
    }
    Ok(())
}

/// Stub launcher for non-native targets.
///
/// On wasm targets, use [`start_web`] or [`start_web_canvas`] to launch
/// `ritk-snap` in a browser.
#[cfg(target_arch = "wasm32")]
pub fn run_app_with_options(_options: AppLaunchOptions) -> anyhow::Result<()> {
    anyhow::bail!(
        "run_app_with_options is native-only; use start_web() or start_web_canvas() on wasm32"
    )
}

/// Start the RITK browser canvas workflow through the Métis host.
///
/// This asynchronous entrypoint is exported only for `wasm32` and keeps the
/// original JavaScript bootstrap contract. It delegates to
/// [`start_web_canvas`], so the browser path receives only the format-neutral
/// Métis canvas and bounded file handoff; RITK retains DICOM decoding and
/// viewer state.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen_futures::JsFuture;

    crate::app::start_web_canvas(canvas_id)?;

    // Preserve the async bootstrap contract so existing JavaScript callers can
    // await startup while the bounded browser task begins its first tick.
    JsFuture::from(js_sys::Promise::resolve(&wasm_bindgen::JsValue::UNDEFINED))
        .await
        .map_err(|e| {
            wasm_bindgen::JsValue::from_str(&format!("web startup promise failed: {e:?}"))
        })?;

    Ok(())
}

/// Start the RITK browser canvas workflow with Métis and Moirai.
///
/// The workflow receives bounded browser file bytes from Métis, lets RITK
/// classify and decode them, and presents the selected RITK frame through the
/// named HTML5 canvas. [`start_web_orthogonal_canvases`] presents the three
/// RITK orthogonal frames through three named canvases. [`start_web`] is the
/// asynchronous JavaScript-compatible wrapper for this single-canvas path.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn start_web_canvas(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_canvas(canvas_id)
}

/// Start the RITK browser canvas workflow with three orthogonal views.
///
/// The identifiers are ordered axial, coronal, sagittal. RITK owns the
/// decoded volume, slice selection and display semantics; Métis owns only the
/// browser canvases and bounded file handoff.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn start_web_orthogonal_canvases(
    axial_id: String,
    coronal_id: String,
    sagittal_id: String,
) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_orthogonal_canvases([axial_id, coronal_id, sagittal_id])
}

/// Stop the RITK browser canvas workflow and release its browser task.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn stop_web_canvas() {
    crate::app::stop_web_canvas();
}
