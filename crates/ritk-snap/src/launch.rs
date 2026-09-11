use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
mod capture;

/// Startup configuration for the native `ritk-snap` application.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppLaunchOptions {
    /// Optional DICOM folder or medical image file to load at startup.
    pub initial_path: Option<PathBuf>,
    /// Save the rendered application frame as PNG and exit.
    ///
    /// A supplied initial study must load successfully. Capture failure is
    /// returned to the caller; closing the window early does not report success.
    #[serde(default)]
    pub capture: Option<PathBuf>,
    /// Use the Métis native host instead of the eframe shell. This requires a
    /// startup path and is currently available on Windows.
    #[serde(default)]
    pub metis_native: bool,
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
/// Métis host starts. With the default eframe shell, `initial_path` is queued
/// for loading on the first UI update. Directory paths are also scanned for the
/// DICOM series browser before the first frame.
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
            crate::presentation::run_native_viewer(path, options.capture.as_deref())?;
            return Ok(());
        }
        #[cfg(not(windows))]
        {
            let _ = path;
            let _ = options.capture;
            anyhow::bail!("--metis-native requires a Windows Métis native host");
        }
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
                Some(path) => crate::app::SnapApp::with_initial_path(path),
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

/// Start the `ritk-snap` egui viewer in a browser canvas.
///
/// This entrypoint is exported only for `wasm32` and is intended to be called
/// from JavaScript after loading the generated wasm module.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen_futures::JsFuture;

    // Mount the generic Métis HTML5/CSS host first. Its bounded file handoff
    // feeds RITK's existing DICOM routing; no format classification occurs in
    // the host crate.
    metis_web::metis_start();

    let web_options = eframe::WebOptions::default();

    let runner = eframe::WebRunner::new();
    runner
        .start(
            &canvas_id,
            web_options,
            Box::new(|_cc| Ok(Box::new(crate::app::SnapApp::default()))),
        )
        .await
        .map_err(|e| {
            wasm_bindgen::JsValue::from_str(&format!("failed to start web runner: {e:?}"))
        })?;

    // Yield once so startup errors surface as rejected promises to JS callers.
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
/// named HTML5 canvas. The existing [`start_web`] eframe entrypoint remains
/// available while the full multi-view browser shell is migrated.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn start_web_canvas(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_canvas(canvas_id)
}

/// Stop the RITK browser canvas workflow and release its browser task.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn stop_web_canvas() {
    crate::app::stop_web_canvas();
}
