use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Startup configuration for the native `ritk-snap` application.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppLaunchOptions {
    /// Optional DICOM folder or medical image file to load at startup.
    pub initial_path: Option<PathBuf> }

/// Launch the `ritk-snap` native GUI application.
///
/// Initialises `eframe` with a 1280Ã—800 viewport, constructs a `app::SnapApp`,
/// via [`Default`], and enters the platform event loop. This function blocks
/// until the window is closed.
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
/// When `initial_path` is present, the app queues that path for loading on the
/// first UI update. Directory paths are also scanned for the DICOM series
/// browser before the first frame.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_app_with_options(options: AppLaunchOptions) -> anyhow::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("ritk-snap â€” DICOM Viewer")
            .with_inner_size([1280.0, 800.0]),
        ..Default::default()
    };

    eframe::run_native(
        "ritk-snap",
        native_options,
        Box::new(move |_cc| {
            let app = match options.initial_path.clone() {
                Some(path) => crate::app::SnapApp::with_initial_path(path),
                None => crate::app::SnapApp::default() };
            Ok(Box::new(app))
        }),
    )
    .map_err(|e| anyhow::anyhow!("eframe error: {e}"))
}

/// Stub launcher for non-native targets.
///
/// On wasm targets, use [`start_web`] to launch `ritk-snap` in a browser.
#[cfg(target_arch = "wasm32")]
pub fn run_app_with_options(_options: AppLaunchOptions) -> anyhow::Result<()> {
    anyhow::bail!("run_app_with_options is native-only; use start_web() on wasm32")
}

/// Start the `ritk-snap` egui viewer in a browser canvas.
///
/// This entrypoint is exported only for `wasm32` and is intended to be called
/// from JavaScript after loading the generated wasm module.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    use wasm_bindgen_futures::JsFuture;

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
