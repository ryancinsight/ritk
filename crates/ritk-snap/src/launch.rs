use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::render::ProjectionStatistic;

mod viewport;
pub use viewport::{EframeViewport, EframeViewportError};

#[cfg(windows)]
use std::path::Path;

#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
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
    /// Orthogonal panels plus the RITK axial minimum-intensity projection.
    #[value(name = "orthogonal-with-minip")]
    OrthogonalWithMinip,
    /// Orthogonal panels plus the RITK axial average-intensity projection.
    #[value(name = "orthogonal-with-average")]
    OrthogonalWithAverage,
}

impl NativePresentationMode {
    /// Return the scalar reduction used by a projection layout, if any.
    #[must_use]
    pub const fn projection_statistic(self) -> Option<ProjectionStatistic> {
        match self {
            Self::Orthogonal => None,
            Self::OrthogonalWithMip => Some(ProjectionStatistic::Maximum),
            Self::OrthogonalWithMinip => Some(ProjectionStatistic::Minimum),
            Self::OrthogonalWithAverage => Some(ProjectionStatistic::Average),
        }
    }
}

/// Presentation selected by the compatibility shell.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, clap::ValueEnum)]
pub enum CompatibilityPresentation {
    /// Render the complete eframe application, including its shell controls.
    #[default]
    #[value(name = "full-application")]
    FullApplication,
    /// Render only the three spacing-aware orthogonal planes.
    #[value(name = "orthogonal-surface")]
    OrthogonalSurface,
}

/// Startup configuration for the native `ritk-snap` application.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(not(windows), derive(Default))]
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
    /// Use the Métis native host instead of the eframe shell. On Windows this
    /// defaults to `true`; callers can select the eframe compatibility shell
    /// explicitly. A missing startup path opens the bounded native folder
    /// picker.
    #[serde(default)]
    pub metis_native: bool,
    /// Native Métis layout. Non-default values require `metis_native`.
    #[serde(default)]
    pub native_presentation_mode: NativePresentationMode,
}

#[cfg(windows)]
impl Default for AppLaunchOptions {
    fn default() -> Self {
        Self {
            initial_path: None,
            initial_series_uid: None,
            capture: None,
            capture_application: false,
            metis_native: cfg!(windows),
            native_presentation_mode: NativePresentationMode::default(),
        }
    }
}

/// Launch the `ritk-snap` native GUI application.
///
/// Initialises the default desktop shell with a 1280×800 viewport, constructs
/// the default viewer state, and enters the platform event loop. Windows uses
/// the Métis native host by default; other native targets retain the eframe
/// shell until their Métis surface provider is available. This function blocks
/// until the window is closed.
///
/// # Errors
/// Returns an error if the selected shell cannot create a window or encounters
/// a fatal platform error during the event loop.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_app() -> anyhow::Result<()> {
    run_app_with_options(AppLaunchOptions::default())
}

/// Launch the complete eframe compatibility shell.
///
/// This entrypoint belongs to the separately named compatibility artifact. It
/// forces the eframe path while reusing the same [`AppLaunchOptions`] and RITK
/// viewer state as the default Métis binary.
///
/// # Errors
/// Returns the same window, event-loop, and capture errors as
/// [`run_app_with_options`].
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
pub fn run_eframe_app_with_options(options: AppLaunchOptions) -> anyhow::Result<()> {
    run_eframe_app_with_presentation(options, CompatibilityPresentation::FullApplication)
}

/// Launch the compatibility shell with an explicit presentation.
///
/// `OrthogonalSurface` is a bounded measurement fixture. It keeps the same
/// RITK loader, textures and physical-aspect placement as the complete shell,
/// while removing shell chrome and the 2×2/MIP layout so a resource record can
/// be compared with a three-plane host surface.
///
/// # Errors
/// Returns a window, event-loop, load, or capture error from the compatibility
/// shell.
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
pub fn run_eframe_app_with_presentation(
    options: AppLaunchOptions,
    presentation: CompatibilityPresentation,
) -> anyhow::Result<()> {
    run_eframe_app_with_viewport(options, presentation, EframeViewport::default())
}

/// Launch the compatibility shell with an explicit logical viewport size.
///
/// The size is useful for matched resource fixtures. The resulting physical
/// capture still depends on the host display scale and must be recorded by the
/// capture provenance.
///
/// # Errors
/// Returns a window, event-loop, load, or capture error from the compatibility
/// shell.
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
pub fn run_eframe_app_with_viewport(
    mut options: AppLaunchOptions,
    presentation: CompatibilityPresentation,
    viewport: EframeViewport,
) -> anyhow::Result<()> {
    options.metis_native = false;
    run_app_with_compatibility(options, presentation, viewport)
}

/// Launch the eframe compatibility shell with its default options.
///
/// # Errors
/// Returns a window or event-loop error from the compatibility shell.
#[cfg(all(not(target_arch = "wasm32"), feature = "eframe-shell"))]
pub fn run_eframe_app() -> anyhow::Result<()> {
    run_eframe_app_with_options(AppLaunchOptions::default())
}

#[cfg(windows)]
fn select_native_initial_path<F>(
    initial_path: Option<&Path>,
    pick_folder: F,
) -> anyhow::Result<PathBuf>
where
    F: FnOnce() -> anyhow::Result<Option<PathBuf>>,
{
    match initial_path {
        Some(path) => Ok(path.to_path_buf()),
        None => pick_folder()?
            .ok_or_else(|| anyhow::anyhow!("native Métis file selection was cancelled")),
    }
}

/// Launch the `ritk-snap` native GUI application with startup options.
///
/// With `metis_native`, `initial_path` is opened by RITK before the interactive
/// Métis host starts. When it is absent on Windows, the host opens a bounded
/// native folder picker and passes the selected path to RITK. A cancelled
/// picker returns an error without starting a window. `initial_series_uid`
/// selects one acquisition after RITK discovery when the path contains several
/// series. With the eframe compatibility shell, `initial_path` is queued for loading
/// on the first UI update. On Windows, the default launch uses the Métis host;
/// pass `metis_native: false` to select the eframe compatibility shell.
/// Directory paths are also scanned for the DICOM
/// series browser before the first frame; a requested capture waits for that
/// load to publish before taking its frame.
///
/// # Errors
/// Returns a host creation/event-loop error. With capture requested, also
/// returns an error if the initial study fails, the PNG cannot be saved, the
/// screenshot response exceeds its deadline, or the window closes early.
#[cfg(not(target_arch = "wasm32"))]
pub fn run_app_with_options(options: AppLaunchOptions) -> anyhow::Result<()> {
    run_app_with_compatibility(
        options,
        CompatibilityPresentation::FullApplication,
        EframeViewport::default(),
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn run_app_with_compatibility(
    options: AppLaunchOptions,
    compatibility_presentation: CompatibilityPresentation,
    viewport: EframeViewport,
) -> anyhow::Result<()> {
    #[cfg(not(feature = "eframe-shell"))]
    let _ = viewport;

    if options.metis_native {
        anyhow::ensure!(
            compatibility_presentation == CompatibilityPresentation::FullApplication,
            "compatibility presentation requires the eframe shell"
        );
        #[cfg(windows)]
        {
            use metis_platform::native::{pick, DialogSelection};

            let path = select_native_initial_path(options.initial_path.as_deref(), || {
                pick(DialogSelection::Folder).map_err(Into::into)
            })?;
            crate::presentation::run_native_viewer(
                &path,
                options.initial_series_uid.as_deref(),
                options.capture.as_deref(),
                options.native_presentation_mode,
                options.capture_application,
            )?;
            return Ok(());
        }
        #[cfg(not(windows))]
        {
            let _ = options.capture;
            let _ = options.capture_application;
            let _ = options.native_presentation_mode;
            anyhow::bail!("--metis-native requires a Windows Métis native host");
        }
    }
    #[cfg(feature = "eframe-shell")]
    {
        if options.native_presentation_mode != NativePresentationMode::Orthogonal {
            anyhow::bail!("native presentation layout requires the Métis native host");
        }
        if options.capture_application {
            anyhow::bail!("application capture requires the Métis native host");
        }
        use std::cell::Cell;
        use std::rc::Rc;

        let completion = Rc::new(Cell::new(None));
        let capture_requested = options.capture.is_some();
        let app_completion = Rc::clone(&completion);
        let native_options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_title("ritk-snap — DICOM Viewer")
                .with_inner_size(viewport.logical_size()),
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
                    Some(path) => crate::app::SnapApp::with_initial_path(
                        path,
                        options.initial_series_uid.clone(),
                    ),
                    None => crate::app::SnapApp::default(),
                };
                Ok(Box::new(capture::CaptureApp::new(
                    crate::app::EguiApp::new_with_presentation(app, compatibility_presentation),
                    options.capture,
                    requirement,
                    compatibility_presentation,
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
    #[cfg(not(feature = "eframe-shell"))]
    {
        let _ = options;
        anyhow::bail!("legacy eframe shell requires the `eframe-shell` feature")
    }
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

/// Start the RITK browser canvas workflow through an explicit WebGPU surface.
///
/// The future resolves after WebGPU adapter/device setup and listener
/// registration complete. Setup errors are returned to JavaScript; the
/// existing raster entrypoint is never selected implicitly.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web_canvas_gpu(canvas_id: String) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_canvas_gpu(canvas_id).await
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

/// Start the RITK orthogonal browser workflow through explicit WebGPU surfaces.
///
/// Identifiers are ordered axial, coronal, sagittal. The future rejects when
/// any canvas cannot acquire WebGPU or register its bounded input listeners.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web_orthogonal_canvases_gpu(
    axial_id: String,
    coronal_id: String,
    sagittal_id: String,
) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_orthogonal_canvases_gpu([axial_id, coronal_id, sagittal_id]).await
}

/// Start the browser workflow with three interactive planes and one
/// display-only scalar projection.
///
/// Canvas identifiers are ordered axial, coronal, sagittal, projection.
/// `projection` is `0` for maximum, `1` for minimum and `2` for average.
/// The value is validated before the viewer mounts.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn start_web_orthogonal_canvases_with_projection(
    axial_id: String,
    coronal_id: String,
    sagittal_id: String,
    projection_id: String,
    projection: f64,
) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_orthogonal_canvases_with_projection(
        [axial_id, coronal_id, sagittal_id, projection_id],
        projection,
    )
}

/// Start the four-canvas browser workflow with explicit WebGPU surfaces.
///
/// Setup errors are returned to JavaScript; the raster provider is never
/// selected implicitly. Canvas identifiers and statistic indices follow
/// [`start_web_orthogonal_canvases_with_projection`].
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub async fn start_web_orthogonal_canvases_gpu_with_projection(
    axial_id: String,
    coronal_id: String,
    sagittal_id: String,
    projection_id: String,
    projection: f64,
) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::start_web_orthogonal_canvases_gpu_with_projection(
        [axial_id, coronal_id, sagittal_id, projection_id],
        projection,
    )
    .await
}

/// Select an exact zero-based slice on one browser viewer axis.
///
/// Axes are `0` axial, `1` coronal and `2` sagittal. A successful change
/// invalidates the cached frames; the next animation frame renders the new
/// slice and republishes its `data-ritk-*` semantics.
///
/// # Errors
///
/// Returns a JavaScript error value when no viewer or study is available, the
/// viewer is handling another callback, or the axis/index is out of range.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn select_web_slice(axis: f64, index: f64) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::select_web_slice(axis, index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Toggle host-neutral cine playback for the loaded browser study.
///
/// The next browser animation frame establishes the timing anchor and
/// republishes `data-ritk-cine-enabled` on every RITK canvas.
///
/// # Errors
///
/// Returns a JavaScript error when the viewer is not mounted, another browser
/// callback owns it, or no study has been loaded.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn toggle_web_cine() -> Result<bool, wasm_bindgen::JsValue> {
    crate::app::toggle_web_cine()
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Set the exact bounded cine playback rate for the loaded browser study.
///
/// `rate` must be a finite integral value from 1 through 60 frames per
/// second. Invalid values are rejected before viewer state changes.
///
/// # Errors
///
/// Returns a JavaScript error when the value is invalid, the viewer is not
/// mounted, another browser callback owns it, or no study has been loaded.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_web_cine_rate(rate: f64) -> Result<bool, wasm_bindgen::JsValue> {
    crate::app::set_web_cine_rate(rate)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Select one loaded-study interaction tool from the RITK browser table.
///
/// The index is a finite integer in the range reported by
/// [`web_tool_count`]. Selecting a tool clears any in-progress gesture while
/// leaving the decoded study and rendered pixels unchanged.
///
/// # Errors
///
/// Returns a JavaScript error when the value is invalid, the viewer is not
/// mounted, another callback owns it, no study is loaded, or the index is
/// outside the table.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn select_web_tool(index: f64) -> Result<bool, wasm_bindgen::JsValue> {
    crate::app::select_web_tool(index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Return the number of interaction tools exposed to the browser palette.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn web_tool_count() -> Result<usize, wasm_bindgen::JsValue> {
    crate::app::web_tool_count()
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Return one interaction-tool label for browser palette construction.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn web_tool_name(index: f64) -> Result<String, wasm_bindgen::JsValue> {
    crate::app::web_tool_name(index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Apply one exact loaded-modality window/level preset to every browser view.
///
/// The index is validated as a finite non-negative integer against the table
/// selected from the loaded DICOM modality. A successful change invalidates
/// all retained frames; the next animation frame renders the new intensity
/// mapping and publishes the updated window attributes.
///
/// # Errors
///
/// Returns a JavaScript error when no viewer or study is available, the viewer
/// is handling another callback, or the index is outside the active preset
/// table.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn set_web_window_preset(index: f64) -> Result<(), wasm_bindgen::JsValue> {
    crate::app::set_web_window_preset(index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Return the number of window/level presets for the loaded modality.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn web_window_preset_count() -> Result<usize, wasm_bindgen::JsValue> {
    crate::app::web_window_preset_count()
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Return one window/level preset name for the loaded modality.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn web_window_preset_name(index: f64) -> Result<String, wasm_bindgen::JsValue> {
    crate::app::web_window_preset_name(index)
        .map_err(|error| wasm_bindgen::JsValue::from_str(&error.to_string()))
}

/// Stop the RITK browser canvas workflow and release its browser task.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
pub fn stop_web_canvas() {
    crate::app::stop_web_canvas();
}

/// Return the number of browser canvas listener guards retained by RITK.
///
/// The count is zero after [`stop_web_canvas`] returns. A mounted single-canvas
/// viewer reports one provider input set; the orthogonal viewer reports three.
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen]
#[must_use]
pub fn web_canvas_listener_count() -> usize {
    crate::app::web_canvas_listener_count()
}

#[cfg(all(test, windows))]
mod tests {
    use super::select_native_initial_path;
    use std::path::Path;

    #[test]
    fn preserves_explicit_path_without_opening_picker() {
        let path = select_native_initial_path(Some(Path::new("study")), || {
            Err(anyhow::anyhow!("picker must not run"))
        })
        .expect("explicit startup paths do not require the picker");

        assert_eq!(path, Path::new("study"));
    }

    #[test]
    fn forwards_selected_folder() {
        let path = select_native_initial_path(None, || Ok(Some("selected-study".into())))
            .expect("selected folder is returned");

        assert_eq!(path, Path::new("selected-study"));
    }

    #[test]
    fn reports_picker_cancellation() {
        let error = select_native_initial_path(None, || Ok(None)).expect_err("cancel is an error");

        assert_eq!(
            error.to_string(),
            "native Métis file selection was cancelled"
        );
    }

    #[test]
    fn preserves_picker_failure_cause() {
        let error = select_native_initial_path(None, || {
            Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "folder selection denied",
            )
            .into())
        })
        .expect_err("provider failure must prevent startup");

        let source = error
            .downcast_ref::<std::io::Error>()
            .expect("provider error type must survive the selection boundary");
        assert_eq!(source.kind(), std::io::ErrorKind::PermissionDenied);
        assert_eq!(source.to_string(), "folder selection denied");
    }
}
