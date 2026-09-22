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

/// Internal native presentation selection used by the host entrypoints.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativePresentationSelection {
    /// Use one of the stable public fixed presentation modes.
    Fixed(NativePresentationMode),
    /// Select one, two or four panes from the current native surface extent.
    Responsive,
}

#[cfg(not(target_arch = "wasm32"))]
impl NativePresentationSelection {
    pub(crate) const fn projection_statistic(self) -> Option<ProjectionStatistic> {
        match self {
            Self::Fixed(mode) => mode.projection_statistic(),
            Self::Responsive => Some(ProjectionStatistic::Maximum),
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
    let presentation_selection =
        NativePresentationSelection::Fixed(options.native_presentation_mode);
    run_app_with_compatibility_selection(
        options,
        CompatibilityPresentation::FullApplication,
        EframeViewport::default(),
        presentation_selection,
    )
}

/// Launch the Métis native host with its responsive pane layout.
///
/// This additive entrypoint preserves the public fixed-layout enum and its
/// existing exhaustive matches while exposing adaptive native presentation to
/// library consumers.
///
/// # Errors
/// Returns the same window, event-loop, DICOM load, and capture errors as
/// [`run_app_with_options`].
#[cfg(not(target_arch = "wasm32"))]
pub fn run_responsive_native_app_with_options(options: AppLaunchOptions) -> anyhow::Result<()> {
    run_app_with_compatibility_selection(
        options,
        CompatibilityPresentation::FullApplication,
        EframeViewport::default(),
        NativePresentationSelection::Responsive,
    )
}

#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "eframe-shell")]
fn run_app_with_compatibility(
    options: AppLaunchOptions,
    compatibility_presentation: CompatibilityPresentation,
    viewport: EframeViewport,
) -> anyhow::Result<()> {
    let presentation_selection =
        NativePresentationSelection::Fixed(options.native_presentation_mode);
    run_app_with_compatibility_selection(
        options,
        compatibility_presentation,
        viewport,
        presentation_selection,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn run_app_with_compatibility_selection(
    options: AppLaunchOptions,
    compatibility_presentation: CompatibilityPresentation,
    viewport: EframeViewport,
    presentation_selection: NativePresentationSelection,
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
            match presentation_selection {
                NativePresentationSelection::Fixed(mode) => {
                    crate::presentation::run_native_viewer(
                        &path,
                        options.initial_series_uid.as_deref(),
                        options.capture.as_deref(),
                        mode,
                        options.capture_application,
                    )?;
                }
                NativePresentationSelection::Responsive => {
                    crate::presentation::run_native_responsive_viewer(
                        &path,
                        options.initial_series_uid.as_deref(),
                        options.capture.as_deref(),
                        options.capture_application,
                    )?;
                }
            }
            return Ok(());
        }
        #[cfg(not(windows))]
        {
            let _ = options.capture;
            let _ = options.capture_application;
            let _ = presentation_selection;
            anyhow::bail!("--metis-native requires a Windows Métis native host");
        }
    }
    #[cfg(feature = "eframe-shell")]
    {
        if presentation_selection
            != NativePresentationSelection::Fixed(NativePresentationMode::Orthogonal)
        {
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

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::{
    select_web_slice, select_web_tool, set_web_cine_rate, set_web_window_preset, start_web,
    start_web_canvas, start_web_canvas_gpu, start_web_orthogonal_canvases,
    start_web_orthogonal_canvases_gpu, start_web_orthogonal_canvases_gpu_with_projection,
    start_web_orthogonal_canvases_with_projection, start_web_responsive_canvases, stop_web_canvas,
    toggle_web_cine, toggle_web_crosshair, web_canvas_listener_count, web_tool_count,
    web_tool_name, web_window_preset_count, web_window_preset_name,
};

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
