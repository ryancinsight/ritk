use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::render::{ProjectionKind, ProjectionStatistic};

#[cfg(not(target_arch = "wasm32"))]
mod series_comparison;
mod viewport;
#[cfg(not(target_arch = "wasm32"))]
pub use series_comparison::run_app_with_series_comparison;
#[cfg(not(target_arch = "wasm32"))]
mod responsive;
#[cfg(not(target_arch = "wasm32"))]
pub use responsive::run_responsive_native_app_with_options;
pub use viewport::{EframeViewport, EframeViewportError};

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
    /// Orthogonal panels plus the RITK axial front-to-back volume rendering.
    #[value(name = "orthogonal-with-vr")]
    OrthogonalWithVr,
}

impl NativePresentationMode {
    /// Return the projection content used by a projection layout, if any.
    #[must_use]
    pub const fn projection_kind(self) -> Option<ProjectionKind> {
        match self {
            Self::Orthogonal => None,
            Self::OrthogonalWithMip => Some(ProjectionKind::Slab(ProjectionStatistic::Maximum)),
            Self::OrthogonalWithMinip => Some(ProjectionKind::Slab(ProjectionStatistic::Minimum)),
            Self::OrthogonalWithAverage => Some(ProjectionKind::Slab(ProjectionStatistic::Average)),
            Self::OrthogonalWithVr => Some(ProjectionKind::VolumeRender),
        }
    }
}

/// Internal native presentation selection used by the host entrypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativePresentationSelection {
    /// Use one of the stable public fixed presentation modes.
    Fixed(NativePresentationMode),
    /// Select one, two or four panes from the current native surface extent.
    Responsive,
}

#[cfg(any(windows, target_arch = "wasm32"))]
impl NativePresentationSelection {
    pub(crate) const fn projection_kind(self) -> Option<ProjectionKind> {
        match self {
            Self::Fixed(mode) => mode.projection_kind(),
            Self::Responsive => Some(ProjectionKind::Slab(ProjectionStatistic::Maximum)),
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
    /// explicitly. An omitted path opens an empty workspace.
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

/// Launch the `ritk-snap` native GUI application with startup options.
///
/// With `metis_native`, `initial_path` is opened by RITK before the interactive
/// Métis host starts. Without a path the viewer starts empty. The user can
/// open a study from the File menu or with Ctrl+O. `initial_series_uid`
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
        None,
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
        None,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn run_app_with_compatibility_selection(
    options: AppLaunchOptions,
    compatibility_presentation: CompatibilityPresentation,
    viewport: EframeViewport,
    presentation_selection: NativePresentationSelection,
    comparison_series_uid: Option<String>,
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
            crate::presentation::run_native_session(
                options.initial_path.as_deref(),
                options.initial_series_uid.as_deref(),
                options.capture.as_deref(),
                presentation_selection,
                options.capture_application,
                comparison_series_uid.as_deref(),
            )?;
            return Ok(());
        }
        #[cfg(not(windows))]
        {
            let _ = options.capture;
            let _ = options.capture_application;
            let _ = presentation_selection;
            let _ = comparison_series_uid;
            anyhow::bail!("--metis-native requires a Windows Métis native host");
        }
    }
    #[cfg(feature = "eframe-shell")]
    {
        anyhow::ensure!(
            comparison_series_uid.is_none(),
            "series comparison requires the Métis native host"
        );
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
                .with_title(crate::app::PRODUCT_NAME)
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
        let _ = comparison_series_uid;
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
