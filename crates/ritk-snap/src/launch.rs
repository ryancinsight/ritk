use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::render::ProjectionStatistic;

#[cfg(windows)]
mod series_comparison;
#[cfg(windows)]
pub use series_comparison::run_app_with_series_comparison;
#[cfg(windows)]
mod responsive;
#[cfg(windows)]
pub use responsive::run_responsive_native_app_with_options;
#[cfg(windows)]
use std::path::Path;

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
#[cfg(windows)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NativePresentationSelection {
    /// Use one of the stable public fixed presentation modes.
    Fixed(NativePresentationMode),
    /// Select one, two or four panes from the current native surface extent.
    Responsive,
}

#[cfg(windows)]
impl NativePresentationSelection {
    pub(crate) const fn projection_statistic(self) -> Option<ProjectionStatistic> {
        match self {
            Self::Fixed(mode) => mode.projection_statistic(),
            Self::Responsive => Some(ProjectionStatistic::Maximum),
        }
    }
}

/// Startup configuration for the native `ritk-snap` application.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AppLaunchOptions {
    /// Optional DICOM folder or medical image file to load at startup.
    pub initial_path: Option<PathBuf>,
    /// Optional SeriesInstanceUID selected from the startup DICOM input.
    ///
    /// This is required when a directory contains more than one acquisition.
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
    /// Native Métis layout.
    #[serde(default)]
    pub native_presentation_mode: NativePresentationMode,
}

/// Launch the `ritk-snap` viewer on the Métis native host.
///
/// Opens the bounded native folder picker, loads the selected study, and enters
/// the platform event loop. This function blocks until the window is closed.
///
/// # Errors
/// Returns an error if the picker is cancelled, the study cannot be loaded, or
/// the host cannot create a window or fails during its event loop.
#[cfg(windows)]
pub fn run_app() -> anyhow::Result<()> {
    run_app_with_options(AppLaunchOptions::default())
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

/// Launch the `ritk-snap` viewer on the Métis native host with startup options.
///
/// `initial_path` is opened by RITK before the interactive host starts. When
/// it is absent, the host opens a bounded native folder picker and passes the
/// selected path to RITK; a cancelled picker returns an error without starting
/// a window. `initial_series_uid` selects one acquisition after RITK discovery
/// when the path contains several series.
///
/// # Errors
/// Returns a host creation or event-loop error. With capture requested, also
/// returns an error if the initial study fails, the PNG cannot be saved, or
/// the window closes early.
#[cfg(windows)]
pub fn run_app_with_options(options: AppLaunchOptions) -> anyhow::Result<()> {
    let presentation_selection =
        NativePresentationSelection::Fixed(options.native_presentation_mode);
    run_native_selection(options, presentation_selection, None)
}

#[cfg(windows)]
fn run_native_selection(
    options: AppLaunchOptions,
    presentation_selection: NativePresentationSelection,
    comparison_series_uid: Option<String>,
) -> anyhow::Result<()> {
    use metis_platform::native::{pick, DialogSelection};

    let path = select_native_initial_path(options.initial_path.as_deref(), || {
        pick(DialogSelection::Folder).map_err(Into::into)
    })?;
    match presentation_selection {
        NativePresentationSelection::Fixed(mode) => {
            if let Some(comparison_uid) = comparison_series_uid.as_deref() {
                let primary_uid = options.initial_series_uid.as_deref().ok_or_else(|| {
                    anyhow::anyhow!("series comparison requires --series-instance-uid for panel 1")
                })?;
                crate::presentation::run_native_comparison_viewer(
                    &path,
                    primary_uid,
                    comparison_uid,
                    options.capture.as_deref(),
                    mode,
                    options.capture_application,
                )?;
            } else {
                crate::presentation::run_native_viewer(
                    &path,
                    options.initial_series_uid.as_deref(),
                    options.capture.as_deref(),
                    mode,
                    options.capture_application,
                )?;
            }
        }
        NativePresentationSelection::Responsive => {
            anyhow::ensure!(
                comparison_series_uid.is_none(),
                "series comparison currently requires a fixed native layout"
            );
            crate::presentation::run_native_responsive_viewer(
                &path,
                options.initial_series_uid.as_deref(),
                options.capture.as_deref(),
                options.capture_application,
            )?;
        }
    }
    Ok(())
}

/// Report that the desktop viewer has no host on this target.
///
/// On wasm targets, use `start_web` or `start_web_canvas` to launch
/// `ritk-snap` in a browser. Other native targets have no Métis surface
/// provider yet.
///
/// # Errors
/// Always returns the unsupported-target error.
#[cfg(not(windows))]
pub fn run_app_with_options(_options: AppLaunchOptions) -> anyhow::Result<()> {
    anyhow::bail!(
        "the ritk-snap desktop viewer requires a Windows Métis host; on wasm32 use start_web() or start_web_canvas()"
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
