//! `ritk-snap` binary entry point.
//!
//! Launches the RITK DICOM viewer on the Métis native host.

// Mnemosyne as the process-wide allocator. Not applicable on WASM where
// the runtime provides its own allocator.
#[cfg(not(target_arch = "wasm32"))]
use mnemosyne::Mnemosyne;

#[cfg(not(target_arch = "wasm32"))]
#[global_allocator]
static ALLOC: Mnemosyne = Mnemosyne;

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use std::io::Write;

#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;

#[cfg(not(target_arch = "wasm32"))]
mod browser_trace;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
enum CliNativeLayout {
    Orthogonal,
    OrthogonalWithMip,
    OrthogonalWithMinip,
    OrthogonalWithAverage,
    Responsive,
}

#[cfg(windows)]
impl CliNativeLayout {
    const fn fixed_mode(self) -> Option<ritk_snap::NativePresentationMode> {
        match self {
            Self::Orthogonal => Some(ritk_snap::NativePresentationMode::Orthogonal),
            Self::OrthogonalWithMip => Some(ritk_snap::NativePresentationMode::OrthogonalWithMip),
            Self::OrthogonalWithMinip => {
                Some(ritk_snap::NativePresentationMode::OrthogonalWithMinip)
            }
            Self::OrthogonalWithAverage => {
                Some(ritk_snap::NativePresentationMode::OrthogonalWithAverage)
            }
            Self::Responsive => None,
        }
    }
}

/// Native RITK DICOM viewer.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// DICOM folder or medical image file. Without PATH, the native folder
    /// picker selects a study before the viewer opens.
    #[arg(value_name = "PATH")]
    initial_path: Option<PathBuf>,
    /// Select one SeriesInstanceUID after discovering the startup DICOM input.
    #[arg(
        long = "series-instance-uid",
        value_name = "UID",
        requires = "initial_path"
    )]
    initial_series_uid: Option<String>,
    /// Load a second series into the side-by-side comparison panel.
    #[arg(
        long = "compare-series-instance-uid",
        value_name = "UID",
        requires_all = ["initial_path", "initial_series_uid"]
    )]
    comparison_series_uid: Option<String>,
    /// Save the rendered application frame as PNG and exit.
    #[arg(long, value_name = "PNG")]
    capture: Option<PathBuf>,
    /// Include the bounded RITK application overlay in a Métis capture.
    #[arg(long, requires = "capture")]
    capture_application: bool,
    /// Select the native Métis framebuffer layout.
    #[arg(long = "layout", value_enum, default_value = "orthogonal")]
    native_presentation_mode: CliNativeLayout,
    /// Validate a RITK-owned semantic canvas trace and exit.
    #[arg(
        long,
        value_name = "JSON",
        conflicts_with_all = [
            "initial_path",
            "initial_series_uid",
            "comparison_series_uid",
            "capture"
        ]
    )]
    validate_browser_trace: Option<PathBuf>,
    /// Canvas identifiers in axial, coronal, sagittal order.
    #[arg(
        long = "canvas-id",
        value_name = "ID",
        action = clap::ArgAction::Append,
        requires = "validate_browser_trace"
    )]
    canvas_ids: Vec<String>,
    /// Require focused ArrowDown keydown/keyup evidence for every canvas.
    #[arg(
        long,
        requires = "validate_browser_trace",
        conflicts_with = "require_cine_rate"
    )]
    require_keyboard: bool,
    /// Require focused Equal/`data-ritk-cine-fps` rate evidence for every canvas.
    #[arg(
        long,
        requires = "validate_browser_trace",
        conflicts_with = "require_keyboard"
    )]
    require_cine_rate: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(path) = args.validate_browser_trace {
        let input_mode = if args.require_keyboard {
            browser_trace::TraceInputMode::PointerWheelKeyboard(
                browser_trace::KeyboardTraceKind::Navigation,
            )
        } else if args.require_cine_rate {
            browser_trace::TraceInputMode::PointerWheelKeyboard(
                browser_trace::KeyboardTraceKind::CineRate,
            )
        } else {
            browser_trace::TraceInputMode::PointerWheel
        };
        let report = browser_trace::validate_file(&path, &args.canvas_ids, input_mode)?;
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "{report}")?;
        return Ok(());
    }
    launch(args)
}

#[cfg(windows)]
fn launch(args: Args) -> anyhow::Result<()> {
    if let Some(comparison_series_uid) = args.comparison_series_uid {
        let native_presentation_mode =
            args.native_presentation_mode.fixed_mode().ok_or_else(|| {
                anyhow::anyhow!("series comparison currently requires a fixed native layout")
            })?;
        return ritk_snap::run_app_with_series_comparison(
            ritk_snap::AppLaunchOptions {
                initial_path: args.initial_path,
                initial_series_uid: args.initial_series_uid,
                capture: args.capture,
                capture_application: args.capture_application,
                native_presentation_mode,
            },
            comparison_series_uid,
        );
    }
    let options = ritk_snap::AppLaunchOptions {
        initial_path: args.initial_path,
        initial_series_uid: args.initial_series_uid,
        capture: args.capture,
        capture_application: args.capture_application,
        native_presentation_mode: args
            .native_presentation_mode
            .fixed_mode()
            .unwrap_or_default(),
    };
    if matches!(args.native_presentation_mode, CliNativeLayout::Responsive) {
        return ritk_snap::run_responsive_native_app_with_options(options);
    }
    ritk_snap::run_app_with_options(options)
}

#[cfg(all(not(target_arch = "wasm32"), not(windows)))]
fn launch(args: Args) -> anyhow::Result<()> {
    let Args { initial_path, .. } = args;
    ritk_snap::run_app_with_options(ritk_snap::AppLaunchOptions {
        initial_path,
        ..Default::default()
    })
}

#[cfg(all(not(target_arch = "wasm32"), test))]
mod tests {
    use super::Args;
    use clap::Parser;

    #[test]
    fn comparison_arguments_require_a_study_and_primary_series() {
        let args = Args::try_parse_from([
            "ritk-snap",
            "study",
            "--series-instance-uid",
            "2.25.1",
            "--compare-series-instance-uid",
            "2.25.2",
        ])
        .expect("parse the two-series launch");
        assert_eq!(args.initial_series_uid.as_deref(), Some("2.25.1"));
        assert_eq!(args.comparison_series_uid.as_deref(), Some("2.25.2"));

        assert!(Args::try_parse_from([
            "ritk-snap",
            "study",
            "--compare-series-instance-uid",
            "2.25.2",
        ])
        .is_err());
    }
}

#[cfg(target_arch = "wasm32")]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "ritk-snap binary is native-only on wasm32; use ritk_snap::start_web or start_web_canvas from JavaScript"
    )
}
