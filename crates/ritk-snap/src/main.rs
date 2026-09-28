//! `ritk-snap` binary entry point.
//!
//! Launches the RITK DICOM viewer with the selected desktop host.

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

#[cfg(not(target_arch = "wasm32"))]
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
    /// Optional DICOM folder or medical image file loaded at startup.
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
    /// Run a study through the Métis native host. Without PATH, open the
    /// Windows native folder picker before loading the selected study. This is
    /// the default Windows shell; use `--eframe` for the compatibility shell.
    #[arg(long)]
    metis_native: bool,
    /// Use the legacy eframe compatibility shell instead of the default Métis
    /// host on Windows.
    #[cfg(feature = "eframe-shell")]
    #[arg(long, conflicts_with = "metis_native")]
    eframe: bool,
    /// Select the native Métis framebuffer layout.
    #[arg(long = "metis-native-layout", value_enum, default_value = "orthogonal")]
    native_presentation_mode: CliNativeLayout,
    /// Validate a RITK-owned semantic canvas trace and exit.
    #[arg(
        long,
        value_name = "JSON",
        conflicts_with_all = [
            "initial_path",
            "initial_series_uid",
            "comparison_series_uid",
            "capture",
            "metis_native"
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
    #[cfg(feature = "eframe-shell")]
    let eframe_requested = args.eframe;
    #[cfg(not(feature = "eframe-shell"))]
    let eframe_requested = false;
    let metis_native = !eframe_requested && (args.metis_native || cfg!(windows));
    if args.capture_application && !metis_native {
        anyhow::bail!("--capture-application requires the Métis native shell");
    }
    if !matches!(args.native_presentation_mode, CliNativeLayout::Orthogonal) && !metis_native {
        anyhow::bail!("native presentation layout requires the Métis native shell");
    }
    if let Some(comparison_series_uid) = args.comparison_series_uid {
        if !metis_native {
            anyhow::bail!("series comparison requires the Métis native shell");
        }
        if matches!(args.native_presentation_mode, CliNativeLayout::Responsive) {
            anyhow::bail!("series comparison currently requires a fixed native layout");
        }
        return ritk_snap::run_app_with_series_comparison(
            ritk_snap::AppLaunchOptions {
                initial_path: args.initial_path,
                initial_series_uid: args.initial_series_uid,
                capture: args.capture,
                capture_application: args.capture_application,
                metis_native,
                native_presentation_mode: args
                    .native_presentation_mode
                    .fixed_mode()
                    .ok_or_else(|| anyhow::anyhow!("comparison layout did not resolve"))?,
            },
            comparison_series_uid,
        );
    }
    if matches!(args.native_presentation_mode, CliNativeLayout::Responsive) {
        return ritk_snap::run_responsive_native_app_with_options(ritk_snap::AppLaunchOptions {
            initial_path: args.initial_path,
            initial_series_uid: args.initial_series_uid,
            capture: args.capture,
            capture_application: args.capture_application,
            metis_native,
            native_presentation_mode: ritk_snap::NativePresentationMode::Orthogonal,
        });
    }
    let Some(native_presentation_mode) = args.native_presentation_mode.fixed_mode() else {
        anyhow::bail!("responsive layout selection did not resolve to a fixed mode");
    };
    ritk_snap::run_app_with_options(ritk_snap::AppLaunchOptions {
        initial_path: args.initial_path,
        initial_series_uid: args.initial_series_uid,
        capture: args.capture,
        capture_application: args.capture_application,
        metis_native,
        native_presentation_mode,
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
