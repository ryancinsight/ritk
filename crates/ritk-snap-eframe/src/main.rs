//! eframe compatibility binary for `ritk-snap`.

#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[cfg(not(target_arch = "wasm32"))]
use clap::Parser;

/// Launch arguments for the eframe compatibility shell.
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
    /// Save the rendered eframe frame as PNG and exit.
    #[arg(long, value_name = "PNG")]
    capture: Option<PathBuf>,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ritk_snap::run_eframe_app_with_options(ritk_snap::AppLaunchOptions {
        initial_path: args.initial_path,
        initial_series_uid: args.initial_series_uid,
        capture: args.capture,
        capture_application: false,
        metis_native: false,
        native_presentation_mode: ritk_snap::NativePresentationMode::Orthogonal,
    })
}

#[cfg(target_arch = "wasm32")]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("ritk-snap-eframe is a native compatibility binary")
}
