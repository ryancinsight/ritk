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
    /// Save the rendered application frame as PNG and exit.
    #[arg(long, value_name = "PNG")]
    capture: Option<PathBuf>,
    /// Run the loaded study through the Métis native host.
    #[arg(long)]
    metis_native: bool,
    /// Validate a RITK-owned semantic canvas trace and exit.
    #[arg(
        long,
        value_name = "JSON",
        conflicts_with_all = ["initial_path", "initial_series_uid", "capture", "metis_native"]
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
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    if let Some(path) = args.validate_browser_trace {
        let report = browser_trace::validate_file(&path, &args.canvas_ids)?;
        let mut stdout = std::io::stdout().lock();
        writeln!(stdout, "{report}")?;
        return Ok(());
    }
    ritk_snap::run_app_with_options(ritk_snap::AppLaunchOptions {
        initial_path: args.initial_path,
        initial_series_uid: args.initial_series_uid,
        capture: args.capture,
        metis_native: args.metis_native,
    })
}

#[cfg(target_arch = "wasm32")]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "ritk-snap binary is native-only on wasm32; use ritk_snap::start_web or start_web_canvas from JavaScript"
    )
}
