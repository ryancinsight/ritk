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
use clap::Parser;

/// Native RITK DICOM viewer.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Optional DICOM folder or medical image file loaded at startup.
    #[arg(value_name = "PATH")]
    initial_path: Option<PathBuf>,
    /// Save the rendered application frame as PNG and exit.
    #[arg(long, value_name = "PNG")]
    capture: Option<PathBuf>,
    /// Run the loaded study through the Métis native host.
    #[arg(long)]
    metis_native: bool,
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ritk_snap::run_app_with_options(ritk_snap::AppLaunchOptions {
        initial_path: args.initial_path,
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
