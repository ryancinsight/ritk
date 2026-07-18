//! `ritk-snap` binary entry point.
//!
//! Launches the eframe/egui DICOM viewer application.

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
    initial_path: Option<PathBuf> }

#[cfg(not(target_arch = "wasm32"))]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ritk_snap::run_app_with_options(ritk_snap::AppLaunchOptions {
        initial_path: args.initial_path })
}

#[cfg(target_arch = "wasm32")]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "ritk-snap binary is native-only on wasm32; use ritk_snap::start_web from JavaScript"
    )
}
