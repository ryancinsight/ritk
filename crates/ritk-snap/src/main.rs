//! `ritk-snap` binary entry point.
//!
//! Launches the eframe/egui DICOM viewer application.

use std::path::PathBuf;

use clap::Parser;

/// Native RITK DICOM viewer.
#[derive(Debug, Parser)]
#[command(author, version, about)]
struct Args {
    /// Optional DICOM folder or medical image file loaded at startup.
    #[arg(value_name = "PATH")]
    initial_path: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    ritk_snap::run_app_with_options(ritk_snap::AppLaunchOptions {
        initial_path: args.initial_path,
    })
}
