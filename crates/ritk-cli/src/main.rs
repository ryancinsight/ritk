//! RITK CLI entrypoint.
//!
//! Parses the command-line, initialises the `tracing` subscriber, and
//! dispatches each subcommand to its dedicated handler module.

use clap::{Parser, Subcommand};

mod commands;

/// RITK — medical imaging toolkit command-line interface.
///
/// All subcommands propagate errors through `anyhow::Result<()>` so that
/// a single, structured error message is printed on failure and the process
/// exits with a non-zero status code.
#[derive(Parser, Debug)]
#[command(
    name = "ritk",
    about = "RITK medical imaging toolkit CLI",
    version,
    propagate_version = true
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Convert an image between supported formats (NIfTI, MetaImage, NRRD, PNG, MGH, TIFF).
    Convert(commands::convert::ConvertArgs),

    /// Inspect a DICOM study using the viewer core.
    Viewer(commands::viewer::ViewerArgs),

    /// Apply an image filter (gaussian, n4-bias, anisotropic, frangi, median, bilateral, canny, sobel, log, recursive-gaussian).
    Filter(commands::filter::FilterArgs),

    /// Register two images using intensity-based or deformable methods.
    Register(commands::register::RegisterArgs),

    /// Segment an image (otsu, multi-otsu, connected-threshold, li, yen, kapur, triangle, watershed, kmeans, distance-transform).
    Segment(commands::segment::SegmentArgs),

    /// Compute image statistics or comparison metrics.
    Stats(commands::stats::StatsArgs),

    /// Resample an image to a new voxel spacing using a configurable interpolation mode.
    Resample(commands::resample::ResampleArgs),

    /// Normalize image intensities (histogram-match, nyul, zscore, minmax, white-stripe).
    Normalize(commands::normalize::NormalizeArgs),
}

fn main() -> anyhow::Result<()> {
    // Structured, filterable logging.  Set RUST_LOG=debug (or trace) to see spans.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Convert(args) => commands::convert::run(args),
        Commands::Viewer(args) => commands::viewer::run(args),
        Commands::Filter(args) => commands::filter::run(args),
        Commands::Register(args) => commands::register::run(args),
        Commands::Segment(args) => commands::segment::run(args),
        Commands::Stats(args) => commands::stats::run(args),
        Commands::Resample(args) => commands::resample::run(args),
        Commands::Normalize(args) => commands::normalize::run(args),
    }
}
