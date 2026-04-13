//! RITK CLI entrypoint.
//!
//! Parses the command-line, initialises the `tracing` subscriber, and
//! dispatches each subcommand to its dedicated handler module.

use clap::{Parser, Subcommand};

mod commands;
mod error;

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
    /// Convert an image between supported formats (NIfTI, MetaImage, NRRD, PNG).
    Convert(commands::convert::ConvertArgs),

    /// Apply an image filter (gaussian, n4-bias, anisotropic, frangi).
    Filter(commands::filter::FilterArgs),

    /// Register two images using intensity-based or landmark-based methods.
    Register(commands::register::RegisterArgs),

    /// Segment an image (otsu, multi-otsu, connected-threshold).
    Segment(commands::segment::SegmentArgs),
}

fn main() -> anyhow::Result<()> {
    // Structured, filterable logging.  Set RUST_LOG=debug (or trace) to see spans.
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Convert(args) => commands::convert::run(args),
        Commands::Filter(args) => commands::filter::run(args),
        Commands::Register(args) => commands::register::run(args),
        Commands::Segment(args) => commands::segment::run(args),
    }
}
