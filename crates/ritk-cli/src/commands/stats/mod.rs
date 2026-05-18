//! `ritk stats` — image statistics and comparison metrics command.
//!
//! Computes single-image statistics or pairwise comparison metrics between
//! an input image and a reference image.
//!
//! | Metric | Requires `--reference` | Description |
//! |-------------|------------------------|--------------------------------------------|
//! | `summary` | No | Min, max, mean, std, percentiles |
//! | `dice` | Yes | Dice coefficient on binary masks (>0.5) |
//! | `hausdorff` | Yes | Hausdorff distance on binary masks |
//! | `psnr` | Yes | Peak signal-to-noise ratio |
//! | `ssim` | Yes | Structural similarity index |
//! | `mean-surface-distance` | Yes | Symmetric mean surface distance (mm) |
//! | `noise-estimate` | No | MAD-based Gaussian noise sigma estimate |

use anyhow::{anyhow, Result};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use super::{read_image, Backend};

mod metrics;

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `stats` subcommand.
#[derive(Args, Debug)]
pub struct StatsArgs {
    /// Input image path. Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Reference image path (required for comparison metrics: dice, hausdorff,
    /// psnr, ssim).
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// Metric to compute.
    ///
    /// Accepted values: `summary`, `dice`, `hausdorff`, `psnr`, `ssim`,
    /// `mean-surface-distance` (alias `msd`), `noise-estimate`.
    #[arg(long, value_name = "METRIC")]
    pub metric: String,

    /// Maximum possible pixel value, used by `psnr` and `ssim`.
    #[arg(long, default_value = "255.0", value_name = "FLOAT")]
    pub max_val: f32,
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `stats` subcommand.
///
/// Dispatches to the appropriate metric implementation based on `args.metric`.
///
/// # Errors
///
/// Returns an error when:
/// - The input or reference image cannot be read.
/// - A comparison metric is requested without `--reference`.
/// - An unknown metric name is supplied.
pub fn run(args: StatsArgs) -> Result<()> {
    info!(
        "stats: starting input={} metric={}",
        args.input.display(),
        args.metric
    );

    match args.metric.as_str() {
        "summary" => metrics::run_summary(&args),
        "dice" => metrics::run_dice(&args),
        "hausdorff" => metrics::run_hausdorff(&args),
        "psnr" => metrics::run_psnr(&args),
        "ssim" => metrics::run_ssim(&args),
        "mean-surface-distance" | "msd" => metrics::run_mean_surface_distance(&args),
        "noise-estimate" => metrics::run_noise_estimate(&args),
        other => Err(anyhow!(
            "Unknown metric '{other}'. \
             Available: summary, dice, hausdorff, psnr, ssim, mean-surface-distance, noise-estimate."
        )),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Load the reference image from `args.reference`, returning a descriptive
/// error when the path is absent.
pub(super) fn require_reference(args: &StatsArgs) -> Result<ritk_core::image::Image<Backend, 3>> {
    let ref_path = args
        .reference
        .as_ref()
        .ok_or_else(|| anyhow!("--reference is required for the '{}' metric", args.metric))?;
    read_image(ref_path)
}

#[cfg(test)]
mod tests;
