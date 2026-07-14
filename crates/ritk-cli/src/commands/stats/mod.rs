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

use anyhow::Result;
use clap::Args;
use std::path::PathBuf;
use tracing::info;

mod metrics;

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Image quality and similarity metric to compute.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum StatMetric {
    Summary,
    Dice,
    Hausdorff,
    Psnr,
    Ssim,
    #[value(name = "mean-surface-distance", alias = "msd")]
    MeanSurfaceDistance,
    #[value(name = "noise-estimate")]
    NoiseEstimate,
}

impl std::fmt::Display for StatMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Summary => "summary",
            Self::Dice => "dice",
            Self::Hausdorff => "hausdorff",
            Self::Psnr => "psnr",
            Self::Ssim => "ssim",
            Self::MeanSurfaceDistance => "mean-surface-distance",
            Self::NoiseEstimate => "noise-estimate",
        })
    }
}

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
    #[arg(long)]
    pub metric: StatMetric,

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
pub fn run(args: StatsArgs) -> Result<()> {
    info!(
        "stats: starting input={} metric={}",
        args.input.display(),
        args.metric
    );

    match &args.metric {
        StatMetric::Summary => metrics::run_summary(&args),
        StatMetric::Dice => metrics::run_dice(&args),
        StatMetric::Hausdorff => metrics::run_hausdorff(&args),
        StatMetric::Psnr => metrics::run_psnr(&args),
        StatMetric::Ssim => metrics::run_ssim(&args),
        StatMetric::MeanSurfaceDistance => metrics::run_mean_surface_distance(&args),
        StatMetric::NoiseEstimate => metrics::run_noise_estimate(&args),
    }
}

#[cfg(test)]
mod tests;
