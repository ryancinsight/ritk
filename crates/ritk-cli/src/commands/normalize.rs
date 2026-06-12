//! `ritk normalize` — image intensity normalization command.
//!
//! Applies one of five normalization strategies to a 3-D medical image:
//!
//! | Method             | Description                                        |
//! |--------------------|----------------------------------------------------|
//! | `histogram-match`  | CDF-based histogram matching to a reference image  |
//! | `nyul`             | Nyúl-Udupa piecewise-linear standardization        |
//! | `zscore`           | Zero-mean, unit-variance normalization             |
//! | `minmax`           | Rescale intensities to \[0, 1\]                    |
//! | `white-stripe`     | Brain MRI white-stripe normalization               |

use anyhow::{anyhow, Result};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use ritk_statistics::normalization::{
    HistogramMatcher, MinMaxNormalizer, MriContrast, NyulUdupaNormalizer, WhiteStripeConfig,
    WhiteStripeNormalizer, ZScoreNormalizer,
};

use super::{read_image, write_image_inferred, Backend};

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// MRI contrast type for white-stripe normalization.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum CliContrast {
    #[value(name = "t1")]
    T1,
    #[value(name = "t2")]
    T2,
}

/// Normalization method for the `normalize` subcommand.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum NormalizeMethod {
    #[value(name = "histogram-match")]
    HistogramMatch,
    Nyul,
    #[value(name = "zscore")]
    Zscore,
    #[value(name = "minmax")]
    Minmax,
    #[value(name = "white-stripe")]
    WhiteStripe,
}

impl std::fmt::Display for NormalizeMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::HistogramMatch => "histogram-match",
            Self::Nyul => "nyul",
            Self::Zscore => "zscore",
            Self::Minmax => "minmax",
            Self::WhiteStripe => "white-stripe",
        })
    }
}

/// Arguments for the `normalize` subcommand.
#[derive(Args, Debug)]
pub struct NormalizeArgs {
    /// Input image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Normalization method.
    ///
    /// Accepted values: `histogram-match`, `nyul`, `zscore`, `minmax`,
    /// `white-stripe`.
    #[arg(long, value_enum, value_name = "METHOD")]
    pub method: NormalizeMethod,

    /// Reference image path.
    ///
    /// Required for `histogram-match`.  When provided with `nyul`, both the
    /// input and reference images are used together to learn the standard
    /// landmarks.
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// Number of histogram bins used by `histogram-match` (default: 256).
    #[arg(long, default_value = "256", value_name = "N")]
    pub num_bins: usize,

    /// MRI contrast type for `white-stripe`.
    ///
    /// Accepted values: `t1`, `t2`.  Defaults to `t1` when not
    /// supplied.
    #[arg(long, value_enum, value_name = "CONTRAST")]
    pub contrast: Option<CliContrast>,

    /// White stripe half-width as a fraction of the intensity range
    /// (default: 0.05).
    #[arg(long, value_name = "WIDTH")]
    pub ws_width: Option<f64>,

    /// Optional binary mask image path for masked Z-score normalization.
    ///
    /// Only used with `--method zscore`. When supplied, μ and σ are computed
    /// from foreground voxels (mask > 0.5); all voxels are still transformed.
    /// If the mask contains no foreground voxels, the method falls back to
    /// full-image statistics.
    #[arg(long, value_name = "MASK")]
    pub mask: Option<PathBuf>,
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `normalize` subcommand.
///
/// Dispatches to the appropriate normalization implementation based on
/// `args.method`, writes the result to `args.output`, and prints one line of
/// progress to stdout.
///
/// # Errors
/// Returns an error when:
/// - The input or reference image cannot be read.
/// - `histogram-match` is requested without `--reference`.
/// - An unknown method name is supplied.
pub fn run(args: NormalizeArgs) -> Result<()> {
    info!(
        "normalize: starting input={} method={}",
        args.input.display(),
        args.method
    );

    let input = read_image(&args.input)?;

    let normalized: ritk_image::Image<Backend, 3> = match args.method {
        NormalizeMethod::HistogramMatch => {
            let ref_path = args.reference.as_ref().ok_or_else(|| {
                anyhow!("--reference is required for the 'histogram-match' method")
            })?;
            let reference = read_image(ref_path)?;
            HistogramMatcher::new(args.num_bins).match_histograms(&input, &reference)
        }

        NormalizeMethod::Nyul => {
            let mut normalizer = NyulUdupaNormalizer::default();
            if let Some(ref_path) = args.reference.as_ref() {
                let reference = read_image(ref_path)?;
                normalizer.learn_standard(&[&input, &reference]);
            } else {
                normalizer.learn_standard(&[&input]);
            }
            normalizer.apply(&input)?
        }

        NormalizeMethod::Zscore => {
            if let Some(mask_path) = &args.mask {
                let mask_img = read_image(mask_path)?;
                ZScoreNormalizer::new().normalize_masked(&input, &mask_img)
            } else {
                ZScoreNormalizer.normalize(&input)
            }
        }

        NormalizeMethod::Minmax => MinMaxNormalizer::default().normalize(&input),

        NormalizeMethod::WhiteStripe => {
            let contrast = match args.contrast.unwrap_or(CliContrast::T1) {
                CliContrast::T1 => MriContrast::T1,
                CliContrast::T2 => MriContrast::T2,
            };
            let config = WhiteStripeConfig {
                contrast,
                width: args.ws_width.unwrap_or(0.05),
                ..Default::default()
            };
            let result = WhiteStripeNormalizer::normalize(&input, None, &config);
            info!(
                "white-stripe: mu={:.4} sigma={:.4} wm_peak={:.4} stripe_size={}",
                result.mu, result.sigma, result.wm_peak, result.stripe_size
            );
            println!(
                "white-stripe stats: mu={:.4}, sigma={:.4}, wm_peak={:.4}, stripe_size={}",
                result.mu, result.sigma, result.wm_peak, result.stripe_size
            );
            result.normalized
        }
    };

    write_image_inferred(&args.output, &normalized)?;
    println!(
        "normalize: wrote {} -> {}",
        args.method,
        args.output.display()
    );
    info!("normalize: done output={}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests_normalize;
