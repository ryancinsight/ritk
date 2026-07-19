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

use super::{infer_format, is_read_capable, is_write_capable, read_image, write_image};

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

    if matches!(
        args.method,
        NormalizeMethod::HistogramMatch
            | NormalizeMethod::Minmax
            | NormalizeMethod::Nyul
            | NormalizeMethod::WhiteStripe
            | NormalizeMethod::Zscore
    ) {
        let input_format = infer_format(&args.input).ok_or_else(|| {
            anyhow!(
                "Cannot infer input format from path: {}",
                args.input.display()
            )
        })?;
        let output_format = infer_format(&args.output).ok_or_else(|| {
            anyhow!(
                "Cannot infer output format from path: {}",
                args.output.display()
            )
        })?;
        anyhow::ensure!(
            is_read_capable(input_format),
            "{} normalization does not support {:?} input until its native reader exists",
            args.method,
            input_format
        );
        anyhow::ensure!(
            is_write_capable(output_format),
            "{} normalization does not support {:?} output until its native writer exists",
            args.method,
            output_format
        );
        let input = read_image(&args.input)?;
        let output = match args.method {
            NormalizeMethod::Minmax => MinMaxNormalizer::default().normalize_native(&input)?,
            NormalizeMethod::Zscore => {
                if let Some(mask_path) = &args.mask {
                    let mask_format = infer_format(mask_path).ok_or_else(|| {
                        anyhow!(
                            "Cannot infer mask format from path: {}",
                            mask_path.display()
                        )
                    })?;
                    anyhow::ensure!(
                        is_read_capable(mask_format),
                        "zscore normalization does not support {:?} mask input until its native reader exists",
                        mask_format
                    );
                    let mask = read_image(mask_path)?;
                    ZScoreNormalizer::new().normalize_masked_native(&input, &mask)?
                } else {
                    ZScoreNormalizer::new().normalize_native(&input)?
                }
            }
            NormalizeMethod::HistogramMatch => {
                let reference_path = args.reference.as_ref().ok_or_else(|| {
                    anyhow!("--reference is required for the 'histogram-match' method")
                })?;
                let reference_format = infer_format(reference_path).ok_or_else(|| {
                    anyhow!(
                        "Cannot infer reference format from path: {}",
                        reference_path.display()
                    )
                })?;
                anyhow::ensure!(
                    is_read_capable(reference_format),
                    "histogram-match normalization does not support {:?} reference input until its native reader exists",
                    reference_format
                );
                let reference = read_image(reference_path)?;
                HistogramMatcher::new(args.num_bins).match_histograms_native(&input, &reference)?
            }
            NormalizeMethod::Nyul => {
                let mut normalizer = NyulUdupaNormalizer::default();
                if let Some(reference_path) = &args.reference {
                    let reference_format = infer_format(reference_path).ok_or_else(|| {
                        anyhow!(
                            "Cannot infer reference format from path: {}",
                            reference_path.display()
                        )
                    })?;
                    anyhow::ensure!(
                        is_read_capable(reference_format),
                        "nyul normalization does not support {:?} reference input until its native reader exists",
                        reference_format
                    );
                    let reference = read_image(reference_path)?;
                    normalizer.learn_standard_native(&[&input, &reference])?;
                } else {
                    normalizer.learn_standard_native(&[&input])?;
                }
                normalizer.apply_native(&input)?
            }
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
                let result = WhiteStripeNormalizer::normalize_native(&input, None, &config)?;
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
        write_image(&args.output, &output, output_format)?;
        println!(
            "normalize: wrote {} -> {}",
            args.method,
            args.output.display()
        );
        info!("normalize: done output={}", args.output.display());
        Ok(())
    } else {
        unreachable!("all normalization methods have native routes")
    }
}

#[cfg(test)]
mod tests_normalize;
