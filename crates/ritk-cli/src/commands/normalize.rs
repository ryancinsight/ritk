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

use ritk_core::statistics::normalization::{
    HistogramMatcher, MinMaxNormalizer, MriContrast, NyulUdupaNormalizer, WhiteStripeConfig,
    WhiteStripeNormalizer, ZScoreNormalizer,
};

use super::{read_image, write_image_inferred, Backend};

// ── CLI arguments ─────────────────────────────────────────────────────────────

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
    #[arg(long, value_name = "METHOD")]
    pub method: String,

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
    /// Accepted values: `t1`, `T1`, `t2`, `T2`.  Defaults to `t1` when not
    /// supplied.
    #[arg(long, value_name = "CONTRAST")]
    pub contrast: Option<String>,

    /// White stripe half-width as a fraction of the intensity range
    /// (default: 0.05).
    #[arg(long, value_name = "WIDTH")]
    pub ws_width: Option<f64>,
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
/// - An invalid contrast string is supplied to `white-stripe`.
/// - An unknown method name is supplied.
pub fn run(args: NormalizeArgs) -> Result<()> {
    info!(
        "normalize: starting input={} method={}",
        args.input.display(),
        args.method
    );

    let input = read_image(&args.input)?;

    let normalized: ritk_core::image::Image<Backend, 3> = match args.method.as_str() {
        "histogram-match" => {
            let ref_path = args.reference.as_ref().ok_or_else(|| {
                anyhow!("--reference is required for the 'histogram-match' method")
            })?;
            let reference = read_image(ref_path)?;
            HistogramMatcher::new(args.num_bins).match_histograms(&input, &reference)
        }

        "nyul" => {
            let mut normalizer = NyulUdupaNormalizer::default();
            if let Some(ref_path) = args.reference.as_ref() {
                let reference = read_image(ref_path)?;
                normalizer.learn_standard(&[&input, &reference]);
            } else {
                normalizer.learn_standard(&[&input]);
            }
            normalizer.apply(&input)?
        }

        "zscore" => ZScoreNormalizer::default().normalize(&input),

        "minmax" => MinMaxNormalizer::default().normalize(&input),

        "white-stripe" => {
            let contrast_str = args
                .contrast
                .as_deref()
                .unwrap_or("t1")
                .to_lowercase();
            let contrast = match contrast_str.as_str() {
                "t1" => MriContrast::T1,
                "t2" => MriContrast::T2,
                other => {
                    return Err(anyhow!(
                        "Unknown contrast '{}'. Accepted values: t1, t2.",
                        other
                    ))
                }
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

        other => {
            return Err(anyhow!(
                "Unknown normalization method '{other}'. \
                 Available: histogram-match, nyul, zscore, minmax, white-stripe."
            ))
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

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use std::path::Path;

    // ── Helper: build a 4×4×4 ramp NIfTI image (voxel i = i as f32) ──────────

    fn write_ramp_image(path: &Path) {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let vals: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let td = TensorData::new(vals, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        ritk_io::write_nifti(&path, &image).unwrap();
    }

    fn default_args(method: &str, input: PathBuf, output: PathBuf) -> NormalizeArgs {
        NormalizeArgs {
            input,
            output,
            method: method.to_string(),
            reference: None,
            num_bins: 256,
            contrast: None,
            ws_width: None,
        }
    }

    // ── zscore ────────────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_zscore_creates_output_file() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        let args = default_args("zscore", input, output.clone());
        run(args).unwrap();
        assert!(output.exists());
    }

    #[test]
    fn test_normalize_zscore_output_has_near_zero_mean() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        run(default_args("zscore", input, output.clone())).unwrap();
        let device: <Backend as BurnBackend>::Device = Default::default();
        let im: Image<Backend, 3> = ritk_io::read_nifti(&output, &device).unwrap();
        let vals: Vec<f32> = im
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let mean: f64 = vals.iter().map(|&v| v as f64).sum::<f64>() / vals.len() as f64;
        assert!(mean.abs() < 1e-4, "zscore mean must be ≈0, got {mean}");
    }

    // ── minmax ────────────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_minmax_output_in_zero_one() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        run(default_args("minmax", input, output.clone())).unwrap();
        let device: <Backend as BurnBackend>::Device = Default::default();
        let im: Image<Backend, 3> = ritk_io::read_nifti(&output, &device).unwrap();
        let vals: Vec<f32> = im
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();
        let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min >= -1e-5, "minmax output min must be >= 0, got {min}");
        assert!(max <= 1.0 + 1e-5, "minmax output max must be <= 1, got {max}");
    }

    // ── histogram-match ───────────────────────────────────────────────────────

    #[test]
    fn test_normalize_histogram_match_creates_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let reference = dir.path().join("ref.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        write_ramp_image(&reference);
        let args = NormalizeArgs {
            reference: Some(reference),
            ..default_args("histogram-match", input, output.clone())
        };
        run(args).unwrap();
        assert!(output.exists());
    }

    #[test]
    fn test_normalize_histogram_match_without_reference_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        let args = default_args("histogram-match", input, output);
        let result = run(args);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("reference"),
            "error must mention 'reference', got: {msg}"
        );
    }

    // ── nyul ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_nyul_creates_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        run(default_args("nyul", input, output.clone())).unwrap();
        assert!(output.exists());
    }

    #[test]
    fn test_normalize_nyul_with_reference_creates_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let reference = dir.path().join("ref.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        write_ramp_image(&reference);
        let args = NormalizeArgs {
            reference: Some(reference),
            ..default_args("nyul", input, output.clone())
        };
        run(args).unwrap();
        assert!(output.exists());
    }

    // ── error cases ───────────────────────────────────────────────────────────

    #[test]
    fn test_normalize_unknown_method_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        let args = default_args("unknown-method", input, output);
        let result = run(args);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unknown"),
            "error must mention 'Unknown', got: {msg}"
        );
    }

    #[test]
    fn test_normalize_white_stripe_invalid_contrast_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("in.nii.gz");
        let output = dir.path().join("out.nii.gz");
        write_ramp_image(&input);
        let args = NormalizeArgs {
            contrast: Some("flair".to_string()),
            ..default_args("white-stripe", input, output)
        };
        let result = run(args);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.to_lowercase().contains("contrast"),
            "error must mention 'contrast', got: {msg}"
        );
    }
}
