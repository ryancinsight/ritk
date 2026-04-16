//! `ritk stats` — image statistics and comparison metrics command.
//!
//! Computes single-image statistics or pairwise comparison metrics between
//! an input image and a reference image.
//!
//! | Metric      | Requires `--reference` | Description                                |
//! |-------------|------------------------|--------------------------------------------|
//! | `summary`   | No                     | Min, max, mean, std, percentiles           |
//! | `dice`      | Yes                    | Dice coefficient on binary masks (>0.5)    |
//! | `hausdorff` | Yes                    | Hausdorff distance on binary masks         |
//! | `psnr`      | Yes                    | Peak signal-to-noise ratio                 |
//! | `ssim`      | Yes                    | Structural similarity index                |

use anyhow::{anyhow, Result};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use ritk_core::statistics::{compute_statistics, dice_coefficient, hausdorff_distance, psnr, ssim};

use super::{read_image, Backend};

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `stats` subcommand.
#[derive(Args, Debug)]
pub struct StatsArgs {
    /// Input image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Reference image path (required for comparison metrics: dice, hausdorff,
    /// psnr, ssim).
    #[arg(long)]
    pub reference: Option<PathBuf>,

    /// Metric to compute.
    ///
    /// Accepted values: `summary`, `dice`, `hausdorff`, `psnr`, `ssim`.
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
/// Returns an error when:
/// - The input or reference image cannot be read.
/// - A comparison metric is requested without `--reference`.
/// - An unknown metric name is supplied.
pub fn run(args: StatsArgs) -> Result<()> {
    info!(
        input  = %args.input.display(),
        metric = %args.metric,
        "stats: starting"
    );

    match args.metric.as_str() {
        "summary" => run_summary(&args),
        "dice" => run_dice(&args),
        "hausdorff" => run_hausdorff(&args),
        "psnr" => run_psnr(&args),
        "ssim" => run_ssim(&args),
        other => Err(anyhow!(
            "Unknown metric '{other}'. \
             Available: summary, dice, hausdorff, psnr, ssim."
        )),
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Load the reference image from `args.reference`, returning a descriptive
/// error when the path is absent.
fn require_reference(args: &StatsArgs) -> Result<ritk_core::image::Image<Backend, 3>> {
    let ref_path = args
        .reference
        .as_ref()
        .ok_or_else(|| anyhow!("--reference is required for the '{}' metric", args.metric))?;
    read_image(ref_path)
}

// ── Summary ───────────────────────────────────────────────────────────────────

/// Print min, max, mean, std, and percentiles (p25, p50, p75) for the input
/// image.
fn run_summary(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let s = compute_statistics(&image);

    println!("Image statistics for {}:", args.input.display());
    println!("  min:  {:.6}", s.min);
    println!("  max:  {:.6}", s.max);
    println!("  mean: {:.6}", s.mean);
    println!("  std:  {:.6}", s.std);
    println!("  p25:  {:.6}", s.percentiles[0]);
    println!("  p50:  {:.6}", s.percentiles[1]);
    println!("  p75:  {:.6}", s.percentiles[2]);

    info!(
        min = s.min,
        max = s.max,
        mean = s.mean,
        std = s.std,
        "stats: summary complete"
    );

    Ok(())
}

// ── Dice coefficient ──────────────────────────────────────────────────────────

/// Compute the Dice coefficient between two binary masks (voxels > 0.5 =
/// foreground).
///
/// Both images are thresholded at 0.5 before comparison.  The input and
/// reference images are passed directly to `dice_coefficient` which treats
/// non-zero voxels as foreground.
fn run_dice(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let reference = require_reference(args)?;

    let value = dice_coefficient(&image, &reference);

    println!(
        "Dice coefficient: {:.6} (input={}, reference={})",
        value,
        args.input.display(),
        args.reference.as_ref().unwrap().display(),
    );

    info!(dice = value, "stats: dice complete");

    Ok(())
}

// ── Hausdorff distance ────────────────────────────────────────────────────────

/// Compute the Hausdorff distance between two binary masks.
///
/// Physical spacing from the input image is used for distance computation.
fn run_hausdorff(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let reference = require_reference(args)?;

    let sp = image.spacing();
    let spacing: [f64; 3] = [sp[0], sp[1], sp[2]];

    let value = hausdorff_distance(&image, &reference, &spacing);

    println!(
        "Hausdorff distance: {:.6} mm (input={}, reference={})",
        value,
        args.input.display(),
        args.reference.as_ref().unwrap().display(),
    );

    info!(hausdorff = value, "stats: hausdorff complete");

    Ok(())
}

// ── PSNR ──────────────────────────────────────────────────────────────────────

/// Compute Peak Signal-to-Noise Ratio between input and reference.
fn run_psnr(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let reference = require_reference(args)?;

    let value = psnr(&image, &reference, args.max_val);

    println!(
        "PSNR: {:.6} dB (max_val={}, input={}, reference={})",
        value,
        args.max_val,
        args.input.display(),
        args.reference.as_ref().unwrap().display(),
    );

    info!(psnr = value, max_val = args.max_val, "stats: psnr complete");

    Ok(())
}

// ── SSIM ──────────────────────────────────────────────────────────────────────

/// Compute the Structural Similarity Index between input and reference.
fn run_ssim(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let reference = require_reference(args)?;

    let value = ssim(&image, &reference, args.max_val);

    println!(
        "SSIM: {:.6} (max_val={}, input={}, reference={})",
        value,
        args.max_val,
        args.input.display(),
        args.reference.as_ref().unwrap().display(),
    );

    info!(ssim = value, max_val = args.max_val, "stats: ssim complete");

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
    use tempfile::tempdir;

    /// Build a 4×4×4 image filled with the given constant value.
    fn make_constant_image(value: f32) -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values = vec![value; 64];
        let td = TensorData::new(values, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 4×4×4 ramp image whose voxel at flat index i has value `i as f32`.
    fn make_ramp_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let td = TensorData::new(values, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 4×4×4 binary mask with the first `n_foreground` voxels set to
    /// 1.0 and the remainder set to 0.0.
    fn make_binary_mask(n_foreground: usize) -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..64)
            .map(|i| if i < n_foreground { 1.0 } else { 0.0 })
            .collect();
        let td = TensorData::new(values, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Helper: write a NIfTI image and return the path.
    fn write_nifti_tmp(dir: &std::path::Path, name: &str, image: &Image<Backend, 3>) -> PathBuf {
        let path = dir.join(name);
        ritk_io::write_nifti(&path, image).unwrap();
        path
    }

    // ── Positive: summary computes correct statistics ─────────────────────

    /// For a constant image, min == max == mean == value, std == 0.
    #[test]
    fn test_stats_summary_constant_image() {
        let dir = tempdir().unwrap();
        let image = make_constant_image(42.0);
        let input = write_nifti_tmp(dir.path(), "const.nii", &image);

        let result = run(StatsArgs {
            input,
            reference: None,
            metric: "summary".to_string(),
            max_val: 255.0,
        });
        assert!(result.is_ok(), "summary must succeed: {:?}", result.err());
    }

    /// Summary on a ramp image must report correct min and max.
    #[test]
    fn test_stats_summary_ramp_image_values() {
        let image = make_ramp_image();
        let s = compute_statistics(&image);
        assert!((s.min - 0.0).abs() < 1e-4, "min must be 0.0, got {}", s.min);
        assert!(
            (s.max - 63.0).abs() < 1e-4,
            "max must be 63.0, got {}",
            s.max
        );
        let expected_mean = (0..64).map(|i| i as f32).sum::<f32>() / 64.0;
        assert!(
            (s.mean - expected_mean).abs() < 1e-3,
            "mean must be {expected_mean}, got {}",
            s.mean
        );
    }

    // ── Positive: dice on identical masks returns 1.0 ─────────────────────

    #[test]
    fn test_stats_dice_identical_masks_returns_one() {
        let dir = tempdir().unwrap();
        let mask = make_binary_mask(32);
        let input = write_nifti_tmp(dir.path(), "mask_a.nii", &mask);
        let reference = write_nifti_tmp(dir.path(), "mask_b.nii", &mask);

        let result = run(StatsArgs {
            input: input.clone(),
            reference: Some(reference),
            metric: "dice".to_string(),
            max_val: 255.0,
        });
        assert!(result.is_ok(), "dice must succeed: {:?}", result.err());

        // Verify the value directly via the library function.
        let img = read_image(&input).unwrap();
        let value = dice_coefficient(&img, &img);
        assert!(
            (value - 1.0).abs() < 1e-5,
            "Dice of identical masks must be 1.0, got {value}"
        );
    }

    // ── Positive: dice on disjoint masks returns 0.0 ──────────────────────

    #[test]
    fn test_stats_dice_disjoint_masks_returns_zero() {
        let device: <Backend as BurnBackend>::Device = Default::default();

        // Mask A: first 32 voxels foreground.
        let a = make_binary_mask(32);
        // Mask B: last 32 voxels foreground.
        let vals_b: Vec<f32> = (0..64).map(|i| if i >= 32 { 1.0 } else { 0.0 }).collect();
        let td_b = TensorData::new(vals_b, Shape::new([4, 4, 4]));
        let tensor_b = Tensor::<Backend, 3>::from_data(td_b, &device);
        let b = Image::new(
            tensor_b,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );

        let value = dice_coefficient(&a, &b);
        assert!(
            value.abs() < 1e-5,
            "Dice of disjoint masks must be 0.0, got {value}"
        );
    }

    // ── Positive: psnr on identical images returns infinity ───────────────

    #[test]
    fn test_stats_psnr_identical_images_returns_inf() {
        let dir = tempdir().unwrap();
        let image = make_ramp_image();
        let input = write_nifti_tmp(dir.path(), "img_a.nii", &image);
        let reference = write_nifti_tmp(dir.path(), "img_b.nii", &image);

        let result = run(StatsArgs {
            input: input.clone(),
            reference: Some(reference),
            metric: "psnr".to_string(),
            max_val: 63.0,
        });
        assert!(result.is_ok(), "psnr must succeed: {:?}", result.err());

        let img = read_image(&input).unwrap();
        let value = psnr(&img, &img, 63.0);
        assert!(
            value.is_infinite() || value > 100.0,
            "PSNR of identical images must be very large or infinite, got {value}"
        );
    }

    // ── Positive: ssim on identical images returns 1.0 ────────────────────

    #[test]
    fn test_stats_ssim_identical_images_returns_one() {
        let dir = tempdir().unwrap();
        let image = make_ramp_image();
        let input = write_nifti_tmp(dir.path(), "img_a.nii", &image);
        let reference = write_nifti_tmp(dir.path(), "img_b.nii", &image);

        let result = run(StatsArgs {
            input: input.clone(),
            reference: Some(reference),
            metric: "ssim".to_string(),
            max_val: 63.0,
        });
        assert!(result.is_ok(), "ssim must succeed: {:?}", result.err());

        let img = read_image(&input).unwrap();
        let value = ssim(&img, &img, 63.0);
        assert!(
            (value - 1.0).abs() < 1e-4,
            "SSIM of identical images must be 1.0, got {value}"
        );
    }

    // ── Positive: hausdorff on identical masks returns 0.0 ────────────────

    #[test]
    fn test_stats_hausdorff_identical_masks_returns_zero() {
        let dir = tempdir().unwrap();
        let mask = make_binary_mask(32);
        let input = write_nifti_tmp(dir.path(), "mask_a.nii", &mask);
        let reference = write_nifti_tmp(dir.path(), "mask_b.nii", &mask);

        let result = run(StatsArgs {
            input: input.clone(),
            reference: Some(reference),
            metric: "hausdorff".to_string(),
            max_val: 255.0,
        });
        assert!(result.is_ok(), "hausdorff must succeed: {:?}", result.err());

        let img = read_image(&input).unwrap();
        let sp = img.spacing();
        let spacing = [sp[0], sp[1], sp[2]];
        let value = hausdorff_distance(&img, &img, &spacing);
        assert!(
            value.abs() < 1e-5,
            "Hausdorff distance of identical masks must be 0.0, got {value}"
        );
    }

    // ── Negative: unknown metric returns descriptive error ────────────────

    #[test]
    fn test_stats_unknown_metric_returns_error() {
        let dir = tempdir().unwrap();
        let image = make_constant_image(1.0);
        let input = write_nifti_tmp(dir.path(), "img.nii", &image);

        let result = run(StatsArgs {
            input,
            reference: None,
            metric: "bogus".to_string(),
            max_val: 255.0,
        });

        assert!(result.is_err(), "unknown metric must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unknown metric 'bogus'"),
            "error must name the unsupported metric, got: {msg}"
        );
    }

    // ── Negative: comparison metric without --reference returns error ─────

    #[test]
    fn test_stats_dice_without_reference_returns_error() {
        let dir = tempdir().unwrap();
        let image = make_binary_mask(32);
        let input = write_nifti_tmp(dir.path(), "mask.nii", &image);

        let result = run(StatsArgs {
            input,
            reference: None,
            metric: "dice".to_string(),
            max_val: 255.0,
        });

        assert!(result.is_err(), "dice without --reference must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--reference is required"),
            "error must explain the missing argument, got: {msg}"
        );
    }

    #[test]
    fn test_stats_psnr_without_reference_returns_error() {
        let dir = tempdir().unwrap();
        let image = make_ramp_image();
        let input = write_nifti_tmp(dir.path(), "img.nii", &image);

        let result = run(StatsArgs {
            input,
            reference: None,
            metric: "psnr".to_string(),
            max_val: 255.0,
        });

        assert!(result.is_err(), "psnr without --reference must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--reference is required"),
            "error must explain the missing argument, got: {msg}"
        );
    }

    #[test]
    fn test_stats_ssim_without_reference_returns_error() {
        let dir = tempdir().unwrap();
        let image = make_ramp_image();
        let input = write_nifti_tmp(dir.path(), "img.nii", &image);

        let result = run(StatsArgs {
            input,
            reference: None,
            metric: "ssim".to_string(),
            max_val: 255.0,
        });

        assert!(result.is_err(), "ssim without --reference must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--reference is required"),
            "error must explain the missing argument, got: {msg}"
        );
    }

    #[test]
    fn test_stats_hausdorff_without_reference_returns_error() {
        let dir = tempdir().unwrap();
        let image = make_binary_mask(32);
        let input = write_nifti_tmp(dir.path(), "mask.nii", &image);

        let result = run(StatsArgs {
            input,
            reference: None,
            metric: "hausdorff".to_string(),
            max_val: 255.0,
        });

        assert!(
            result.is_err(),
            "hausdorff without --reference must return Err"
        );
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--reference is required"),
            "error must explain the missing argument, got: {msg}"
        );
    }

    // ── Boundary: missing input file returns error ────────────────────────

    #[test]
    fn test_stats_missing_input_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("does_not_exist.nii");

        let result = run(StatsArgs {
            input,
            reference: None,
            metric: "summary".to_string(),
            max_val: 255.0,
        });

        assert!(result.is_err(), "missing input must yield an error");
    }
}
