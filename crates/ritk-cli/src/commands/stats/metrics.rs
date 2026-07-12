// ── Metric implementations ───────────────────────────────────────────────────

use anyhow::Result;
use ritk_statistics::image_statistics::native::compute_statistics as compute_native_statistics;
use ritk_statistics::{
    dice_coefficient, estimate_noise_mad, hausdorff_distance, mean_surface_distance, psnr, ssim,
};
use tracing::info;

use super::{read_image, require_reference, StatsArgs};

// ── Summary ───────────────────────────────────────────────────────────────────

/// Print min, max, mean, std, and percentiles (p25, p50, p75) for the input
/// image.
pub(super) fn run_summary(args: &StatsArgs) -> Result<()> {
    let image = super::super::read_image_native(&args.input)?;
    let s = compute_native_statistics(&image)?;

    println!("Image statistics for {}:", args.input.display());
    println!("  min: {:.6}", s.min);
    println!("  max: {:.6}", s.max);
    println!("  mean: {:.6}", s.mean);
    println!("  std: {:.6}", s.std);
    println!("  p25: {:.6}", s.percentiles[0]);
    println!("  p50: {:.6}", s.percentiles[1]);
    println!("  p75: {:.6}", s.percentiles[2]);

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
/// Both images are thresholded at 0.5 before comparison. The input and
/// reference images are passed directly to `dice_coefficient` which treats
/// non-zero voxels as foreground.
pub(super) fn run_dice(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let (reference, ref_path) = require_reference(args)?;

    let value = dice_coefficient(&image, &reference);
    println!(
        "Dice coefficient: {:.6} (input={}, reference={})",
        value,
        args.input.display(),
        ref_path.display(),
    );
    info!(dice = value, "stats: dice complete");

    Ok(())
}

// ── Hausdorff distance ────────────────────────────────────────────────────────

/// Compute the Hausdorff distance between two binary masks.
///
/// Physical spacing from the input image is used for distance computation.
pub(super) fn run_hausdorff(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let (reference, ref_path) = require_reference(args)?;

    let sp = image.spacing();
    let spacing: [f64; 3] = [sp[0], sp[1], sp[2]];
    let value = hausdorff_distance(&image, &reference, &spacing);
    println!(
        "Hausdorff distance: {:.6} mm (input={}, reference={})",
        value,
        args.input.display(),
        ref_path.display(),
    );
    info!(hausdorff = value, "stats: hausdorff complete");

    Ok(())
}

// ── PSNR ──────────────────────────────────────────────────────────────────────

/// Compute Peak Signal-to-Noise Ratio between input and reference.
pub(super) fn run_psnr(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let (reference, ref_path) = require_reference(args)?;

    let value = psnr(&image, &reference, args.max_val);
    println!(
        "PSNR: {:.6} dB (max_val={}, input={}, reference={})",
        value,
        args.max_val,
        args.input.display(),
        ref_path.display(),
    );
    info!(psnr = value, max_val = args.max_val, "stats: psnr complete");

    Ok(())
}

// ── SSIM ──────────────────────────────────────────────────────────────────────

/// Compute the Structural Similarity Index between input and reference.
pub(super) fn run_ssim(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let (reference, ref_path) = require_reference(args)?;

    let value = ssim(&image, &reference, args.max_val);
    println!(
        "SSIM: {:.6} (max_val={}, input={}, reference={})",
        value,
        args.max_val,
        args.input.display(),
        ref_path.display(),
    );
    info!(ssim = value, max_val = args.max_val, "stats: ssim complete");

    Ok(())
}

// ── Mean surface distance ─────────────────────────────────────────────────────

/// Compute the symmetric mean surface distance between two binary masks.
///
/// Physical spacing from the input image is used for distance computation.
pub(super) fn run_mean_surface_distance(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let (reference, ref_path) = require_reference(args)?;

    let sp = image.spacing();
    let spacing: [f64; 3] = [sp[0], sp[1], sp[2]];
    let value = mean_surface_distance(&image, &reference, &spacing);
    println!(
        "Mean surface distance: {:.6} mm (input={}, reference={})",
        value,
        args.input.display(),
        ref_path.display(),
    );
    info!(msd = value, "stats: mean-surface-distance complete");

    Ok(())
}

// ── Noise estimate ────────────────────────────────────────────────────────────

/// Estimate Gaussian noise standard deviation using the MAD estimator.
///
/// Formula: sigma_hat = 1.4826 * MAD(I). No reference image required.
pub(super) fn run_noise_estimate(args: &StatsArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let sigma = estimate_noise_mad(&image);

    println!(
        "Noise estimate (MAD): sigma_hat = {:.6} (input={})",
        sigma,
        args.input.display(),
    );
    info!(sigma = sigma, "stats: noise-estimate complete");

    Ok(())
}
