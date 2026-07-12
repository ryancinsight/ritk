use anyhow::{anyhow, Result};
use tracing::info;

use ritk_segmentation::{
    multi_otsu_threshold, AutoThreshold, BinaryThreshold, KapurThreshold, LiThreshold,
    MultiOtsuThreshold, OtsuThreshold, TriangleThreshold, YenThreshold,
};

use super::super::{
    infer_format, is_native_read_capable, is_native_write_capable, read_image, read_image_native,
    write_image_inferred, write_image_native, NativeBackend,
};
use super::args::SegmentArgs;
use super::helpers::count_foreground;

/// Apply one automatic-threshold strategy through the shared native boundary.
///
/// # Errors
///
/// Returns an error when either format is unknown or lacks native support,
/// native image I/O fails, or backend storage is not host-addressable.
fn apply_auto_threshold_native<A: AutoThreshold>(
    args: &SegmentArgs,
    algorithm: &A,
) -> Result<(f32, usize)> {
    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "automatic thresholding requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let (mask, threshold) = algorithm.apply_native_with_threshold(&image, &backend)?;
    let foreground = mask
        .data_slice()?
        .iter()
        .filter(|&&value| value > 0.5)
        .count();
    write_image_native(&args.output, &mask, output_format)?;
    Ok((threshold, foreground))
}

// ── Otsu thresholding ─────────────────────────────────────────────────────────

/// Apply single-threshold Otsu segmentation.
///
/// Computes the optimal threshold t* that maximises between-class variance,
/// then maps voxels ≥ t* to 1.0 (foreground) and voxels < t* to 0.0
/// (background).
pub(super) fn run_otsu(args: &SegmentArgs) -> Result<()> {
    let (threshold, n_foreground) = apply_auto_threshold_native(args, &OtsuThreshold::new())?;

    println!(
        "Segmented {}: found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        "segment: otsu complete input={} threshold={} foreground={}",
        args.input.display(),
        threshold,
        n_foreground
    );

    Ok(())
}

// ── Multi-Otsu thresholding ───────────────────────────────────────────────────

/// Apply multi-class Otsu segmentation with `args.classes` intensity classes.
///
/// Computes K−1 optimal thresholds and maps each voxel to the class label
/// (0.0, 1.0, …, K−1.0) whose intensity interval it falls into.
pub(super) fn run_multi_otsu(args: &SegmentArgs) -> Result<()> {
    if args.classes < 2 {
        return Err(anyhow!(
            "--classes must be ≥ 2 for multi-otsu, got {}",
            args.classes
        ));
    }

    let image = read_image(&args.input)?;

    let thresholds = multi_otsu_threshold::<super::super::Backend, 3>(&image, args.classes);
    let labeled = MultiOtsuThreshold::new(args.classes).apply(&image);

    let n_labeled = count_foreground(&labeled);

    write_image_inferred(&args.output, &labeled)?;

    let thresh_str: Vec<String> = thresholds.iter().map(|t| format!("{t:.4}")).collect();
    println!(
        "Segmented {}: found {} labeled voxels / thresholds=[{}]",
        args.input.display(),
        n_labeled,
        thresh_str.join(", "),
    );

    info!(
        "segment: multi-otsu complete input={} classes={} thresholds={:?} labeled={}",
        args.input.display(),
        args.classes,
        thresholds,
        n_labeled
    );

    Ok(())
}

// ── Li thresholding ───────────────────────────────────────────────────────────

/// Apply Li minimum cross-entropy thresholding.
///
/// Computes t* that minimises the cross-entropy between the image and its
/// binary thresholded version, then maps voxels ≥ t* to 1.0.
pub(super) fn run_li(args: &SegmentArgs) -> Result<()> {
    let (threshold, n_foreground) = apply_auto_threshold_native(args, &LiThreshold::new())?;

    println!(
        "Segmented {} (li): found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        "segment: li complete input={} threshold={} foreground={}",
        args.input.display(),
        threshold,
        n_foreground
    );

    Ok(())
}

// ── Yen thresholding ──────────────────────────────────────────────────────────

/// Apply Yen maximum correlation criterion thresholding.
///
/// Computes t* that maximises the correlation criterion, then maps
/// voxels ≥ t* to 1.0.
pub(super) fn run_yen(args: &SegmentArgs) -> Result<()> {
    let (threshold, n_foreground) = apply_auto_threshold_native(args, &YenThreshold::new())?;

    println!(
        "Segmented {} (yen): found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        "segment: yen complete input={} threshold={} foreground={}",
        args.input.display(),
        threshold,
        n_foreground
    );

    Ok(())
}

// ── Kapur thresholding ────────────────────────────────────────────────────────

/// Apply Kapur maximum entropy thresholding.
///
/// Computes t* that maximises the sum of foreground and background
/// entropies, then maps voxels ≥ t* to 1.0.
pub(super) fn run_kapur(args: &SegmentArgs) -> Result<()> {
    let (threshold, n_foreground) = apply_auto_threshold_native(args, &KapurThreshold::new())?;

    println!(
        "Segmented {} (kapur): found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        "segment: kapur complete input={} threshold={} foreground={}",
        args.input.display(),
        threshold,
        n_foreground
    );

    Ok(())
}

// ── Triangle thresholding ─────────────────────────────────────────────────────

/// Apply Triangle (geometric) thresholding.
///
/// Constructs a line between the histogram peak and the histogram tail,
/// then selects the bin with maximum perpendicular distance as the
/// threshold.  Maps voxels ≥ t* to 1.0.
pub(super) fn run_triangle(args: &SegmentArgs) -> Result<()> {
    let (threshold, n_foreground) = apply_auto_threshold_native(args, &TriangleThreshold::new())?;

    println!(
        "Segmented {} (triangle): found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        "segment: triangle complete input={} threshold={} foreground={}",
        args.input.display(),
        threshold,
        n_foreground
    );

    Ok(())
}

// ── Binary threshold segmentation ─────────────────────────────────────────────

/// Apply user-specified binary threshold segmentation.
///
/// Classifies voxels in `[lower, upper]` as foreground (1.0) and all others as
/// background (0.0).  Bounds default to `[-∞, +∞]` when not supplied, making
/// all finite voxels foreground.
///
/// # Mathematical Specification
///
/// S(v) = 1.0  if lower ≤ I(v) ≤ upper
/// S(v) = 0.0  otherwise
///
/// # Errors
/// Returns an error if either bound is NaN, `lower > upper`, the input or
/// output format is unknown or lacks native support, native image I/O fails,
/// or the backend storage is not available as a contiguous host slice.
pub(super) fn run_binary(args: &SegmentArgs) -> Result<()> {
    let lower = args.lower.unwrap_or(f32::NEG_INFINITY);
    let upper = args.upper.unwrap_or(f32::INFINITY);
    if lower.is_nan() || upper.is_nan() {
        return Err(anyhow!(
            "binary threshold: bounds must not be NaN (lower={lower}, upper={upper})"
        ));
    }
    if lower > upper {
        return Err(anyhow!(
            "binary threshold: lower ({lower}) must be ≤ upper ({upper})"
        ));
    }

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "binary threshold requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let backend = NativeBackend::default();
    let mask = BinaryThreshold::new(lower, upper).apply_native(&image, &backend)?;
    let n_foreground = mask
        .data_slice()?
        .iter()
        .filter(|&&value| value > 0.5)
        .count();
    write_image_native(&args.output, &mask, output_format)?;

    println!(
        "Segmented {} (binary): [{lower:.4}, {upper:.4}] → {n_foreground} foreground voxels",
        args.input.display(),
    );
    info!(
        "segment: binary complete input={} lower={lower} upper={upper} foreground={n_foreground}",
        args.input.display()
    );
    Ok(())
}
