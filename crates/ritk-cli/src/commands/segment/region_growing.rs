use anyhow::{anyhow, Context, Result};
use tracing::info;

use ritk_segmentation::{
    ConfidenceConnectedFilter, ConnectedThresholdFilter, NeighborhoodConnectedFilter,
};

use super::super::{
    infer_format, is_native_read_capable, is_native_write_capable, read_image_native,
    write_image_native, NativeBackend,
};
use super::args::SegmentArgs;
use super::helpers::parse_seed;

struct NativeRegionInput {
    image: ritk_image::native::Image<f32, NativeBackend, 3>,
    output_format: ritk_io::ImageFormat,
    seed: [usize; 3],
    lower: f32,
    upper: f32,
}

/// Parse, validate, and read the shared native region-growing inputs.
fn read_native_region_input(args: &SegmentArgs, method: &str) -> Result<NativeRegionInput> {
    let lower = args
        .lower
        .ok_or_else(|| anyhow!("--lower is required for {method}"))?;
    let upper = args
        .upper
        .ok_or_else(|| anyhow!("--upper is required for {method}"))?;
    if lower.is_nan() || upper.is_nan() {
        return Err(anyhow!(
            "{method}: bounds must not be NaN (lower={lower}, upper={upper})"
        ));
    }
    if lower > upper {
        return Err(anyhow!("--lower ({lower}) must be <= --upper ({upper})"));
    }
    let seed_text = args
        .seed
        .as_deref()
        .ok_or_else(|| anyhow!("--seed is required for {method} (format: Z,Y,X)"))?;
    let seed = parse_seed(seed_text).with_context(|| {
        format!("Failed to parse --seed '{seed_text}' for {method} (expected Z,Y,X integer format)")
    })?;
    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_native_read_capable(input_format) && is_native_write_capable(output_format),
        "{method} requires native input/output formats"
    );
    let image = read_image_native(&args.input)?;
    let shape = image.shape();
    anyhow::ensure!(
        seed[0] < shape[0] && seed[1] < shape[1] && seed[2] < shape[2],
        "Seed [{},{},{}] is out of bounds for image shape [{}x{}x{}]",
        seed[0],
        seed[1],
        seed[2],
        shape[0],
        shape[1],
        shape[2]
    );
    Ok(NativeRegionInput {
        image,
        output_format,
        seed,
        lower,
        upper,
    })
}

fn write_native_region_mask(
    args: &SegmentArgs,
    mask: &ritk_image::native::Image<f32, NativeBackend, 3>,
    format: ritk_io::ImageFormat,
) -> Result<usize> {
    let foreground = mask
        .data_slice()?
        .iter()
        .filter(|&&value| value > 0.5)
        .count();
    write_image_native(&args.output, mask, format)?;
    Ok(foreground)
}

// ── Connected-threshold region growing ───────────────────────────────────────

/// Apply connected-threshold BFS region growing from a user-specified seed.
///
/// Voxels reachable from `seed` whose intensities lie in `[lower, upper]`
/// are set to 1.0 (foreground); all others are set to 0.0 (background).
///
/// # Argument validation
/// `--lower`, `--upper`, and `--seed` are all required for this method.
///
/// # Errors
///
/// Returns an error for missing or invalid bounds/seed, unsupported native
/// formats, out-of-bounds seed coordinates, native I/O failures, or
/// non-host-addressable storage.
pub(super) fn run_connected_threshold(args: &SegmentArgs) -> Result<()> {
    let input = read_native_region_input(args, "connected-threshold")?;
    let backend = NativeBackend::default();
    let mask = ConnectedThresholdFilter::new(input.seed, input.lower, input.upper)
        .apply_native(&input.image, &backend)?;
    let n_foreground = write_native_region_mask(args, &mask, input.output_format)?;

    println!(
        "Segmented {}: found {} foreground voxels (seed=[{},{},{}], range=[{:.4},{:.4}])",
        args.input.display(),
        n_foreground,
        input.seed[0],
        input.seed[1],
        input.seed[2],
        input.lower,
        input.upper,
    );

    info!(
        "segment: connected-threshold complete input={} seed={:?} lower={} upper={} foreground={}",
        args.input.display(),
        input.seed,
        input.lower,
        input.upper,
        n_foreground
    );

    Ok(())
}

// -- Confidence-connected region growing --------------------------------------

/// Apply confidence-connected region growing through the native boundary.
///
/// # Errors
///
/// Returns an error for an invalid multiplier or any shared region-input,
/// native I/O, storage, or output-construction failure.
pub(super) fn run_confidence_connected(args: &SegmentArgs) -> Result<()> {
    anyhow::ensure!(
        args.multiplier.is_finite() && args.multiplier >= 0.0,
        "confidence-connected multiplier must be finite and non-negative, got {}",
        args.multiplier
    );
    let input = read_native_region_input(args, "confidence-connected")?;
    let filter = ConfidenceConnectedFilter::new(input.seed, input.lower, input.upper)
        .with_multiplier(args.multiplier)?
        .with_max_iterations(args.max_iterations);
    let backend = NativeBackend::default();
    let mask = filter.apply_native(&input.image, &backend)?;
    let n_foreground = write_native_region_mask(args, &mask, input.output_format)?;

    println!(
        "Segmented {}: confidence-connected found {} foreground voxels (seed=[{},{},{}], range=[{:.4},{:.4}], k={})",
        args.input.display(), n_foreground, input.seed[0], input.seed[1], input.seed[2], input.lower, input.upper, args.multiplier,
    );

    info!("segment: confidence-connected complete input={} seed={:?} lower={} upper={} multiplier={} foreground={}", args.input.display(), input.seed, input.lower, input.upper, args.multiplier, n_foreground);
    Ok(())
}

// -- Neighbourhood-connected region growing -----------------------------------

/// Apply neighborhood-connected region growing through the native boundary.
///
/// # Errors
///
/// Returns an error for any shared region-input, native I/O, storage, or
/// output-construction failure.
pub(super) fn run_neighborhood_connected(args: &SegmentArgs) -> Result<()> {
    let input = read_native_region_input(args, "neighborhood-connected")?;
    let r = args.neighborhood_radius;
    let filter = NeighborhoodConnectedFilter::new(input.seed, input.lower, input.upper)
        .with_radius([r, r, r]);
    let backend = NativeBackend::default();
    let mask = filter.apply_native(&input.image, &backend)?;
    let n_foreground = write_native_region_mask(args, &mask, input.output_format)?;

    println!(
        "Segmented {}: neighborhood-connected found {} foreground voxels (seed=[{},{},{}], range=[{:.4},{:.4}], radius={})",
        args.input.display(), n_foreground, input.seed[0], input.seed[1], input.seed[2], input.lower, input.upper, r,
    );

    info!("segment: neighborhood-connected complete input={} seed={:?} lower={} upper={} radius={} foreground={}", args.input.display(), input.seed, input.lower, input.upper, r, n_foreground);
    Ok(())
}
