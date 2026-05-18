use anyhow::{anyhow, Context, Result};
use tracing::info;

use ritk_core::segmentation::connected_threshold;

use super::super::{read_image, write_image_inferred};
use super::args::SegmentArgs;
use super::helpers::{count_foreground, parse_seed};

// ── Connected-threshold region growing ───────────────────────────────────────

/// Apply connected-threshold BFS region growing from a user-specified seed.
///
/// Voxels reachable from `seed` whose intensities lie in `[lower, upper]`
/// are set to 1.0 (foreground); all others are set to 0.0 (background).
///
/// # Argument validation
/// `--lower`, `--upper`, and `--seed` are all required for this method.
pub(super) fn run_connected_threshold(args: &SegmentArgs) -> Result<()> {
    let lower = args
        .lower
        .ok_or_else(|| anyhow!("--lower is required for the connected-threshold method"))?;
    let upper = args
        .upper
        .ok_or_else(|| anyhow!("--upper is required for the connected-threshold method"))?;
    let seed_str = args.seed.as_deref().ok_or_else(|| {
        anyhow!("--seed is required for the connected-threshold method (format: Z,Y,X)")
    })?;

    if lower > upper {
        return Err(anyhow!(
            "--lower ({lower}) must be \u{2264} --upper ({upper})"
        ));
    }

    let seed = parse_seed(seed_str).with_context(|| {
        format!("Failed to parse --seed '{seed_str}' (expected Z,Y,X integer format)")
    })?;

    let image = read_image(&args.input)?;

    let shape = image.shape();
    if seed[0] >= shape[0] || seed[1] >= shape[1] || seed[2] >= shape[2] {
        return Err(anyhow!(
            "Seed [{},{},{}] is out of bounds for image shape [{}×{}×{}]",
            seed[0],
            seed[1],
            seed[2],
            shape[0],
            shape[1],
            shape[2],
        ));
    }

    let mask = connected_threshold::<super::super::Backend>(&image, seed, lower, upper);
    let n_foreground = count_foreground(&mask);

    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: found {} foreground voxels (seed=[{},{},{}], range=[{:.4},{:.4}])",
        args.input.display(),
        n_foreground,
        seed[0],
        seed[1],
        seed[2],
        lower,
        upper,
    );

    info!(
        "segment: connected-threshold complete input={} seed={:?} lower={} upper={} foreground={}",
        args.input.display(),
        seed,
        lower,
        upper,
        n_foreground
    );

    Ok(())
}

// -- Confidence-connected region growing --------------------------------------

pub(super) fn run_confidence_connected(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::ConfidenceConnectedFilter;

    let lower = args
        .lower
        .ok_or_else(|| anyhow!("--lower is required for confidence-connected"))?;
    let upper = args
        .upper
        .ok_or_else(|| anyhow!("--upper is required for confidence-connected"))?;
    if lower > upper {
        return Err(anyhow!("--lower ({lower}) must be <= --upper ({upper})"));
    }
    let seed_str = args
        .seed
        .as_deref()
        .ok_or_else(|| anyhow!("--seed is required for confidence-connected (format: Z,Y,X)"))?;
    let seed =
        parse_seed(seed_str).with_context(|| format!("Failed to parse --seed '{seed_str}'"))?;

    let image = read_image(&args.input)?;
    let shape = image.shape();
    if seed[0] >= shape[0] || seed[1] >= shape[1] || seed[2] >= shape[2] {
        return Err(anyhow!(
            "Seed [{},{},{}] is out of bounds for image shape [{}x{}x{}]",
            seed[0],
            seed[1],
            seed[2],
            shape[0],
            shape[1],
            shape[2],
        ));
    }

    let filter = ConfidenceConnectedFilter::new(seed, lower, upper)
        .with_multiplier(args.multiplier)
        .with_max_iterations(args.max_iterations);
    let mask = filter.apply(&image);
    let n_foreground = count_foreground(&mask);

    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: confidence-connected found {} foreground voxels (seed=[{},{},{}], range=[{:.4},{:.4}], k={})",
        args.input.display(), n_foreground, seed[0], seed[1], seed[2], lower, upper, args.multiplier,
    );

    info!("segment: confidence-connected complete input={} seed={:?} lower={} upper={} multiplier={} foreground={}", args.input.display(), seed, lower, upper, args.multiplier, n_foreground);
    Ok(())
}

// -- Neighbourhood-connected region growing -----------------------------------

pub(super) fn run_neighborhood_connected(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::NeighborhoodConnectedFilter;

    let lower = args
        .lower
        .ok_or_else(|| anyhow!("--lower is required for neighborhood-connected"))?;
    let upper = args
        .upper
        .ok_or_else(|| anyhow!("--upper is required for neighborhood-connected"))?;
    if lower > upper {
        return Err(anyhow!("--lower ({lower}) must be <= --upper ({upper})"));
    }
    let seed_str = args
        .seed
        .as_deref()
        .ok_or_else(|| anyhow!("--seed is required for neighborhood-connected (format: Z,Y,X)"))?;
    let seed =
        parse_seed(seed_str).with_context(|| format!("Failed to parse --seed '{seed_str}'"))?;

    let image = read_image(&args.input)?;
    let shape = image.shape();
    if seed[0] >= shape[0] || seed[1] >= shape[1] || seed[2] >= shape[2] {
        return Err(anyhow!(
            "Seed [{},{},{}] is out of bounds for image shape [{}x{}x{}]",
            seed[0],
            seed[1],
            seed[2],
            shape[0],
            shape[1],
            shape[2],
        ));
    }

    let r = args.neighborhood_radius;
    let filter = NeighborhoodConnectedFilter::new(seed, lower, upper).with_radius([r, r, r]);
    let mask = filter.apply(&image);
    let n_foreground = count_foreground(&mask);

    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: neighborhood-connected found {} foreground voxels (seed=[{},{},{}], range=[{:.4},{:.4}], radius={})",
        args.input.display(), n_foreground, seed[0], seed[1], seed[2], lower, upper, r,
    );

    info!("segment: neighborhood-connected complete input={} seed={:?} lower={} upper={} radius={} foreground={}", args.input.display(), seed, lower, upper, r, n_foreground);
    Ok(())
}
