use anyhow::Result;
use tracing::info;

use ritk_filter::{DistanceMeasure, DistanceTransformImageFilter};
use ritk_segmentation::{
    BinaryFillHoles, ConnectedComponentsFilter, KMeansSegmentation, MorphologicalGradient,
    Skeletonization,
};

use super::super::{write_image, Backend};
use super::args::SegmentArgs;
use super::helpers::{count_native_foreground, read_native_input};

// ── K-Means clustering ────────────────────────────────────────────────────────

/// Apply K-Means intensity clustering.
///
/// Each voxel in the output contains its assigned cluster index (0..K−1)
/// as `f32`.  Spatial metadata is preserved.
pub(super) fn run_kmeans(args: &SegmentArgs) -> Result<()> {
    let (image, output_format) = read_native_input(&args.input, &args.output, "kmeans")?;

    let mut km = KMeansSegmentation::new(args.classes)?;
    if let Some(mi) = args.kmeans_max_iterations {
        km = km.with_max_iterations(mi)?;
    }
    if let Some(tol) = args.kmeans_tolerance {
        km = km.with_tolerance(tol)?;
    }
    if let Some(seed) = args.kmeans_seed {
        km = km.with_seed(seed);
    }
    let backend = Backend::default();
    let labeled = km.apply_native(&image, &backend)?;

    write_image(&args.output, &labeled, output_format)?;

    println!(
        "Segmented {} (kmeans): k={} clusters",
        args.input.display(),
        args.classes,
    );

    info!(
        "segment: kmeans complete input={} k={}",
        args.input.display(),
        args.classes
    );

    Ok(())
}

// ── Distance transform ────────────────────────────────────────────────────────

/// Compute the Euclidean distance transform of a binary mask.
///
/// The input is binarised at threshold 0.5 (voxels > 0.5 = foreground).
/// Foreground voxels receive zero; each background voxel receives its physical
/// Euclidean distance to the nearest foreground voxel using image spacing.
pub(super) fn run_distance_transform(args: &SegmentArgs) -> Result<()> {
    let (image, output_format) =
        read_native_input(&args.input, &args.output, "distance-transform")?;
    let backend = Backend::default();
    let dt = DistanceTransformImageFilter::new()
        .with_measure(DistanceMeasure::Euclidean)
        .apply_native(&image, &backend)?;
    write_image(&args.output, &dt, output_format)?;

    println!(
        "Computed distance-transform for {} \u{2192} {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        "segment: distance-transform complete input={} output={}",
        args.input.display(),
        args.output.display()
    );

    Ok(())
}

/// Apply binary hole filling.
///
/// The input must be a binary mask (0.0 / 1.0). All background voxels not
/// reachable from the border are converted to foreground.
pub(super) fn run_fill_holes(args: &SegmentArgs) -> Result<()> {
    let (image, output_format) = read_native_input(&args.input, &args.output, "fill-holes")?;
    let backend = Backend::default();
    let filled = BinaryFillHoles.apply_native(&image, &backend)?;
    write_image(&args.output, &filled, output_format)?;

    println!(
        "Segmented {} (fill-holes) -> {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        "segment: fill-holes complete input={} output={}",
        args.input.display(),
        args.output.display()
    );

    Ok(())
}

/// Apply binary morphological gradient.
///
/// Produces a boundary mask from the binary input via dilation AND NOT erosion.
pub(super) fn run_morphological_gradient(args: &SegmentArgs) -> Result<()> {
    let (image, output_format) =
        read_native_input(&args.input, &args.output, "morphological-gradient")?;
    let backend = Backend::default();
    let gradient = MorphologicalGradient::new(1).apply_native(&image, &backend)?;
    write_image(&args.output, &gradient, output_format)?;

    println!(
        "Segmented {} (morphological-gradient) -> {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        "segment: morphological-gradient complete input={} output={}",
        args.input.display(),
        args.output.display()
    );

    Ok(())
}

// -- Connected components -------------------------------------------------

pub(super) fn run_connected_components(args: &SegmentArgs) -> Result<()> {
    let (image, output_format) =
        read_native_input(&args.input, &args.output, "connected-components")?;
    let connectivity = if args.connectivity == 6 {
        ritk_segmentation::labeling::Connectivity::Six
    } else {
        ritk_segmentation::labeling::Connectivity::TwentySix
    };
    let backend = Backend::default();
    let (labels, stats) = ConnectedComponentsFilter::with_connectivity(connectivity)
        .apply_native(&image, &backend)?;
    write_image(&args.output, &labels, output_format)?;

    println!(
        "Labeled {}: connected-components found {} components (connectivity={})",
        args.input.display(),
        stats.len(),
        args.connectivity,
    );
    info!(
        "segment: connected-components complete input={} output={} n_components={} connectivity={}",
        args.input.display(),
        args.output.display(),
        stats.len(),
        args.connectivity
    );
    Ok(())
}

// -- Skeletonization ------------------------------------------------------

pub(super) fn run_skeletonization(args: &SegmentArgs) -> Result<()> {
    let (image, output_format) = read_native_input(&args.input, &args.output, "skeletonization")?;
    let backend = Backend::default();
    let skeleton = Skeletonization::new().apply_native(&image, &backend)?;
    let n_skeleton = count_native_foreground(&skeleton)?;
    write_image(&args.output, &skeleton, output_format)?;

    println!(
        "Computed skeleton for {} -> {} ({} skeleton voxels)",
        args.input.display(),
        args.output.display(),
        n_skeleton,
    );

    info!(
        "segment: skeletonization complete input={} output={} skeleton={}",
        args.input.display(),
        args.output.display(),
        n_skeleton
    );
    Ok(())
}
