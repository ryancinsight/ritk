use anyhow::{anyhow, Context, Result};
use std::path::PathBuf;
use tracing::info;

use ritk_core::segmentation::{MarkerControlledWatershed, WatershedSegmentation};

use super::args::SegmentArgs;
use super::super::{read_image, write_image_inferred};

// ── Watershed segmentation ────────────────────────────────────────────────────

/// Apply watershed flooding segmentation.
///
/// The input should be a scalar 3-D image (e.g. gradient magnitude).
/// Returns a label image where label 0 = watershed boundary and
/// labels 1..K = catchment basin indices.
pub(super) fn run_watershed(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    let ws = WatershedSegmentation::new();
    let labeled = ws.apply(&image)?;

    let td = labeled.data().clone().into_data();
    let vals = td
        .as_slice::<f32>()
        .expect("watershed output must contain f32 data");
    let max_label = vals.iter().cloned().fold(0.0_f32, f32::max);
    let n_basins = max_label as usize;

    write_image_inferred(&args.output, &labeled)?;

    println!(
        "Segmented {} (watershed): found {} catchment basins",
        args.input.display(),
        n_basins,
    );

    info!(
        "segment: watershed complete input={} basins={}",
        args.input.display(),
        n_basins
    );

    Ok(())
}

// ── Marker-controlled watershed segmentation ──────────────────────────────────

/// Apply marker-controlled watershed segmentation.
///
/// The gradient image (`--input`) drives the flooding order; the marker image
/// (`--markers`) seeds the initial basin labels.  Non-zero voxels in the marker
/// image define basin seeds; zero voxels are unlabeled.  Each unlabeled voxel
/// receives the label of the adjacent seed reached via the lowest-gradient path.
/// Voxels on the boundary between two distinct basins receive label 0.
///
/// # Errors
/// Returns an error when:
/// - `--markers` is not supplied.
/// - The gradient and marker images have different shapes.
pub(super) fn run_marker_watershed(args: &SegmentArgs) -> Result<()> {
    let markers_path = args
        .markers
        .as_ref()
        .ok_or_else(|| anyhow!("marker-watershed requires --markers <PATH>"))?;

    let gradient = read_image(&args.input)?;
    let markers = read_image(&PathBuf::from(markers_path))?;

    let labeled = MarkerControlledWatershed::new()
        .apply(&gradient, &markers)
        .with_context(|| format!("marker-watershed failed for input={}", args.input.display()))?;

    let td = labeled.data().clone().into_data();
    let vals = td
        .as_slice::<f32>()
        .expect("marker-watershed output must contain f32 data");
    let max_label = vals.iter().cloned().fold(0.0_f32, f32::max);
    let n_basins = max_label as usize;

    write_image_inferred(&args.output, &labeled)?;

    println!(
        "Segmented {} (marker-watershed): found {} basins",
        args.input.display(),
        n_basins,
    );
    info!(
        "segment: marker-watershed complete input={} markers={} basins={n_basins}",
        args.input.display(),
        markers_path,
    );
    Ok(())
}
