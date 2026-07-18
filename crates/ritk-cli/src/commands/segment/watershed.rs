use anyhow::{anyhow, Context, Result};

use tracing::info;

use ritk_segmentation::{MarkerControlledWatershed, WatershedSegmentation};

use super::super::{
    infer_format, is_native_read_capable, read_image_native, write_image_native, NativeBackend };
use super::args::SegmentArgs;
use super::helpers::read_native_input;

// â”€â”€ Watershed segmentation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Apply watershed flooding segmentation.
///
/// The input should be a scalar 3-D image (e.g. gradient magnitude).
/// Returns a label image where label 0 = watershed boundary and
/// labels 1..K = catchment basin indices.
pub(super) fn run_watershed(args: &SegmentArgs) -> Result<()> {
    let (image, output_format) = read_native_input(&args.input, &args.output, "watershed")?;
    let backend = NativeBackend::default();

    let ws = WatershedSegmentation::new();
    let labeled = ws.apply_native(&image, &backend)?;

    let max_label = labeled
        .data_slice()?
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let n_basins = max_label as usize;

    write_image_native(&args.output, &labeled, output_format)?;

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

// â”€â”€ Marker-controlled watershed segmentation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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

    let (gradient, output_format) =
        read_native_input(&args.input, &args.output, "marker-watershed")?;
    let marker_format = infer_format(markers_path)
        .ok_or_else(|| anyhow!("Cannot infer marker format: {}", markers_path.display()))?;
    anyhow::ensure!(
        is_native_read_capable(marker_format),
        "marker-watershed requires native marker format"
    );
    let markers = read_image_native(markers_path)?;
    let backend = NativeBackend::default();

    let labeled = MarkerControlledWatershed::new()
        .apply_native(&gradient, &markers, &backend)
        .with_context(|| format!("marker-watershed failed for input={}", args.input.display()))?;

    let max_label = labeled
        .data_slice()?
        .iter()
        .copied()
        .fold(0.0_f32, f32::max);
    let n_basins = max_label as usize;

    write_image_native(&args.output, &labeled, output_format)?;

    println!(
        "Segmented {} (marker-watershed): found {} basins",
        args.input.display(),
        n_basins,
    );
    info!(
        "segment: marker-watershed complete input={} markers={} basins={n_basins}",
        args.input.display(),
        markers_path.display(),
    );
    Ok(())
}
