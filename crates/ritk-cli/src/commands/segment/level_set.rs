use anyhow::{anyhow, Context, Result};
use tracing::info;

use ritk_core::segmentation::{ChanVeseSegmentation, LaplacianLevelSet};

use super::args::SegmentArgs;
use super::helpers::count_foreground;
use super::super::{read_image, write_image_inferred};

// -- Shape-detection level set ------------------------------------------

pub(super) fn run_shape_detection(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::ShapeDetectionSegmentation;

    let phi_path = args
        .initial_phi
        .as_ref()
        .ok_or_else(|| anyhow!("--initial-phi is required for shape-detection"))?;

    let image = read_image(&args.input)?;
    let initial_phi = read_image(phi_path)?;

    let mut seg = ShapeDetectionSegmentation::new();
    seg.curvature_weight = args.curvature_weight as f64;
    seg.propagation_weight = args.propagation_weight as f64;
    seg.advection_weight = args.advection_weight as f64;
    seg.edge_k = args.edge_k as f64;
    seg.sigma = args.sigma as f64;
    seg.dt = args.dt as f64;
    seg.max_iterations = args.level_set_max_iterations;
    seg.tolerance = args.tolerance as f64;

    let mask = seg
        .apply(&image, &initial_phi)
        .with_context(|| "shape-detection segmentation failed")?;
    let n_foreground = count_foreground(&mask);

    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: shape-detection found {} foreground voxels",
        args.input.display(),
        n_foreground,
    );

    info!(
        "segment: shape-detection complete input={} output={} foreground={}",
        args.input.display(),
        args.output.display(),
        n_foreground
    );
    Ok(())
}

// -- Threshold level set --------------------------------------------------

pub(super) fn run_threshold_level_set(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::ThresholdLevelSet;

    let phi_path = args
        .initial_phi
        .as_ref()
        .ok_or_else(|| anyhow!("--initial-phi is required for threshold-level-set"))?;
    let lower = args
        .lower_threshold
        .ok_or_else(|| anyhow!("--lower-threshold is required for threshold-level-set"))?;
    let upper = args
        .upper_threshold
        .ok_or_else(|| anyhow!("--upper-threshold is required for threshold-level-set"))?;
    if lower > upper {
        return Err(anyhow!(
            "--lower-threshold ({lower}) must be <= --upper-threshold ({upper})"
        ));
    }

    let image = read_image(&args.input)?;
    let initial_phi = read_image(phi_path)?;

    let mut seg = ThresholdLevelSet::new(lower as f64, upper as f64);
    seg.propagation_weight = args.propagation_weight as f64;
    seg.curvature_weight = args.curvature_weight as f64;
    seg.dt = args.dt as f64;
    seg.max_iterations = args.level_set_max_iterations;
    seg.tolerance = args.tolerance as f64;

    let mask = seg
        .apply(&image, &initial_phi)
        .with_context(|| "threshold-level-set segmentation failed")?;
    let n_foreground = count_foreground(&mask);

    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: threshold-level-set found {} foreground voxels (range=[{:.4},{:.4}])",
        args.input.display(),
        n_foreground,
        lower,
        upper,
    );

    info!(
        "segment: threshold-level-set complete input={} output={} lower={} upper={} foreground={}",
        args.input.display(),
        args.output.display(),
        lower,
        upper,
        n_foreground
    );
    Ok(())
}

// -- Laplacian level set --------------------------------------------------

pub(super) fn run_laplacian_level_set(args: &SegmentArgs) -> Result<()> {
    let phi_path = args
        .initial_phi
        .as_ref()
        .ok_or_else(|| anyhow!("--initial-phi is required for laplacian-level-set"))?;

    let image = read_image(&args.input)?;
    let initial_phi = read_image(phi_path)?;

    let mut seg = LaplacianLevelSet::new();
    seg.propagation_weight = args.propagation_weight as f64;
    seg.curvature_weight = args.curvature_weight as f64;
    seg.sigma = args.sigma as f64;
    seg.dt = args.dt as f64;
    seg.max_iterations = args.level_set_max_iterations;
    seg.tolerance = args.tolerance as f64;

    let mask = seg
        .apply(&image, &initial_phi)
        .with_context(|| "laplacian-level-set segmentation failed")?;
    let n_foreground = count_foreground(&mask);

    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: laplacian-level-set found {} foreground voxels",
        args.input.display(),
        n_foreground,
    );

    info!("segment: laplacian-level-set complete input={} output={} foreground={} propagation={} curvature={} sigma={}", args.input.display(), args.output.display(), n_foreground, args.propagation_weight, args.curvature_weight, args.sigma);
    Ok(())
}

// -- Chan-Vese active contours --------------------------------------------

pub(super) fn run_chan_vese(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    let mut seg = ChanVeseSegmentation::new();
    seg.mu = args.mu;
    seg.nu = args.nu;
    seg.lambda1 = args.lambda1;
    seg.lambda2 = args.lambda2;
    seg.epsilon = args.epsilon;
    seg.dt = args.dt as f64;
    seg.max_iterations = args.level_set_max_iterations;
    seg.tolerance = args.tolerance as f64;

    let mask = seg
        .apply(&image)
        .with_context(|| "chan-vese segmentation failed")?;
    let n_foreground = count_foreground(&mask);
    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: chan-vese found {} foreground voxels",
        args.input.display(),
        n_foreground,
    );
    info!("segment: chan-vese complete input={} output={} foreground={} mu={} nu={} lambda1={} lambda2={}", args.input.display(), args.output.display(), n_foreground, args.mu, args.nu, args.lambda1, args.lambda2);
    Ok(())
}

// -- Geodesic active contour ----------------------------------------------

pub(super) fn run_geodesic_active_contour(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::GeodesicActiveContourSegmentation;

    let phi_path = args
        .initial_phi
        .as_ref()
        .ok_or_else(|| anyhow!("--initial-phi is required for geodesic-active-contour"))?;
    let image = read_image(&args.input)?;
    let initial_phi = read_image(phi_path)?;

    let mut seg = GeodesicActiveContourSegmentation::new();
    seg.propagation_weight = args.propagation_weight as f64;
    seg.curvature_weight = args.curvature_weight as f64;
    seg.advection_weight = args.advection_weight as f64;
    seg.edge_k = args.edge_k as f64;
    seg.sigma = args.sigma as f64;
    seg.dt = args.dt as f64;
    seg.max_iterations = args.level_set_max_iterations;
    seg.tolerance = args.tolerance as f64;

    let mask = seg
        .apply(&image, &initial_phi)
        .with_context(|| "geodesic-active-contour segmentation failed")?;
    let n_foreground = count_foreground(&mask);
    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: geodesic-active-contour found {} foreground voxels",
        args.input.display(),
        n_foreground,
    );
    info!("segment: geodesic-active-contour complete input={} output={} foreground={} propagation={} curvature={} advection={}", args.input.display(), args.output.display(), n_foreground, args.propagation_weight, args.curvature_weight, args.advection_weight);
    Ok(())
}
