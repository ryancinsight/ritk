use anyhow::{anyhow, Context, Result};
use tracing::info;

use super::super::{
    infer_format, is_read_capable, is_write_capable, read_image, write_image, Backend,
};
use super::FilterArgs;

// ── Gradient magnitude ────────────────────────────────────────────────────────

pub(super) fn run_gradient_magnitude(args: &FilterArgs) -> Result<()> {
    use ritk_filter::GradientMagnitudeFilter;

    let image = read_image(&args.input)?;
    let filter = GradientMagnitudeFilter::new(*image.spacing());
    let filtered = filter.apply_native(&image)?;
    let fmt = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    write_image(&args.output, &filtered, fmt)?;

    println!(
        "Applied gradient-magnitude to {} \u{2192} {}",
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: gradient-magnitude complete input={} output={}",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

// ── Laplacian ─────────────────────────────────────────────────────────────────

pub(super) fn run_laplacian(args: &FilterArgs) -> Result<()> {
    use ritk_filter::LaplacianFilter;

    let image = read_image(&args.input)?;
    let filter = LaplacianFilter::new(*image.spacing());
    let filtered = filter.apply_native(&image)?;
    let fmt = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    write_image(&args.output, &filtered, fmt)?;

    println!(
        "Applied laplacian to {} \u{2192} {}",
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: laplacian complete input={} output={}",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

// ── Frangi vesselness ─────────────────────────────────────────────────────────

pub(super) fn run_frangi(args: &FilterArgs) -> Result<()> {
    use ritk_filter::vesselness::FrangiConfig;
    use ritk_filter::FrangiVesselnessFilter;

    let image = read_image(&args.input)?;

    // Use the scales from the vesselness Args chunk.
    let scales = args.vesselness.scales.clone();
    let scales = if scales.is_empty() {
        vec![0.5, 1.0, 2.0]
    } else {
        scales
    };

    let config = FrangiConfig {
        scales,
        alpha: args.vesselness.alpha,
        beta: args.vesselness.beta,
        gamma: args.vesselness.gamma,
        polarity: ritk_filter::VesselPolarity::Bright,
    };
    let filter = FrangiVesselnessFilter { config };
    let backend = Backend::default();
    let filtered = filter.apply_native(&image, &backend)?;
    let fmt = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    write_image(&args.output, &filtered, fmt)?;

    println!(
        "Applied frangi (scales={:?}, \u{03b1}={}, \u{03b2}={}, \u{03b3}={}) to {} \u{2192} {}",
        filter.config.scales,
        args.vesselness.alpha,
        args.vesselness.beta,
        args.vesselness.gamma,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: frangi complete input={} output={} alpha={} beta={} gamma={}",
        args.input.display(),
        args.output.display(),
        args.vesselness.alpha,
        args.vesselness.beta,
        args.vesselness.gamma
    );
    Ok(())
}

// ── Median filter ─────────────────────────────────────────────────────────────

/// Apply a median filter with the given neighbourhood radius.
///
/// Radius 1 → 3×3×3 kernel (27 samples per voxel).
pub(super) fn run_median(args: &FilterArgs) -> Result<()> {
    use ritk_filter::MedianFilter;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "median requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let filter = MedianFilter::new(args.kernel.radius);
    let filtered = filter.apply_native(&image)?;
    write_image(&args.output, &filtered, output_format)?;

    println!(
        "Applied median (radius={}) to {} \u{2192} {}",
        args.kernel.radius,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: median complete input={} output={} radius={}",
        args.input.display(),
        args.output.display(),
        args.kernel.radius
    );
    Ok(())
}

// ── Bilateral filter ──────────────────────────────────────────────────────────

/// Apply a bilateral filter preserving edges.
///
/// Edge σ values control bilateral spatial/intensity Gaussians.
pub(super) fn run_bilateral(args: &FilterArgs) -> Result<()> {
    use ritk_filter::BilateralFilter;

    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filter = BilateralFilter::new(args.edge.sigma_spatial, args.edge.sigma_range);
    let filtered = filter.apply_native(&image, &backend)?;
    let fmt = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    write_image(&args.output, &filtered, fmt)?;

    println!(
        "Applied bilateral (\u{03c3}_spatial={}, \u{03c3}_range={}) to {} \u{2192} {}",
        args.edge.sigma_spatial,
        args.edge.sigma_range,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: bilateral complete input={} output={} sigma_spatial={} sigma_range={}",
        args.input.display(),
        args.output.display(),
        args.edge.sigma_spatial,
        args.edge.sigma_range
    );
    Ok(())
}

// ── Canny edge detector ───────────────────────────────────────────────────────

/// Apply the Canny edge detector.  Reads shared `sigma` from
/// [`crate::commands::filter::SmoothingArgs`] (Gaussian family) and edge
/// hysteresis from [`crate::commands::filter::EdgeArgs`].
pub(super) fn run_canny(args: &FilterArgs) -> Result<()> {
    use ritk_filter::edge::GaussianSigma;
    use ritk_filter::CannyEdgeDetector;

    let image = read_image(&args.input)?;
    let sigma = GaussianSigma::new(args.smoothing.sigma)
        .ok_or_else(|| anyhow!("--sigma must be > 0, got {}", args.smoothing.sigma))?;
    let backend = Backend::default();
    let detector = CannyEdgeDetector::new(sigma, args.edge.low as f64, args.edge.high as f64);
    let filtered = detector.apply_native(&image, &backend)?;
    let fmt = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    write_image(&args.output, &filtered, fmt)?;

    println!(
        "Applied canny (\u{03c3}={}, low={}, high={}) to {} \u{2192} {}",
        args.smoothing.sigma,
        args.edge.low,
        args.edge.high,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: canny complete input={} output={} sigma={} low={} high={}",
        args.input.display(),
        args.output.display(),
        args.smoothing.sigma,
        args.edge.low,
        args.edge.high
    );
    Ok(())
}

// ── Sobel filter ──────────────────────────────────────────────────────────────

/// Apply the Sobel gradient magnitude filter.
///
/// Uses the image's physical spacing to compute properly scaled gradients.
pub(super) fn run_sobel(args: &FilterArgs) -> Result<()> {
    use ritk_filter::SobelFilter;

    let image = read_image(&args.input)?;
    let filter = SobelFilter::new(*image.spacing());
    let filtered = filter.apply_native(&image)?;
    let fmt = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    write_image(&args.output, &filtered, fmt)?;

    println!(
        "Applied sobel to {} \u{2192} {}",
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: sobel complete input={} output={}",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

// ── Laplacian of Gaussian (LoG) ───────────────────────────────────────────────

/// Apply the Laplacian of Gaussian filter.  Reads σ from `SmoothingArgs`.
pub(super) fn run_log(args: &FilterArgs) -> Result<()> {
    use ritk_filter::edge::GaussianSigma;
    use ritk_filter::LaplacianOfGaussianFilter;

    let image = read_image(&args.input)?;
    let sigma = GaussianSigma::new(args.smoothing.sigma)
        .ok_or_else(|| anyhow!("--sigma must be > 0, got {}", args.smoothing.sigma))?;
    let backend = Backend::default();
    let filter = LaplacianOfGaussianFilter::new(sigma);
    let filtered = filter.apply_native(&image, &backend)?;
    let fmt = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    write_image(&args.output, &filtered, fmt)?;

    println!(
        "Applied log (\u{03c3}={}) to {} \u{2192} {}",
        args.smoothing.sigma,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: log complete input={} output={} sigma={}",
        args.input.display(),
        args.output.display(),
        args.smoothing.sigma
    );
    Ok(())
}

// ── Recursive Gaussian filter ─────────────────────────────────────────────────

/// Apply the recursive Gaussian (Young–van Vliet IIR) filter.
pub(super) fn run_recursive_gaussian(args: &FilterArgs) -> Result<()> {
    use crate::commands::filter::CliDerivativeOrder;
    use ritk_filter::recursive_gaussian::DerivativeOrder;
    use ritk_filter::RecursiveGaussianFilter;

    let order = match args.recursive.order {
        CliDerivativeOrder::Zero => DerivativeOrder::Zero,
        CliDerivativeOrder::First => DerivativeOrder::First,
        CliDerivativeOrder::Second => DerivativeOrder::Second,
    };

    let image = read_image(&args.input)?;
    let backend = Backend::default();
    let filter = RecursiveGaussianFilter::new(args.smoothing.sigma).with_derivative_order(order);
    let filtered = filter.apply_native(&image, &backend)?;
    let fmt = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    write_image(&args.output, &filtered, fmt)?;

    println!(
        "Applied recursive-gaussian (\u{03c3}={}, order={}) to {} \u{2192} {}",
        args.smoothing.sigma,
        args.recursive.order,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: recursive-gaussian complete input={} output={} sigma={} order={}",
        args.input.display(),
        args.output.display(),
        args.smoothing.sigma,
        args.recursive.order
    );
    Ok(())
}

// ── CPR (Curved Planar Reformation) ────────────────────────────────────────
pub(super) fn run_cpr(args: &FilterArgs) -> Result<()> {
    use ritk_filter::{CprConfig, CprImageFilter};
    use ritk_spatial::{Direction, Point, Spacing};

    if args.cpr.cpr_points.len() < 2 {
        return Err(anyhow!(
            "CPR requires at least 2 control points (got {}); use --cpr-point z,y,x",
            args.cpr.cpr_points.len()
        ));
    }

    let control_points: Vec<[f64; 3]> = args
        .cpr
        .cpr_points
        .iter()
        .map(|s| {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() != 3 {
                return Err(anyhow!("each --cpr-point must be 'z,y,x', got '{}'", s));
            }
            let z = parts[0]
                .parse::<f64>()
                .with_context(|| format!("invalid z in --cpr-point '{}'", s))?;
            let y = parts[1]
                .parse::<f64>()
                .with_context(|| format!("invalid y in --cpr-point '{}'", s))?;
            let x = parts[2]
                .parse::<f64>()
                .with_context(|| format!("invalid x in --cpr-point '{}'", s))?;
            Ok([z, y, x])
        })
        .collect::<Result<Vec<_>>>()?;

    let input_format = infer_format(&args.input)
        .ok_or_else(|| anyhow!("Cannot infer input format: {}", args.input.display()))?;
    let output_format = infer_format(&args.output)
        .ok_or_else(|| anyhow!("Cannot infer output format: {}", args.output.display()))?;
    anyhow::ensure!(
        is_read_capable(input_format) && is_write_capable(output_format),
        "cpr requires native input/output formats"
    );
    let image = read_image(&args.input)?;
    let backend = Backend::default();

    let cpr_filter = CprImageFilter::new(
        control_points,
        CprConfig {
            num_path_samples: args.cpr.cpr_path_samples as usize,
            cross_section_half_width: f64::from(args.cpr.cpr_half_width),
            num_cross_samples: args.cpr.cpr_cross_samples as usize,
        },
    );

    let image_2d = cpr_filter.apply_native(&image, &backend)?;

    // Promote 2-D [rows, cols] → 3-D [1, rows, cols] for the 3-D writer pipeline.
    let [nr, nc] = image_2d.shape();
    let origin = image_2d.origin();
    let spacing = image_2d.spacing();
    let image_3d = ritk_image::Image::from_flat_on(
        image_2d
            .data_slice()
            .expect("invariant: image storage is contiguous")
            .to_vec(),
        [1, nr, nc],
        Point::new([0.0, origin[0], origin[1]]),
        Spacing::new([1.0, spacing[0], spacing[1]]),
        Direction::identity(),
        &backend,
    )?;
    write_image(&args.output, &image_3d, output_format)?;

    println!(
        "Applied CPR to {} → {}",
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: CPR complete input={} output={}",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}
