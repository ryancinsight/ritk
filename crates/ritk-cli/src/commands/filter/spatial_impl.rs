use anyhow::{anyhow, Context, Result};
use tracing::info;

use super::{read_image, write_image_inferred, FilterArgs};

// ── Gradient magnitude ────────────────────────────────────────────────────────

pub(super) fn run_gradient_magnitude(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GradientMagnitudeFilter;

    let image = read_image(&args.input)?;
    let spacing = image.spacing();
    let filter = GradientMagnitudeFilter::new(*spacing);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

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
    use ritk_core::filter::LaplacianFilter;

    let image = read_image(&args.input)?;
    let spacing = image.spacing();
    let filter = LaplacianFilter::new(*spacing);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

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
    use ritk_core::filter::vesselness::FrangiConfig;
    use ritk_core::filter::FrangiVesselnessFilter;

    let image = read_image(&args.input)?;

    // Parse comma-separated scale list (e.g. "0.5,1.0,2.0").
    let scales: Vec<f64> = args
        .scales
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    let scales = if scales.is_empty() {
        vec![0.5, 1.0, 2.0]
    } else {
        scales
    };

    let config = FrangiConfig {
        scales: scales.clone(),
        alpha: args.alpha,
        beta: args.beta,
        gamma: args.gamma,
        polarity: ritk_core::filter::VesselPolarity::Bright,
    };
    let filter = FrangiVesselnessFilter { config };
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied frangi (scales={:?}, \u{03b1}={}, \u{03b2}={}, \u{03b3}={}) to {} \u{2192} {}",
        scales,
        args.alpha,
        args.beta,
        args.gamma,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: frangi complete input={} output={} alpha={} beta={} gamma={}",
        args.input.display(),
        args.output.display(),
        args.alpha,
        args.beta,
        args.gamma
    );
    Ok(())
}

// ── Median filter ─────────────────────────────────────────────────────────────

/// Apply a median filter with the given neighbourhood radius.
///
/// Radius 1 → 3×3×3 kernel (27 samples per voxel).
pub(super) fn run_median(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::MedianFilter;

    let image = read_image(&args.input)?;
    let filter = MedianFilter::new(args.radius);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied median (radius={}) to {} \u{2192} {}",
        args.radius,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: median complete input={} output={} radius={}",
        args.input.display(),
        args.output.display(),
        args.radius
    );
    Ok(())
}

// ── Bilateral filter ──────────────────────────────────────────────────────────

/// Apply a bilateral filter preserving edges.
///
/// `--sigma-spatial` controls the spatial Gaussian (voxels),
/// `--sigma-range` controls the intensity Gaussian.
pub(super) fn run_bilateral(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::BilateralFilter;

    let image = read_image(&args.input)?;
    let filter = BilateralFilter::new(args.sigma_spatial, args.sigma_range);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied bilateral (\u{03c3}_spatial={}, \u{03c3}_range={}) to {} \u{2192} {}",
        args.sigma_spatial,
        args.sigma_range,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: bilateral complete input={} output={} sigma_spatial={} sigma_range={}",
        args.input.display(),
        args.output.display(),
        args.sigma_spatial,
        args.sigma_range
    );
    Ok(())
}

// ── Canny edge detector ───────────────────────────────────────────────────────

/// Apply the Canny edge detector.
///
/// `--sigma` controls pre-smoothing, `--low` and `--high` set the
/// hysteresis thresholds on gradient magnitude.
pub(super) fn run_canny(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::CannyEdgeDetector;

    let image = read_image(&args.input)?;
    let detector = CannyEdgeDetector::new(args.sigma, args.low as f64, args.high as f64);
    let filtered = detector.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied canny (\u{03c3}={}, low={}, high={}) to {} \u{2192} {}",
        args.sigma,
        args.low,
        args.high,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: canny complete input={} output={} sigma={} low={} high={}",
        args.input.display(),
        args.output.display(),
        args.sigma,
        args.low,
        args.high
    );
    Ok(())
}

// ── Sobel filter ──────────────────────────────────────────────────────────────

/// Apply the Sobel gradient magnitude filter.
///
/// Uses the image's physical spacing to compute properly scaled gradients.
pub(super) fn run_sobel(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::SobelFilter;

    let image = read_image(&args.input)?;
    let spacing = image.spacing();
    let filter = SobelFilter::new(*spacing);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

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

/// Apply the Laplacian of Gaussian filter.
///
/// Computes G_σ * ∇²I by smoothing with Gaussian of standard deviation σ
/// then computing the discrete Laplacian.
pub(super) fn run_log(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LaplacianOfGaussianFilter;

    let image = read_image(&args.input)?;
    let filter = LaplacianOfGaussianFilter::new(args.sigma);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied log (\u{03c3}={}) to {} \u{2192} {}",
        args.sigma,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: log complete input={} output={} sigma={}",
        args.input.display(),
        args.output.display(),
        args.sigma
    );
    Ok(())
}

// ── Recursive Gaussian filter ─────────────────────────────────────────────────

/// Apply the recursive Gaussian (Young–van Vliet IIR) filter.
///
/// `--order` selects the derivative: 0 = smooth, 1 = first, 2 = second.
pub(super) fn run_recursive_gaussian(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::recursive_gaussian::DerivativeOrder;
    use ritk_core::filter::RecursiveGaussianFilter;

    let order = match args.order {
        0 => DerivativeOrder::Zero,
        1 => DerivativeOrder::First,
        2 => DerivativeOrder::Second,
        other => {
            return Err(anyhow!(
                "Invalid --order {other} for recursive-gaussian. \
                 Valid values: 0 (smooth), 1 (first derivative), 2 (second derivative)."
            ));
        }
    };

    let image = read_image(&args.input)?;
    let filter = RecursiveGaussianFilter::new(args.sigma).with_derivative_order(order);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied recursive-gaussian (\u{03c3}={}, order={}) to {} \u{2192} {}",
        args.sigma,
        args.order,
        args.input.display(),
        args.output.display()
    );
    info!(
        "filter: recursive-gaussian complete input={} output={} sigma={} order={}",
        args.input.display(),
        args.output.display(),
        args.sigma,
        args.order
    );
    Ok(())
}

// ── CPR (Curved Planar Reformation) ────────────────────────────────────────
pub(super) fn run_cpr(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::{CprConfig, CprImageFilter};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};

    if args.cpr_points.len() < 2 {
        return Err(anyhow!(
            "CPR requires at least 2 control points (got {}); use --cpr-point z,y,x",
            args.cpr_points.len()
        ));
    }

    let control_points: Vec<[f64; 3]> = args
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

    let image = read_image(&args.input)?;

    let cpr_filter = CprImageFilter::new(
        control_points,
        CprConfig {
            num_path_samples: args.cpr_path_samples as usize,
            cross_section_half_width: f64::from(args.cpr_half_width),
            num_cross_samples: args.cpr_cross_samples as usize,
        },
    );

    let image_2d = cpr_filter.apply(&image)?;

    // Promote 2-D [rows, cols] → 3-D [1, rows, cols] for the 3-D writer pipeline.
    let (tensor_2d, origin_2d, spacing_2d, _dir_2d) = image_2d.into_parts();
    let [nr, nc] = tensor_2d
        .shape()
        .dims
        .try_into()
        .expect("CPR output must be 2-D");
    let device = tensor_2d.device();
    let vals = tensor_2d
        .into_data()
        .into_vec::<f32>()
        .expect("CPR requires f32 backend");
    let td_3d = burn::tensor::TensorData::new(vals, burn::tensor::Shape::new([1, nr, nc]));
    let tensor_3d = burn::tensor::Tensor::<super::super::Backend, 3>::from_data(td_3d, &device);
    let origin_3d = Point::new([0.0, origin_2d[0], origin_2d[1]]);
    let spacing_3d = Spacing::new([1.0, spacing_2d[0], spacing_2d[1]]);
    let image_3d = Image::new(tensor_3d, origin_3d, spacing_3d, Direction::identity());
    write_image_inferred(&args.output, &image_3d)?;

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
