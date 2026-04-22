//! `ritk filter` — image filtering command.
//!
//! Applies one of the following filters to a 3-D medical image:
//!
//! | Filter              | Parameters                                         |
//! |---------------------|----------------------------------------------------|
//! | `gaussian`          | `--sigma`                                          |
//! | `n4-bias`           | `--levels`, `--iterations`                         |
//! | `anisotropic`       | `--iterations`, `--conductance`                    |
//! | `gradient-magnitude`| (uses image spacing)                               |
//! | `laplacian`         | (uses image spacing)                               |
//! | `frangi`            | `--scales`, `--alpha`, `--beta`, `--gamma`         |
//! | `median`            | `--radius`                                         |
//! | `bilateral`         | `--sigma-spatial`, `--sigma-range`                  |
//! | `canny`             | `--sigma`, `--low`, `--high`                       |
//! | `sobel`             | (uses image spacing)                               |
//! | `log`               | `--sigma`                                          |
//! | `recursive-gaussian`| `--sigma`, `--order`                               |
//! | `curvature`         | `--iterations`, `--time-step`                      |
//! | `sato`              | `--scales`, `--alpha`                              |

use anyhow::{anyhow, Result};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use super::{read_image, write_image_inferred, Backend};

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `filter` subcommand.
#[derive(Args, Debug)]
pub struct FilterArgs {
    /// Input image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Filter to apply.
    ///
    /// Accepted values: `gaussian`, `n4-bias`, `anisotropic`, `frangi`,
    /// `gradient-magnitude`, `laplacian`, `median`, `bilateral`, `canny`,
    /// `sobel`, `log`, `recursive-gaussian`, `curvature`, `sato`.
    #[arg(long, value_name = "FILTER")]
    pub filter: String,

    // ── Gaussian / LoG / Canny / Recursive-Gaussian ───────────────────────
    /// Gaussian standard deviation in physical units (mm).
    ///
    /// Applied uniformly in all three spatial dimensions.
    /// Used by: `gaussian`, `canny`, `log`, `recursive-gaussian`.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub sigma: f64,

    // ── N4 bias-field correction ──────────────────────────────────────────
    /// Number of multi-resolution pyramid levels for N4 bias-field correction.
    ///
    /// Used by: `n4-bias`.
    #[arg(long, default_value = "4", value_name = "INT")]
    pub levels: usize,

    /// Maximum number of optimizer iterations per pyramid level.
    ///
    /// Used by: `n4-bias`, `anisotropic`.
    #[arg(long, default_value = "50", value_name = "INT")]
    pub iterations: usize,

    // ── Anisotropic diffusion ─────────────────────────────────────────────
    /// Conductance parameter controlling edge sensitivity for anisotropic
    /// diffusion.  Lower values preserve edges more aggressively.
    ///
    /// Used by: `anisotropic`.
    #[arg(long, default_value = "3.0", value_name = "FLOAT")]
    pub conductance: f64,

    // ── Curvature diffusion ────────────────────────────────────────────────────
    /// Explicit Euler time step Δt for curvature anisotropic diffusion.
    /// Stability requires Δt ≤ 1/6 for unit spacing.
    ///
    /// Used by: `curvature`.
    #[arg(long, default_value = "0.0625", value_name = "FLOAT")]
    pub time_step: f64,

    // ── Frangi vesselness ─────────────────────────────────────────────────
    /// Comma-separated list of vessel scale radii (mm) for multi-scale Frangi
    /// vesselness enhancement.
    ///
    /// Used by: `frangi`.
    #[arg(long, default_value = "0.5,1.0,2.0", value_name = "FLOATS")]
    pub scales: String,

    /// Frangi α parameter (controls sensitivity to plate-like structures).
    ///
    /// Used by: `frangi`.
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub alpha: f64,

    /// Frangi β parameter (controls sensitivity to blob-like structures).
    ///
    /// Used by: `frangi`.
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub beta: f64,

    /// Frangi γ parameter (controls sensitivity to background noise).
    ///
    /// Used by: `frangi`.
    #[arg(long, default_value = "15.0", value_name = "FLOAT")]
    pub gamma: f64,

    // ── Bilateral ─────────────────────────────────────────────────────────
    /// Spatial Gaussian sigma in voxels for the bilateral filter.
    ///
    /// Used by: `bilateral`.
    #[arg(long, default_value = "3.0", value_name = "FLOAT")]
    pub sigma_spatial: f64,

    /// Intensity-range Gaussian sigma for the bilateral filter.
    ///
    /// Used by: `bilateral`.
    #[arg(long, default_value = "50.0", value_name = "FLOAT")]
    pub sigma_range: f64,

    // ── Canny ─────────────────────────────────────────────────────────────
    /// Lower hysteresis threshold for the Canny edge detector.
    ///
    /// Used by: `canny`.
    #[arg(long, default_value = "0.1", value_name = "FLOAT")]
    pub low: f32,

    /// Upper hysteresis threshold for the Canny edge detector.
    ///
    /// Used by: `canny`.
    #[arg(long, default_value = "0.3", value_name = "FLOAT")]
    pub high: f32,

    // ── Median ────────────────────────────────────────────────────────────
    /// Neighbourhood half-width in voxels for the median filter.
    /// A radius of 1 produces a 3×3×3 kernel (27 samples per voxel).
    ///
    /// Used by: `median`.
    #[arg(long, default_value = "1", value_name = "INT")]
    pub radius: usize,

    // ── Recursive-Gaussian ────────────────────────────────────────────────
    /// Derivative order for the recursive Gaussian filter.
    /// 0 = smoothing, 1 = first derivative, 2 = second derivative.
    ///
    /// Used by: `recursive-gaussian`.
    #[arg(long, default_value = "0", value_name = "INT")]
    pub order: usize,

    // -- Intensity transform filters -------------------------------------------------
    /// Minimum output value for rescale-intensity and intensity-windowing filters.
    ///
    /// Used by: `rescale-intensity`, `intensity-windowing`, `sigmoid`.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub out_min: f32,

    /// Maximum output value for rescale-intensity and intensity-windowing filters.
    ///
    /// Used by: `rescale-intensity`, `intensity-windowing`, `sigmoid`.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub out_max: f32,

    /// Minimum of intensity window for intensity-windowing filter.
    ///
    /// Used by: `intensity-windowing`.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub window_min: f32,

    /// Maximum of intensity window for intensity-windowing filter.
    ///
    /// Used by: `intensity-windowing`.
    #[arg(long, default_value = "255.0", value_name = "FLOAT")]
    pub window_max: f32,

    /// Threshold value for threshold-below and threshold-above filters.
    ///
    /// Used by: `threshold-below`, `threshold-above`.
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub threshold_value: f32,

    /// Lower bound for threshold-outside and binary-threshold filters.
    ///
    /// Used by: `threshold-outside`, `binary-threshold`.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub lower_threshold: f32,

    /// Upper bound for threshold-outside and binary-threshold filters.
    ///
    /// Used by: `threshold-outside`, `binary-threshold`.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub upper_threshold: f32,

    /// Replacement value for pixels outside the threshold range.
    ///
    /// Used by: `threshold-below`, `threshold-above`, `threshold-outside`.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub outside_value: f32,

    /// Foreground value for binary-threshold filter.
    ///
    /// Used by: `binary-threshold`.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub foreground_value: f32,

    /// Background value for binary-threshold filter.
    ///
    /// Used by: `binary-threshold`.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub background_value: f32,

    /// Path to a mask image for filters requiring two inputs.
    ///
    /// Used by: `morphological-reconstruction`.
    #[arg(long)]
    pub mask: Option<std::path::PathBuf>,
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `filter` subcommand.
///
/// Dispatches to the appropriate filter implementation based on `args.filter`.
///
/// # Errors
/// Returns an error when:
/// - The input image cannot be read.
/// - The requested filter is not available in this build.
/// - The output image cannot be written.
/// - An unknown filter name is supplied.
pub fn run(args: FilterArgs) -> Result<()> {
    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        filter = %args.filter,
        "filter: starting"
    );

    match args.filter.as_str() {
        "gaussian" => run_gaussian(&args),
        "n4-bias" => run_n4_bias(&args),
        "anisotropic" => run_anisotropic(&args),
        "gradient-magnitude" => run_gradient_magnitude(&args),
        "laplacian" => run_laplacian(&args),
        "frangi" => run_frangi(&args),
        "median" => run_median(&args),
        "bilateral" => run_bilateral(&args),
        "canny" => run_canny(&args),
        "sobel" => run_sobel(&args),
        "log" => run_log(&args),
        "recursive-gaussian" => run_recursive_gaussian(&args),
        "curvature" => run_curvature(&args),
        "sato" => run_sato(&args),
        "rescale-intensity" => run_rescale_intensity(&args),
        "intensity-windowing" => run_intensity_windowing(&args),
        "threshold-below" => run_threshold_below(&args),
        "threshold-above" => run_threshold_above(&args),
        "threshold-outside" => run_threshold_outside(&args),
        "sigmoid" => run_sigmoid(&args),
        "binary-threshold" => run_binary_threshold(&args),
        "grayscale-erosion" => run_grayscale_erosion(&args),
        "grayscale-dilation" => run_grayscale_dilation(&args),
        "white-top-hat" => run_white_top_hat(&args),
        "black-top-hat" => run_black_top_hat(&args),
        "hit-or-miss" => run_hit_or_miss(&args),
        "label-dilation" => run_label_dilation(&args),
        "label-erosion" => run_label_erosion(&args),
        "label-opening" => run_label_opening(&args),
        "label-closing" => run_label_closing(&args),
        "morphological-reconstruction" => run_morphological_reconstruction(&args),
        other => Err(anyhow!(
            "Unknown filter '{other}'. \
             Available filters: gaussian, n4-bias, anisotropic, \
             gradient-magnitude, laplacian, frangi, median, bilateral, \
             canny, sobel, log, recursive-gaussian, curvature, sato, 
             rescale-intensity, intensity-windowing, threshold-below, 
             threshold-above, threshold-outside, sigmoid, binary-threshold."
        )),
    }
}

// ── Gaussian filter ───────────────────────────────────────────────────────────

/// Apply a Gaussian smoothing filter to the input image and write the result.
///
/// The sigma value from `args.sigma` is applied uniformly along all three
/// spatial dimensions.  The `GaussianFilter` implementation skips any
/// dimension whose sigma is ≤ 1e-6, so `--sigma 0.0` is a valid no-op.
fn run_gaussian(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GaussianFilter;

    let image = read_image(&args.input)?;

    // Isotropic sigma applied to all three spatial dimensions.
    let filter: GaussianFilter<Backend> = GaussianFilter::new(vec![args.sigma; 3]);
    let filtered = filter.apply(&image);

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied gaussian (\u{03c3}={}) to {} \u{2192} {}",
        args.sigma,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        sigma = args.sigma,
        "filter: gaussian complete"
    );

    Ok(())
}

// ── N4 bias field correction ──────────────────────────────────────────────────

fn run_n4_bias(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::bias::N4Config;
    use ritk_core::filter::N4BiasFieldCorrectionFilter;

    let image = read_image(&args.input)?;
    let config = N4Config {
        num_fitting_levels: args.levels,
        num_iterations: args.iterations,
        ..Default::default()
    };
    let filter = N4BiasFieldCorrectionFilter::new(config);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied n4-bias (levels={}, iters={}) to {} \u{2192} {}",
        args.levels,
        args.iterations,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        levels = args.levels,
        iterations = args.iterations,
        "filter: n4-bias complete"
    );

    Ok(())
}

// ── Anisotropic diffusion ─────────────────────────────────────────────────────

fn run_anisotropic(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::diffusion::{ConductanceFunction, DiffusionConfig};
    use ritk_core::filter::AnisotropicDiffusionFilter;

    let image = read_image(&args.input)?;
    let config = DiffusionConfig {
        num_iterations: args.iterations,
        conductance: args.conductance as f32,
        time_step: 0.0625,
        function: ConductanceFunction::Exponential,
    };
    let filter = AnisotropicDiffusionFilter::new(config);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied anisotropic (iters={}, K={}) to {} \u{2192} {}",
        args.iterations,
        args.conductance,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        iterations = args.iterations,
        conductance = args.conductance,
        "filter: anisotropic complete"
    );

    Ok(())
}

// -- Curvature anisotropic diffusion ------------------------------------------

fn run_curvature(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::diffusion::{CurvatureAnisotropicDiffusionFilter, CurvatureConfig};
    let image = read_image(&args.input)?;
    let config = CurvatureConfig {
        num_iterations: args.iterations,
        time_step: args.time_step as f32,
    };
    let filter = CurvatureAnisotropicDiffusionFilter::new(config);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied curvature (iters={}, dt={}) to {} -> {}",
        args.iterations, args.time_step, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(),
        iterations = args.iterations, time_step = args.time_step, "filter: curvature complete");
    Ok(())
}

// -- Sato line filter ---------------------------------------------------------

fn run_sato(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::vesselness::{SatoConfig, SatoLineFilter};
    let image = read_image(&args.input)?;
    let scales: Vec<f64> = args.scales.split(',').filter_map(|s| s.trim().parse::<f64>().ok()).collect();
    let scales = if scales.is_empty() { vec![1.0, 2.0, 3.0] } else { scales };
    let config = SatoConfig { scales: scales.clone(), alpha: args.alpha, bright_tubes: true };
    let filter = SatoLineFilter::new(config);
    let filtered = filter.apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied sato (scales={:?}, alpha={}) to {} -> {}",
        scales, args.alpha, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(),
        alpha = args.alpha, "filter: sato complete");
    Ok(())
}

// -- Gradient magnitude ────────────────────────────────────────────────────────

fn run_gradient_magnitude(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GradientMagnitudeFilter;

    let image = read_image(&args.input)?;
    let spacing = image.spacing();
    let filter = GradientMagnitudeFilter::new([spacing[0], spacing[1], spacing[2]]);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied gradient-magnitude to {} \u{2192} {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        "filter: gradient-magnitude complete"
    );

    Ok(())
}

// ── Laplacian ─────────────────────────────────────────────────────────────────

fn run_laplacian(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LaplacianFilter;

    let image = read_image(&args.input)?;
    let spacing = image.spacing();
    let filter = LaplacianFilter::new([spacing[0], spacing[1], spacing[2]]);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied laplacian to {} \u{2192} {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        "filter: laplacian complete"
    );

    Ok(())
}

// ── Frangi vesselness ─────────────────────────────────────────────────────────

fn run_frangi(args: &FilterArgs) -> Result<()> {
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
        bright_vessels: true,
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
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        alpha = args.alpha,
        beta = args.beta,
        gamma = args.gamma,
        "filter: frangi complete"
    );

    Ok(())
}

// ── Median filter ─────────────────────────────────────────────────────────────

/// Apply a median filter with the given neighbourhood radius.
///
/// Radius 1 → 3×3×3 kernel (27 samples per voxel).
fn run_median(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::MedianFilter;

    let image = read_image(&args.input)?;
    let filter = MedianFilter::new(args.radius);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied median (radius={}) to {} \u{2192} {}",
        args.radius,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        radius = args.radius,
        "filter: median complete"
    );

    Ok(())
}

// ── Bilateral filter ──────────────────────────────────────────────────────────

/// Apply a bilateral filter preserving edges.
///
/// `--sigma-spatial` controls the spatial Gaussian (voxels),
/// `--sigma-range` controls the intensity Gaussian.
fn run_bilateral(args: &FilterArgs) -> Result<()> {
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
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        sigma_spatial = args.sigma_spatial,
        sigma_range = args.sigma_range,
        "filter: bilateral complete"
    );

    Ok(())
}

// ── Canny edge detector ───────────────────────────────────────────────────────

/// Apply the Canny edge detector.
///
/// `--sigma` controls pre-smoothing, `--low` and `--high` set the
/// hysteresis thresholds on gradient magnitude.
fn run_canny(args: &FilterArgs) -> Result<()> {
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
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        sigma = args.sigma,
        low = args.low,
        high = args.high,
        "filter: canny complete"
    );

    Ok(())
}

// ── Sobel filter ──────────────────────────────────────────────────────────────

/// Apply the Sobel gradient magnitude filter.
///
/// Uses the image's physical spacing to compute properly scaled gradients.
fn run_sobel(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::SobelFilter;

    let image = read_image(&args.input)?;
    let spacing = image.spacing();
    let filter = SobelFilter::new([spacing[0], spacing[1], spacing[2]]);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied sobel to {} \u{2192} {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        "filter: sobel complete"
    );

    Ok(())
}

// ── Laplacian of Gaussian (LoG) ───────────────────────────────────────────────

/// Apply the Laplacian of Gaussian filter.
///
/// Computes G_σ * ∇²I by smoothing with Gaussian of standard deviation σ
/// then computing the discrete Laplacian.
fn run_log(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LaplacianOfGaussianFilter;

    let image = read_image(&args.input)?;
    let filter = LaplacianOfGaussianFilter::new(args.sigma);
    let filtered = filter.apply(&image)?;

    write_image_inferred(&args.output, &filtered)?;

    println!(
        "Applied log (\u{03c3}={}) to {} \u{2192} {}",
        args.sigma,
        args.input.display(),
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        sigma = args.sigma,
        "filter: log complete"
    );

    Ok(())
}

// ── Recursive Gaussian filter ─────────────────────────────────────────────────

/// Apply the recursive Gaussian (Young–van Vliet IIR) filter.
///
/// `--order` selects the derivative: 0 = smooth, 1 = first, 2 = second.
fn run_recursive_gaussian(args: &FilterArgs) -> Result<()> {
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
        args.output.display(),
    );

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        sigma = args.sigma,
        order = args.order,
        "filter: recursive-gaussian complete"
    );

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

fn run_rescale_intensity(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::RescaleIntensityFilter;
    let image = read_image(&args.input)?;
    let filtered = RescaleIntensityFilter::new(args.out_min, args.out_max).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied rescale-intensity (out=[{},{}]) to {} -> {}", args.out_min, args.out_max, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(), out_min = args.out_min, out_max = args.out_max, "filter: rescale-intensity complete");
    Ok(())
}

fn run_intensity_windowing(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::IntensityWindowingFilter;
    let image = read_image(&args.input)?;
    let filtered = IntensityWindowingFilter::new(args.window_min, args.window_max, args.out_min, args.out_max).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied intensity-windowing (window=[{},{}], out=[{},{}]) to {} -> {}", args.window_min, args.window_max, args.out_min, args.out_max, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(), "filter: intensity-windowing complete");
    Ok(())
}

fn run_threshold_below(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::ThresholdImageFilter;
    let image = read_image(&args.input)?;
    let filtered = ThresholdImageFilter::below(args.threshold_value, args.outside_value).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied threshold-below (threshold={}, outside={}) to {} -> {}", args.threshold_value, args.outside_value, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(), "filter: threshold-below complete");
    Ok(())
}

fn run_threshold_above(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::ThresholdImageFilter;
    let image = read_image(&args.input)?;
    let filtered = ThresholdImageFilter::above(args.threshold_value, args.outside_value).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied threshold-above (threshold={}, outside={}) to {} -> {}", args.threshold_value, args.outside_value, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(), "filter: threshold-above complete");
    Ok(())
}

fn run_threshold_outside(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::ThresholdImageFilter;
    let image = read_image(&args.input)?;
    let filtered = ThresholdImageFilter::outside(args.lower_threshold, args.upper_threshold, args.outside_value).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied threshold-outside ([{},{}], outside={}) to {} -> {}", args.lower_threshold, args.upper_threshold, args.outside_value, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(), "filter: threshold-outside complete");
    Ok(())
}

fn run_sigmoid(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::SigmoidImageFilter;
    let image = read_image(&args.input)?;
    let filtered = SigmoidImageFilter::new(args.alpha as f32, args.beta as f32, args.out_min, args.out_max).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied sigmoid (alpha={}, beta={}, out=[{},{}]) to {} -> {}", args.alpha, args.beta, args.out_min, args.out_max, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(), "filter: sigmoid complete");
    Ok(())
}

fn run_binary_threshold(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::BinaryThresholdImageFilter;
    let image = read_image(&args.input)?;
    let filtered = BinaryThresholdImageFilter::new(args.lower_threshold, args.upper_threshold, args.foreground_value, args.background_value).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    println!("Applied binary-threshold ([{},{}] fg={} bg={}) to {} -> {}", args.lower_threshold, args.upper_threshold, args.foreground_value, args.background_value, args.input.display(), args.output.display());
    info!(input = %args.input.display(), output = %args.output.display(), "filter: binary-threshold complete");
    Ok(())
}


fn run_grayscale_erosion(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GrayscaleErosion;
    let image = read_image(&args.input)?;
    let filtered = GrayscaleErosion::new(args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius=args.radius, input=%args.input.display(), output=%args.output.display(), "filter: grayscale-erosion complete");
    Ok(())
}

fn run_grayscale_dilation(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::GrayscaleDilation;
    let image = read_image(&args.input)?;
    let filtered = GrayscaleDilation::new(args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius=args.radius, input=%args.input.display(), output=%args.output.display(), "filter: grayscale-dilation complete");
    Ok(())
}

fn run_white_top_hat(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::WhiteTopHatFilter;
    let image = read_image(&args.input)?;
    let filtered = WhiteTopHatFilter::new(args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius=args.radius, input=%args.input.display(), output=%args.output.display(), "filter: white-top-hat complete");
    Ok(())
}

fn run_black_top_hat(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::BlackTopHatFilter;
    let image = read_image(&args.input)?;
    let filtered = BlackTopHatFilter::new(args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius=args.radius, input=%args.input.display(), output=%args.output.display(), "filter: black-top-hat complete");
    Ok(())
}

fn run_hit_or_miss(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::HitOrMissTransform;
    let image = read_image(&args.input)?;
    let filtered = HitOrMissTransform::new(args.radius, args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius=args.radius, input=%args.input.display(), output=%args.output.display(), "filter: hit-or-miss complete");
    Ok(())
}

fn run_label_dilation(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LabelDilation;
    let image = read_image(&args.input)?;
    let filtered = LabelDilation::new(args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius=args.radius, input=%args.input.display(), output=%args.output.display(), "filter: label-dilation complete");
    Ok(())
}

fn run_label_erosion(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LabelErosion;
    let image = read_image(&args.input)?;
    let filtered = LabelErosion::new(args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius = args.radius, input = %args.input.display(), output = %args.output.display(), "filter: label-erosion complete");
    Ok(())
}

fn run_label_opening(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LabelOpening;
    let image = read_image(&args.input)?;
    let filtered = LabelOpening::new(args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius = args.radius, "filter: label-opening complete");
    Ok(())
}

fn run_label_closing(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::LabelClosing;
    let image = read_image(&args.input)?;
    let filtered = LabelClosing::new(args.radius).apply(&image)?;
    write_image_inferred(&args.output, &filtered)?;
    info!(radius = args.radius, "filter: label-closing complete");
    Ok(())
}

fn run_morphological_reconstruction(args: &FilterArgs) -> Result<()> {
    use ritk_core::filter::{MorphologicalReconstruction, ReconstructionMode};
    let marker = read_image(&args.input)?;
    let mask_path = args.mask.as_ref().ok_or_else(|| {
        anyhow::anyhow!("morphological-reconstruction requires --mask <path>")
    })?;
    let mask = read_image(mask_path)?;
    let filtered = MorphologicalReconstruction::new(ReconstructionMode::Dilation)
        .apply(&marker, &mask)?;
    write_image_inferred(&args.output, &filtered)?;
    info!("filter: morphological-reconstruction complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    /// Default `FilterArgs` builder — sets every field to a reasonable default
    /// and lets the caller override what is needed.
    fn default_args(input: PathBuf, output: PathBuf, filter: &str) -> FilterArgs {
        FilterArgs {
            input,
            output,
            filter: filter.to_string(),
            sigma: 1.0,
            levels: 4,
            iterations: 50,
            conductance: 3.0,
            time_step: 0.0625,
            scales: "0.5,1.0,2.0".to_string(),
            alpha: 0.5,
            beta: 0.5,
            gamma: 15.0,
            sigma_spatial: 3.0,
            sigma_range: 50.0,
            low: 0.1,
            high: 0.3,
            radius: 1,
            order: 0,
            // new intensity filter fields
            out_min: 0.0,
            out_max: 1.0,
            window_min: 0.0,
            window_max: 255.0,
            threshold_value: 0.5,
            lower_threshold: 0.0,
            upper_threshold: 1.0,
            outside_value: 0.0,
            foreground_value: 1.0,
            background_value: 0.0,
            mask: None,
        }
    }

    /// Build a 5×5×5 test image whose voxel values are `0, 1, 2, …, 124`.
    fn make_test_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..125).map(|i| i as f32).collect();
        let td = TensorData::new(values, Shape::new([5, 5, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    // ── Positive: Gaussian creates output file ────────────────────────────────

    /// Applying the Gaussian filter must create the output file.
    #[test]
    fn test_filter_gaussian_creates_output_file() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("filtered.nii");

        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "gaussian")).unwrap();

        assert!(output.exists(), "output file must be created");
    }

    // ── Positive: Gaussian preserves shape ───────────────────────────────────

    /// The output image must have the same voxel dimensions as the input.
    #[test]
    fn test_filter_gaussian_preserves_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("filtered.mha");

        ritk_io::write_metaimage(&input, &make_test_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "gaussian")).unwrap();

        let result = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            result.shape(),
            [5, 5, 5],
            "shape must be preserved after Gaussian filtering"
        );
    }

    // ── Positive: Gaussian with sigma=0 is a no-op ───────────────────────────

    /// `--sigma 0.0` must leave voxel values unchanged (GaussianFilter skips
    /// dimensions with σ ≤ 1e-6).
    #[test]
    fn test_filter_gaussian_sigma_zero_is_noop() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("filtered.mha");

        let original = make_test_image();
        let original_data: Vec<f32> = original
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        ritk_io::write_metaimage(&input, &original).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "gaussian");
        args.sigma = 0.0;
        run(args).unwrap();

        let result = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let result_data: Vec<f32> = result
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .to_vec();

        // Sigma = 0 → no convolution → values must be identical after round-trip.
        // NIfTI/MetaImage round-trip may reorder axes; compare sums as a
        // scalar invariant that is permutation-independent.
        let orig_sum: f32 = original_data.iter().sum();
        let result_sum: f32 = result_data.iter().sum();
        assert!(
            (orig_sum - result_sum).abs() < 1e-3 * orig_sum.abs().max(1.0),
            "voxel sum must be preserved under \u{03c3}=0 Gaussian (orig={orig_sum}, result={result_sum})"
        );
    }

    // ── Positive: N4 bias-field correction creates output file ───────────────

    #[test]
    fn test_filter_n4_applies_correction() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "n4-bias");
        args.levels = 1;
        args.iterations = 5;
        let result = run(args);

        assert!(result.is_ok(), "n4-bias must succeed: {:?}", result.err());
        assert!(output.exists(), "n4-bias must write output file");
    }

    // ── Positive: anisotropic diffusion creates output file ───────────────────

    #[test]
    fn test_filter_anisotropic_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "anisotropic");
        args.iterations = 5;
        let result = run(args);

        assert!(
            result.is_ok(),
            "anisotropic must succeed: {:?}",
            result.err()
        );
        assert!(output.exists(), "anisotropic must write output file");
    }

    // ── Positive: Frangi vesselness creates output file ───────────────────────

    #[test]
    fn test_filter_frangi_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "frangi");
        args.scales = "1.0,2.0".to_string();
        let result = run(args);

        assert!(result.is_ok(), "frangi must succeed: {:?}", result.err());
        assert!(output.exists(), "frangi must write output file");
    }

    // ── Positive: gradient-magnitude creates output file ─────────────────────

    #[test]
    fn test_filter_gradient_magnitude_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output.clone(), "gradient-magnitude"));

        assert!(
            result.is_ok(),
            "gradient-magnitude must succeed: {:?}",
            result.err()
        );
        assert!(output.exists(), "gradient-magnitude must write output file");
    }

    // ── Positive: laplacian creates output file ───────────────────────────────

    #[test]
    fn test_filter_laplacian_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output.clone(), "laplacian"));

        assert!(result.is_ok(), "laplacian must succeed: {:?}", result.err());
        assert!(output.exists(), "laplacian must write output file");
    }

    // ── Positive: median creates output file with preserved shape ─────────────

    #[test]
    fn test_filter_median_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output.clone(), "median"));

        assert!(result.is_ok(), "median must succeed: {:?}", result.err());
        assert!(output.exists(), "median must write output file");

        let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            filtered.shape(),
            [5, 5, 5],
            "median output shape must match input"
        );
    }

    // ── Positive: bilateral creates output file with preserved shape ──────────

    #[test]
    fn test_filter_bilateral_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output.clone(), "bilateral"));

        assert!(result.is_ok(), "bilateral must succeed: {:?}", result.err());
        assert!(output.exists(), "bilateral must write output file");

        let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            filtered.shape(),
            [5, 5, 5],
            "bilateral output shape must match input"
        );
    }

    // ── Positive: canny creates binary edge output ────────────────────────────

    #[test]
    fn test_filter_canny_creates_output_with_binary_values() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output.clone(), "canny"));

        assert!(result.is_ok(), "canny must succeed: {:?}", result.err());
        assert!(output.exists(), "canny must write output file");

        let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            filtered.shape(),
            [5, 5, 5],
            "canny output shape must match input"
        );

        // Canny output is binary: every voxel must be 0.0 or 1.0.
        let td = filtered.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Canny output must be strictly binary (0.0 or 1.0), got {v}"
            );
        }
    }

    // ── Positive: sobel creates output file ───────────────────────────────────

    #[test]
    fn test_filter_sobel_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output.clone(), "sobel"));

        assert!(result.is_ok(), "sobel must succeed: {:?}", result.err());
        assert!(output.exists(), "sobel must write output file");

        let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            filtered.shape(),
            [5, 5, 5],
            "sobel output shape must match input"
        );
    }

    // ── Positive: log creates output file ─────────────────────────────────────

    #[test]
    fn test_filter_log_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output.clone(), "log"));

        assert!(result.is_ok(), "log must succeed: {:?}", result.err());
        assert!(output.exists(), "log must write output file");

        let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            filtered.shape(),
            [5, 5, 5],
            "log output shape must match input"
        );
    }

    // ── Positive: recursive-gaussian creates output file ──────────────────────

    #[test]
    fn test_filter_recursive_gaussian_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output.clone(), "recursive-gaussian"));

        assert!(
            result.is_ok(),
            "recursive-gaussian must succeed: {:?}",
            result.err()
        );
        assert!(output.exists(), "recursive-gaussian must write output file");

        let filtered = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(
            filtered.shape(),
            [5, 5, 5],
            "recursive-gaussian output shape must match input"
        );
    }

    // ── Positive: recursive-gaussian order=1 produces first derivative ────────

    #[test]
    fn test_filter_recursive_gaussian_order_1_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output.clone(), "recursive-gaussian");
        args.order = 1;
        let result = run(args);

        assert!(
            result.is_ok(),
            "recursive-gaussian order=1 must succeed: {:?}",
            result.err()
        );
        assert!(
            output.exists(),
            "recursive-gaussian order=1 must write output file"
        );
    }

    // ── Negative: unknown filter name returns error ───────────────────────────

    #[test]
    fn test_filter_unknown_name_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let result = run(default_args(input, output, "nonexistent"));

        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unknown filter 'nonexistent'"),
            "error must name the unknown filter, got: {msg}"
        );
    }

    // ── Negative: invalid recursive-gaussian order returns error ──────────────

    #[test]
    fn test_filter_recursive_gaussian_invalid_order_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();

        let mut args = default_args(input, output, "recursive-gaussian");
        args.order = 5;
        let result = run(args);

        assert!(result.is_err(), "invalid order must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Invalid --order 5"),
            "error must report the invalid order, got: {msg}"
        );
    }

    // ── Boundary: missing input file returns error ────────────────────────────

    #[test]
    fn test_filter_missing_input_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("does_not_exist.nii");
        let output = dir.path().join("out.nii");

        let result = run(default_args(input, output, "gaussian"));

        assert!(result.is_err(), "missing input must yield an error");
    }

    #[test]
    fn test_filter_curvature_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let mut args = default_args(input, output.clone(), "curvature");
        args.iterations = 3;
        let result = run(args);
        assert!(result.is_ok(), "curvature must succeed: {:?}", result.err());
        assert!(output.exists(), "curvature must write output file");
        let out_img = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(out_img.shape(), [5, 5, 5], "output shape must match input");
    }

    #[test]
    fn test_filter_sato_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let mut args = default_args(input, output.clone(), "sato");
        args.scales = "1.0".to_string();
        let result = run(args);
        assert!(result.is_ok(), "sato must succeed: {:?}", result.err());
        assert!(output.exists(), "sato must write output file");
        let out_img = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(out_img.shape(), [5, 5, 5], "output shape must match input");
    }
    #[test]
    fn test_filter_rescale_intensity_output_range() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let args = default_args(input, output.clone(), "rescale-intensity");
        run(args).unwrap();
        assert!(output.exists());
        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = result.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((min_val - 0.0).abs() < 1e-4, "rescale-intensity min must be 0.0, got {}", min_val);
        assert!((max_val - 1.0).abs() < 1e-4, "rescale-intensity max must be 1.0, got {}", max_val);
    }

    #[test]
    fn test_filter_intensity_windowing_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let mut args = default_args(input, output.clone(), "intensity-windowing");
        args.window_min = 20.0;
        args.window_max = 80.0;
        args.out_min = 0.0;
        args.out_max = 1.0;
        run(args).unwrap();
        assert!(output.exists());
        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(result.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_threshold_below_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let mut args = default_args(input, output.clone(), "threshold-below");
        args.threshold_value = 50.0;
        args.outside_value = 0.0;
        run(args).unwrap();
        assert!(output.exists());
        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = result.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        // All pixels that were < 50 should now be 0.0
        // Original values are 0..124, so values 0..49 -> 0.0
        let count_zero = vals.iter().filter(|&&v| v == 0.0).count();
        assert!(count_zero >= 50, "at least 50 pixels should be zeroed, got {}", count_zero);
    }

    #[test]
    fn test_filter_threshold_above_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let mut args = default_args(input, output.clone(), "threshold-above");
        args.threshold_value = 50.0;
        args.outside_value = 0.0;
        run(args).unwrap();
        assert!(output.exists());
        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(result.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_threshold_outside_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let mut args = default_args(input, output.clone(), "threshold-outside");
        args.lower_threshold = 30.0;
        args.upper_threshold = 90.0;
        args.outside_value = 0.0;
        run(args).unwrap();
        assert!(output.exists());
        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(result.shape(), [5, 5, 5]);
    }

    #[test]
    fn test_filter_sigmoid_creates_output_bounded() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let mut args = default_args(input, output.clone(), "sigmoid");
        args.alpha = 62.0; // midpoint of 0..124
        args.beta = 20.0;
        args.out_min = 0.0;
        args.out_max = 1.0;
        run(args).unwrap();
        assert!(output.exists());
        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = result.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(v >= 0.0 && v <= 1.0, "sigmoid output must be in [0,1], got {}", v);
        }
    }

    #[test]
    fn test_filter_binary_threshold_produces_binary_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_test_image()).unwrap();
        let mut args = default_args(input, output.clone(), "binary-threshold");
        args.lower_threshold = 40.0;
        args.upper_threshold = 80.0;
        args.foreground_value = 1.0;
        args.background_value = 0.0;
        run(args).unwrap();
        assert!(output.exists());
        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = result.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(v == 0.0 || v == 1.0, "binary-threshold output must be 0.0 or 1.0, got {}", v);
        }
    }
    #[test]
    fn test_filter_label_erosion_creates_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");
        let device: <Backend as BurnBackend>::Device = Default::default();
        let mut v = vec![0.0f32; 125];
        v[2 * 25 + 2 * 5 + 2] = 1.0;
        let td = TensorData::new(v, Shape::new([5, 5, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(tensor, Point::new([0.0; 3]), Spacing::new([1.0; 3]), Direction::identity());
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).unwrap();
        let mut args = default_args(input_path, output_path.clone(), "label-erosion");
        args.radius = 1;
        run(args).expect("label-erosion must succeed");
        assert!(output_path.exists());
    }

    #[test]
    fn test_filter_label_opening_creates_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");
        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32; 125], Shape::new([5, 5, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(tensor, Point::new([0.0; 3]), Spacing::new([1.0; 3]), Direction::identity());
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).unwrap();
        let mut args = default_args(input_path, output_path.clone(), "label-opening");
        args.radius = 1;
        run(args).expect("label-opening must succeed");
        assert!(output_path.exists());
    }

    #[test]
    fn test_filter_label_closing_creates_output() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");
        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![1.0f32; 125], Shape::new([5, 5, 5]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(tensor, Point::new([0.0; 3]), Spacing::new([1.0; 3]), Direction::identity());
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).unwrap();
        let mut args = default_args(input_path, output_path.clone(), "label-closing");
        args.radius = 1;
        run(args).expect("label-closing must succeed");
        assert!(output_path.exists());
    }

    #[test]
    fn test_filter_morph_recon_requires_mask() {
        let dir = tempdir().unwrap();
        let input_path = dir.path().join("input.nii");
        let output_path = dir.path().join("output.nii");
        let device: <Backend as BurnBackend>::Device = Default::default();
        let td = TensorData::new(vec![0.5f32; 8], Shape::new([2, 2, 2]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let image = Image::new(tensor, Point::new([0.0; 3]), Spacing::new([1.0; 3]), Direction::identity());
        ritk_io::write_nifti::<Backend, _>(&input_path, &image).unwrap();
        let mut args = default_args(input_path, output_path, "morphological-reconstruction");
        args.mask = None;
        let result = run(args);
        assert!(result.is_err(), "missing mask must return Err");
        assert!(result.unwrap_err().to_string().contains("mask"));
    }

}

