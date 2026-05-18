//! `ritk filter` — image filtering command.
//!
//! Applies one of the following filters to a 3-D medical image:
//!
//! | Filter              | Parameters                                                          |
//! |---------------------|---------------------------------------------------------------------|
//! | `gaussian`          | `--sigma`                                                           |
//! | `n4-bias`           | `--levels`, `--iterations`                                          |
//! | `anisotropic`       | `--iterations`, `--conductance`                                     |
//! | `gradient-magnitude`| (uses image spacing)                                                |
//! | `laplacian`         | (uses image spacing)                                                |
//! | `frangi`            | `--scales`, `--alpha`, `--beta`, `--gamma`                         |
//! | `median`            | `--radius`                                                          |
//! | `bilateral`         | `--sigma-spatial`, `--sigma-range`                                  |
//! | `canny`             | `--sigma`, `--low`, `--high`                                        |
//! | `sobel`             | (uses image spacing)                                                |
//! | `log`               | `--sigma`                                                           |
//! | `recursive-gaussian`| `--sigma`, `--order`                                                |
//! | `curvature`         | `--iterations`, `--time-step`                                       |
//! | `sato`              | `--scales`, `--alpha`                                               |
//! | `discrete-gaussian` | `--variance`, `--maximum-error`, `--use-image-spacing`              |
//! | `bed-separation`    | `--body-threshold`, `--closing-radius`, `--opening-radius`, `--outside-value` |

use anyhow::{anyhow, Result};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

pub(crate) use super::Backend;
use super::{read_image, write_image_inferred};
#[cfg(test)]
use ritk_core::image::Image;

mod intensity;
mod morphology;
mod smoothing;
mod spatial;
#[path = "spatial_impl.rs"]
mod spatial_file;

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `filter` subcommand.
#[derive(Args, Debug)]
pub struct FilterArgs {
    /// Input image path. Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output image path. Format is inferred from the file extension.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Filter to apply.
    ///
    /// Accepted values: `gaussian`, `n4-bias`, `anisotropic`, `frangi`,
    /// `gradient-magnitude`, `laplacian`, `median`, `bilateral`, `canny`,
    /// `sobel`, `log`, `recursive-gaussian`, `curvature`, `sato`,
    /// `discrete-gaussian`, `bed-separation`, `cpr`.
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
    /// diffusion. Lower values preserve edges more aggressively.
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

    // ── Discrete Gaussian ─────────────────────────────────────────────────────
    /// Gaussian variance σ² in physical units².
    ///
    /// Used by: `discrete-gaussian`.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub variance: f64,

    /// Kernel truncation tolerance in (0, 1).
    ///
    /// Used by: `discrete-gaussian`.
    #[arg(long, default_value = "0.01", value_name = "FLOAT")]
    pub maximum_error: f64,

    /// Convert physical σ to pixel σ using image spacing.
    ///
    /// Used by: `discrete-gaussian`.
    #[arg(long, default_value = "true", value_name = "BOOL")]
    pub use_image_spacing: bool,

    /// Lower intensity threshold for the bed separation filter.
    ///
    /// Used by: `bed-separation`.
    #[arg(long, default_value = "-350.0", value_name = "FLOAT")]
    pub body_threshold: f32,

    /// Closing radius for the bed separation mask.
    ///
    /// Used by: `bed-separation`.
    #[arg(long, default_value = "2", value_name = "INT")]
    pub bed_closing_radius: usize,

    /// Opening radius for the bed separation mask.
    ///
    /// Used by: `bed-separation`.
    #[arg(long, default_value = "1", value_name = "INT")]
    pub bed_opening_radius: usize,

    /// Replacement value written outside the retained foreground mask.
    ///
    /// Used by: `bed-separation`.
    #[arg(long, default_value = "-1024.0", value_name = "FLOAT")]
    pub bed_outside_value: f32,

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

    // ── CPR (Curved Planar Reformation) ──────────────────────────────────
    /// Control points for the CPR path in physical coordinates `[z,y,x]`.
    /// Repeat this flag for each point: `--cpr-point 0,0,0 --cpr-point 10,0,0`.
    ///
    /// Used by: `cpr`.
    #[arg(long = "cpr-point", value_name = "Z,Y,X", num_args = 1..)]
    pub cpr_points: Vec<String>,
    /// Number of samples along the CPR path (output columns).
    ///
    /// Used by: `cpr`.
    #[arg(long, default_value = "256", value_name = "INT")]
    pub cpr_path_samples: u32,
    /// Cross-section half-width in physical units (mm) for CPR.
    ///
    /// Used by: `cpr`.
    #[arg(long, default_value = "10.0", value_name = "FLOAT")]
    pub cpr_half_width: f32,
    /// Number of cross-section samples (output rows) for CPR.
    ///
    /// Used by: `cpr`.
    #[arg(long, default_value = "64", value_name = "INT")]
    pub cpr_cross_samples: u32,
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
        "filter: starting input={} output={} filter={}",
        args.input.display(),
        args.output.display(),
        args.filter
    );

    match args.filter.as_str() {
        "gaussian" => smoothing::run_gaussian(&args),
        "n4-bias" => smoothing::run_n4_bias(&args),
        "anisotropic" => smoothing::run_anisotropic(&args),
        "gradient-magnitude" => spatial::run_gradient_magnitude(&args),
        "laplacian" => spatial::run_laplacian(&args),
        "frangi" => spatial::run_frangi(&args),
        "median" => spatial::run_median(&args),
        "bilateral" => spatial::run_bilateral(&args),
        "canny" => spatial::run_canny(&args),
        "sobel" => spatial::run_sobel(&args),
        "log" => spatial::run_log(&args),
        "recursive-gaussian" => spatial::run_recursive_gaussian(&args),
        "curvature" => smoothing::run_curvature(&args),
        "sato" => smoothing::run_sato(&args),
        "discrete-gaussian" => smoothing::run_discrete_gaussian(&args),
        "bed-separation" => intensity::run_bed_separation(&args),
        "cpr" => spatial::run_cpr(&args),
        "rescale-intensity" => intensity::run_rescale_intensity(&args),
        "intensity-windowing" => intensity::run_intensity_windowing(&args),
        "threshold-below" => intensity::run_threshold_below(&args),
        "threshold-above" => intensity::run_threshold_above(&args),
        "threshold-outside" => intensity::run_threshold_outside(&args),
        "sigmoid" => intensity::run_sigmoid(&args),
        "binary-threshold" => intensity::run_binary_threshold(&args),
        "grayscale-erosion" => morphology::run_grayscale_erosion(&args),
        "grayscale-dilation" => morphology::run_grayscale_dilation(&args),
        "white-top-hat" => morphology::run_white_top_hat(&args),
        "black-top-hat" => morphology::run_black_top_hat(&args),
        "hit-or-miss" => morphology::run_hit_or_miss(&args),
        "label-dilation" => morphology::run_label_dilation(&args),
        "label-erosion" => morphology::run_label_erosion(&args),
        "label-opening" => morphology::run_label_opening(&args),
        "label-closing" => morphology::run_label_closing(&args),
        "morphological-reconstruction" => morphology::run_morphological_reconstruction(&args),
        other => {
            let available = concat!(
                "gaussian, n4-bias, anisotropic, gradient-magnitude, ",
                "laplacian, frangi, median, bilateral, canny, sobel, log, ",
                "recursive-gaussian, curvature, sato, discrete-gaussian, ",
                "rescale-intensity, intensity-windowing, threshold-below, ",
                "threshold-above, threshold-outside, sigmoid, binary-threshold, ",
                "grayscale-erosion, grayscale-dilation, white-top-hat, ",
                "black-top-hat, hit-or-miss, label-dilation, label-erosion, ",
                "label-opening, label-closing, morphological-reconstruction, cpr."
            );
            Err(anyhow!(
                "Unknown filter '{}'. Available filters: {}",
                other,
                available
            ))
        }
    }
}

// ── Test helpers (shared across leaf modules) ─────────────────────────────────

/// Default `FilterArgs` builder — sets every field to a reasonable default
/// and lets the caller override what is needed.
#[cfg(test)]
pub(crate) fn default_args(input: PathBuf, output: PathBuf, filter: &str) -> FilterArgs {
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
        variance: 1.0,
        maximum_error: 0.01,
        use_image_spacing: true,
        body_threshold: -350.0,
        bed_closing_radius: 2,
        bed_opening_radius: 1,
        bed_outside_value: -1024.0,
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
        cpr_points: vec![],
        cpr_path_samples: 256,
        cpr_half_width: 10.0,
        cpr_cross_samples: 64,
    }
}

/// Build a 5×5×5 test image whose voxel values are `0, 1, 2, …, 124`.
#[cfg(test)]
pub(crate) fn make_test_image() -> Image<Backend, 3> {
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};

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

// ── Dispatch-level tests ──────────────────────────────────────────────────────
#[cfg(test)]
mod tests;
