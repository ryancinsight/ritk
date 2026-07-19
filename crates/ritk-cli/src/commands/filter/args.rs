use clap::Args;
use ritk_filter::SpacingMode;
use std::path::PathBuf;

/// Derivative order for `recursive-gaussian`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
pub enum CliDerivativeOrder {
    #[default]
    #[value(name = "0")]
    Zero,
    #[value(name = "1")]
    First,
    #[value(name = "2")]
    Second,
}

impl std::fmt::Display for CliDerivativeOrder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Zero => "0",
            Self::First => "1",
            Self::Second => "2",
        })
    }
}

/// Closed set of filter kinds the `filter` subcommand accepts.
///
/// Replaces the stringly-typed `filter: String` dispatch. clamps rejects
/// unknown values at parse time, so the runtime match is exhaustive.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterKind {
    Gaussian,
    #[value(name = "n4-bias")]
    N4Bias,
    Anisotropic,
    #[value(name = "gradient-magnitude")]
    GradientMagnitude,
    Laplacian,
    Frangi,
    Median,
    Bilateral,
    Canny,
    Sobel,
    Log,
    #[value(name = "recursive-gaussian")]
    RecursiveGaussian,
    Curvature,
    Sato,
    #[value(name = "discrete-gaussian")]
    DiscreteGaussian,
    #[value(name = "bed-separation")]
    BedSeparation,
    #[value(name = "rescale-intensity")]
    RescaleIntensity,
    #[value(name = "intensity-windowing")]
    IntensityWindowing,
    #[value(name = "threshold-below")]
    ThresholdBelow,
    #[value(name = "threshold-above")]
    ThresholdAbove,
    #[value(name = "threshold-outside")]
    ThresholdOutside,
    Sigmoid,
    #[value(name = "binary-threshold")]
    BinaryThreshold,
    #[value(name = "grayscale-erosion")]
    GrayscaleErosion,
    #[value(name = "grayscale-dilation")]
    GrayscaleDilation,
    #[value(name = "white-top-hat")]
    WhiteTopHat,
    #[value(name = "black-top-hat")]
    BlackTopHat,
    #[value(name = "hit-or-miss")]
    HitOrMiss,
    #[value(name = "label-dilation")]
    LabelDilation,
    #[value(name = "label-erosion")]
    LabelErosion,
    #[value(name = "label-opening")]
    LabelOpening,
    #[value(name = "label-closing")]
    LabelClosing,
    #[value(name = "morphological-reconstruction")]
    MorphologicalReconstruction,
    Cpr,
}

impl std::fmt::Display for FilterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Gaussian => "gaussian",
            Self::N4Bias => "n4-bias",
            Self::Anisotropic => "anisotropic",
            Self::GradientMagnitude => "gradient-magnitude",
            Self::Laplacian => "laplacian",
            Self::Frangi => "frangi",
            Self::Median => "median",
            Self::Bilateral => "bilateral",
            Self::Canny => "canny",
            Self::Sobel => "sobel",
            Self::Log => "log",
            Self::RecursiveGaussian => "recursive-gaussian",
            Self::Curvature => "curvature",
            Self::Sato => "sato",
            Self::DiscreteGaussian => "discrete-gaussian",
            Self::BedSeparation => "bed-separation",
            Self::RescaleIntensity => "rescale-intensity",
            Self::IntensityWindowing => "intensity-windowing",
            Self::ThresholdBelow => "threshold-below",
            Self::ThresholdAbove => "threshold-above",
            Self::ThresholdOutside => "threshold-outside",
            Self::Sigmoid => "sigmoid",
            Self::BinaryThreshold => "binary-threshold",
            Self::GrayscaleErosion => "grayscale-erosion",
            Self::GrayscaleDilation => "grayscale-dilation",
            Self::WhiteTopHat => "white-top-hat",
            Self::BlackTopHat => "black-top-hat",
            Self::HitOrMiss => "hit-or-miss",
            Self::LabelDilation => "label-dilation",
            Self::LabelErosion => "label-erosion",
            Self::LabelOpening => "label-opening",
            Self::LabelClosing => "label-closing",
            Self::MorphologicalReconstruction => "morphological-reconstruction",
            Self::Cpr => "cpr",
        })
    }
}

// â”€â”€ Per-family Args structs (#[command(flatten)] chunks) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

/// Smoothing Ïƒ (Gaussian family: gaussian, canny, log, recursive-gaussian).
#[derive(Args, Debug, Default)]
pub struct SmoothingArgs {
    /// Gaussian standard deviation in physical units (mm).
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub sigma: f64,
}

/// Iterative / diffusion parameters (n4-bias, anisotropic, curvature).
#[derive(Args, Debug, Default)]
pub struct DiffusionArgs {
    /// Number of multi-resolution pyramid levels for N4 bias-field correction.
    #[arg(long, default_value = "4", value_name = "INT")]
    pub levels: usize,
    /// Maximum number of optimizer iterations per pyramid level.
    #[arg(long, default_value = "50", value_name = "INT")]
    pub iterations: usize,
    /// Conductance parameter controlling edge sensitivity for anisotropic
    /// diffusion. Lower values preserve edges more aggressively.
    #[arg(long, default_value = "3.0", value_name = "FLOAT")]
    pub conductance: f64,
    /// Explicit Euler time step Î”t for curvature anisotropic diffusion.
    /// Stability requires Î”t â‰¤ 1/6 for unit spacing.
    #[arg(long, default_value = "0.0625", value_name = "FLOAT")]
    pub time_step: f64,
}

/// Vesselness parameters (frangi, sato).
#[derive(Args, Debug, Default)]
pub struct VesselnessArgs {
    /// Comma-separated list of vessel scale radii (mm) for multi-scale Frangi
    /// / Sato vesselness enhancement.
    #[arg(long, value_name = "FLOATS", value_delimiter = ',', default_values = ["0.5", "1.0", "2.0"])]
    pub scales: Vec<f64>,
    /// Frangi Î± (plate-like sensitivity) / Sato polarity weight.
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub alpha: f64,
    /// Frangi Î² parameter (blob-like sensitivity).
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub beta: f64,
    /// Frangi Î³ parameter (background-noise sensitivity).
    #[arg(long, default_value = "15.0", value_name = "FLOAT")]
    pub gamma: f64,
}

/// Edge parameters (bilateral, canny). Distinct field names avoid
/// `#[command(flatten)]` collisions with `SmoothingArgs::sigma`.
#[derive(Args, Debug, Default)]
pub struct EdgeArgs {
    /// Spatial Gaussian Ïƒ in voxels for the bilateral filter.
    #[arg(long, default_value = "3.0", value_name = "FLOAT")]
    pub sigma_spatial: f64,
    /// Intensity-range Gaussian Ïƒ for the bilateral filter.
    #[arg(long, default_value = "50.0", value_name = "FLOAT")]
    pub sigma_range: f64,
    /// Lower hysteresis threshold for the Canny edge detector.
    #[arg(long, default_value = "0.1", value_name = "FLOAT")]
    pub low: f32,
    /// Upper hysteresis threshold for the Canny edge detector.
    #[arg(long, default_value = "0.3", value_name = "FLOAT")]
    pub high: f32,
}

/// Discrete Gaussian parameters (variance-based smoothing).
#[derive(Args, Debug, Default)]
pub struct DiscreteArgs {
    /// Gaussian variance ÏƒÂ² in physical unitsÂ².
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub variance: f64,
    /// Kernel truncation tolerance in (0, 1).
    #[arg(long, default_value = "0.01", value_name = "FLOAT")]
    pub maximum_error: f64,
    /// Controls whether Gaussian variance is in physical or pixel units.
    ///
    /// `physical` (default): variance is in physical units; converted to pixel
    /// sigma using image spacing (`sigma_pixel = sqrt(v) / spacing`).
    /// `voxel`: treat variance as already in pixel units.
    #[arg(long = "spacing-mode", default_value = "physical", value_name = "MODE",
          value_parser = |s: &str| s.parse::<SpacingMode>().map_err(|e| e.to_string()))]
    pub spacing_mode: SpacingMode,
}

/// Neighbourhood radius (median + morphology family).
///
/// A radius of 1 produces a 3Ã—3Ã—3 kernel (27 samples per voxel).
#[derive(Args, Debug, Default)]
pub struct KernelArgs {
    /// Neighbourhood half-width in voxels.
    #[arg(long, default_value = "1", value_name = "INT")]
    pub radius: usize,
}

/// Derivative order for the recursive Gaussian filter.
#[derive(Args, Debug, Default)]
pub struct RecursiveArgs {
    /// Derivative order (0 = smoothing, 1 = first derivative, 2 = second).
    #[arg(long, default_value = "0", value_enum, value_name = "ORDER")]
    pub order: CliDerivativeOrder,
}

/// Bed-separation parameters.
///
/// Explicit `#[arg(long = "...")]` overrides preserve the historical
/// `--bed-closing-radius` / `--bed-opening-radius` / `--bed-outside-value`
/// flag names (named before the SRP-362-20 refactor split the flag prefix
/// into the `BedArgs` group).
#[derive(Args, Debug, Default)]
pub struct BedArgs {
    /// Lower intensity threshold for the bed separation filter.
    #[arg(long, default_value = "-350.0", value_name = "FLOAT")]
    pub body_threshold: f32,
    /// Closing radius for the bed separation mask.
    #[arg(long = "bed-closing-radius", default_value = "2", value_name = "INT")]
    pub closing_radius: usize,
    /// Opening radius for the bed separation mask.
    #[arg(long = "bed-opening-radius", default_value = "1", value_name = "INT")]
    pub opening_radius: usize,
    /// Replacement value written outside the retained foreground mask.
    #[arg(
        long = "bed-outside-value",
        default_value = "-1024.0",
        value_name = "FLOAT"
    )]
    pub outside_value: f32,
}

/// Output range (rescale-intensity, sigmoid, intensity-windowing).
#[derive(Args, Debug, Default)]
pub struct RangeArgs {
    /// Minimum output value.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub out_min: f32,
    /// Maximum output value.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub out_max: f32,
}

/// Window boundaries (intensity-windowing).
#[derive(Args, Debug, Default)]
pub struct WindowArgs {
    /// Minimum of intensity window.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub window_min: f32,
    /// Maximum of intensity window.
    #[arg(long, default_value = "255.0", value_name = "FLOAT")]
    pub window_max: f32,
}

/// Threshold band (threshold-outside, binary-threshold).
#[derive(Args, Debug, Default)]
pub struct BandArgs {
    /// Lower bound for threshold-outside and binary-threshold filters.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub lower_threshold: f32,
    /// Upper bound for threshold-outside and binary-threshold filters.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub upper_threshold: f32,
    /// Replacement value for pixels outside the threshold range.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub outside_value: f32,
    /// Foreground value for binary-threshold filter.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub foreground_value: f32,
    /// Background value for binary-threshold filter.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub background_value: f32,
}

/// Single-threshold parameter (threshold-below, threshold-above).
#[derive(Args, Debug, Default)]
pub struct ThresholdArgs {
    /// Threshold value for threshold-below and threshold-above filters.
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub threshold_value: f32,
}

/// Optional mask image path (morphological-reconstruction).
#[derive(Args, Debug, Default)]
pub struct MaskInputArgs {
    /// Path to a mask image for filters requiring two inputs.
    #[arg(long)]
    pub mask: Option<std::path::PathBuf>,
}

/// CPR (Curved Planar Reformation) parameters.
#[derive(Args, Debug, Default)]
pub struct CprArgs {
    /// Control points for the CPR path in physical coordinates `[z,y,x]`.
    /// Repeat this flag for each point: `--cpr-point 0,0,0 --cpr-point 10,0,0`.
    #[arg(long = "cpr-point", value_name = "Z,Y,X", num_args = 1..)]
    pub cpr_points: Vec<String>,
    /// Number of samples along the CPR path (output columns).
    #[arg(long, default_value = "256", value_name = "INT")]
    pub cpr_path_samples: u32,
    /// Cross-section half-width in physical units (mm) for CPR.
    #[arg(long, default_value = "10.0", value_name = "FLOAT")]
    pub cpr_half_width: f32,
    /// Number of cross-section samples (output rows) for CPR.
    #[arg(long, default_value = "64", value_name = "INT")]
    pub cpr_cross_samples: u32,
}

/// Sigmoid parameters.
///
/// `midpoint` and `steepness` are exposed via `--sigmoid-midpoint` and
/// `--sigmoid-steepness` (descriptive names), with `--alpha` and `--beta`
/// retained as aliases for backward compatibility with the historical
/// `FilterArgs` API where these fields lived flat alongside
/// `VesselnessArgs::alpha`/`beta`.
#[derive(Args, Debug, Default)]
pub struct SigmoidArgs {
    /// Sigmoid midpoint (intensity at which the output equals the middle of
    /// the output range).  Alias: `--alpha`.
    #[arg(
        long = "sigmoid-midpoint",
        alias = "alpha",
        default_value = "62.0",
        value_name = "FLOAT"
    )]
    pub midpoint: f32,
    /// Sigmoid steepness (controls the width of the transition band).
    /// Alias: `--beta`.
    #[arg(
        long = "sigmoid-steepness",
        alias = "beta",
        default_value = "20.0",
        value_name = "FLOAT"
    )]
    pub steepness: f32,
}

/// Arguments for the `filter` subcommand.
///
/// Composed of input/output + filter kind + 14 per-family `#[command(flatten)]`
/// Args chunks (SRP-362-20: major).
#[derive(Args, Debug)]
pub struct FilterArgs {
    /// Input image path. Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output image path. Format is inferred from the file extension.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Filter kind to apply.
    #[arg(long, value_enum, value_name = "KIND")]
    pub kind: FilterKind,

    #[command(flatten)]
    pub smoothing: SmoothingArgs,

    #[command(flatten)]
    pub diffusion: DiffusionArgs,

    #[command(flatten)]
    pub vesselness: VesselnessArgs,

    #[command(flatten)]
    pub edge: EdgeArgs,

    #[command(flatten)]
    pub discrete: DiscreteArgs,

    #[command(flatten)]
    pub kernel: KernelArgs,

    #[command(flatten)]
    pub recursive: RecursiveArgs,

    #[command(flatten)]
    pub bed: BedArgs,

    #[command(flatten)]
    pub range: RangeArgs,

    #[command(flatten)]
    pub window: WindowArgs,

    #[command(flatten)]
    pub band: BandArgs,

    #[command(flatten)]
    pub threshold: ThresholdArgs,

    #[command(flatten)]
    pub mask_input: MaskInputArgs,

    #[command(flatten)]
    pub cpr: CprArgs,

    #[command(flatten)]
    pub sigmoid: SigmoidArgs,
}
