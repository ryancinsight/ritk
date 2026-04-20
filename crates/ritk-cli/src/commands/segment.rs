//! `ritk segment` — image segmentation command.
//!
//! Applies one of the following segmentation algorithms to a 3-D medical image:
//!
//! | Method               | Algorithm                                      |
//! |----------------------|------------------------------------------------|
//! | `otsu`               | Single-threshold Otsu (maximises between-class variance) |
//! | `multi-otsu`         | Multi-class Otsu (K−1 thresholds, K classes)   |
//! | `connected-threshold`| BFS flood-fill region growing from a seed voxel|
//! | `li`                 | Li minimum cross-entropy threshold             |
//! | `yen`                | Yen maximum correlation criterion threshold    |
//! | `kapur`              | Kapur maximum entropy threshold                |
//! | `triangle`           | Triangle (geometric) threshold                 |
//! | `watershed`          | Watershed flooding segmentation                |
//! | `kmeans`             | K-Means intensity clustering                   |
//! | `distance-transform` | Euclidean distance transform of binary mask     |
//! | `fill-holes`          | 6-connected binary hole fill                    |
//! | `morphological-gradient` | Morphological gradient (dilation - erosion)  |
//! | `shape-detection`        | Shape Detection level set (Malladi, Sethian & Vemuri 1995) |
//! | `threshold-level-set`    | Threshold Level Set (Whitaker 1998)            |
//! | `laplacian-level-set`    | Laplacian Level Set                             |
//!
//! # Output
//! - `otsu`, `li`, `yen`, `kapur`, `triangle`: binary mask (0.0 / 1.0) + printed threshold.
//! - `multi-otsu`: label image (0.0, 1.0, …, K−1.0) + printed threshold list.
//! - `connected-threshold`: binary mask (0.0 / 1.0) + foreground voxel count.
//! - `watershed`: label image (0 = boundary, 1..K = basins).
//! - `kmeans`: label image (0..K−1 cluster indices).
//! - `distance-transform`: float distance map (Euclidean distance to nearest background voxel).
//! - `shape-detection`, `threshold-level-set`, `laplacian-level-set`: binary mask (0.0 / 1.0) + foreground voxel count.

use anyhow::{anyhow, Context, Result};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use ritk_core::segmentation::{
    connected_threshold, multi_otsu_threshold, otsu_threshold, BinaryFillHoles, KMeansSegmentation,
    KapurThreshold, LaplacianLevelSet, LiThreshold, MorphologicalGradient, MorphologicalOperation,
    MultiOtsuThreshold, OtsuThreshold, TriangleThreshold, WatershedSegmentation, YenThreshold,
};

use super::{read_image, write_image_inferred, Backend};

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `segment` subcommand.
#[derive(Args, Debug)]
pub struct SegmentArgs {
    /// Input image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub input: PathBuf,

    /// Output mask / label image path.  Format is inferred from the file extension.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Segmentation method.
    ///
    /// Accepted values: `otsu`, `multi-otsu`, `connected-threshold`, `li`,
    /// `yen`, `kapur`, `triangle`, `watershed`, `kmeans`, `distance-transform`.
    #[arg(long, value_name = "METHOD")]
    pub method: String,

    // ── Multi-Otsu / K-Means ──────────────────────────────────────────────
    /// Number of intensity classes for `multi-otsu` or `kmeans`.
    ///
    /// Must be ≥ 2 for `multi-otsu`.  For `kmeans`, must be ≥ 1.
    #[arg(long, default_value = "3", value_name = "INT")]
    pub classes: usize,

    // ── Connected-threshold ───────────────────────────────────────────────
    /// Inclusive lower intensity bound for `connected-threshold`.
    #[arg(long, value_name = "FLOAT")]
    pub lower: Option<f32>,

    /// Inclusive upper intensity bound for `connected-threshold`.
    #[arg(long, value_name = "FLOAT")]
    pub upper: Option<f32>,

    /// Seed voxel for `connected-threshold` in `Z,Y,X` index order.
    ///
    /// Example: `--seed 4,5,6` sets z=4, y=5, x=6.
    #[arg(long, value_name = "Z,Y,X")]
    pub seed: Option<String>,

    // -- Confidence-connected -----------------------------------------------
    /// Multiplier k for adaptive k*sigma window in confidence-connected region growing.
    #[arg(long, default_value = "2.5", value_name = "FLOAT")]
    pub multiplier: f32,

    /// Maximum iterations for confidence-connected region growing.
    #[arg(long, default_value = "15", value_name = "INT")]
    pub max_iterations: usize,
    // -- Neighborhood-connected ---------------------------------------------
    /// Neighbourhood half-radius (uniform, all 3 axes) for neighborhood-connected.
    /// Radius 1 => 3x3x3 neighbourhood.
    #[arg(long, default_value = "1", value_name = "INT")]
    pub neighborhood_radius: usize,
    // -- Shape-detection / Threshold-level-set / Laplacian-level-set ------
    /// Path to initial level set φ image for `shape-detection`, `threshold-level-set`, or `laplacian-level-set`.
    #[arg(long, value_name = "PATH")]
    pub initial_phi: Option<PathBuf>,
    /// Curvature weight for level-set methods.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub curvature_weight: f32,
    /// Gaussian pre-smoothing σ for shape-detection and laplacian-level-set.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub sigma: f32,
    /// Propagation (balloon) weight for level-set methods.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub propagation_weight: f32,
    /// Advection weight for shape-detection.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub advection_weight: f32,
    /// Edge-stopping sensitivity k for shape-detection.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub edge_k: f32,
    /// Euler forward time step Δt for level-set methods.
    #[arg(long, default_value = "0.05", value_name = "FLOAT")]
    pub dt: f32,
    /// Maximum iterations for level-set methods (shape-detection, threshold-level-set, laplacian-level-set).
    #[arg(
        long = "level-set-max-iterations",
        default_value = "200",
        value_name = "INT"
    )]
    pub level_set_max_iterations: usize,
    /// Convergence tolerance on max |dφ|/dt.
    #[arg(long, default_value = "1e-3", value_name = "FLOAT")]
    pub tolerance: f32,
    /// Lower intensity threshold for threshold-level-set.
    #[arg(long, value_name = "FLOAT")]
    pub lower_threshold: Option<f32>,
    /// Upper intensity threshold for threshold-level-set.
    #[arg(long, value_name = "FLOAT")]
    pub upper_threshold: Option<f32>,
    // -- Connected-components -------------------------------------------------
    /// Connectivity for connected-components: 6 (faces) or 26 (faces+edges+corners).
    #[arg(long, default_value = "6", value_name = "INT")]
    pub connectivity: u32,
    // -- Chan-Vese -----------------------------------------------------------
    /// Length (curvature) penalty weight μ for chan-vese.
    #[arg(long, default_value = "0.1", value_name = "FLOAT")]
    pub mu: f64,
    /// Area penalty weight ν for chan-vese. Positive penalises large regions.
    #[arg(long, default_value = "0.0", value_name = "FLOAT")]
    pub nu: f64,
    /// Data fidelity weight for inside region in chan-vese.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub lambda1: f64,
    /// Data fidelity weight for outside region in chan-vese.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub lambda2: f64,
    /// Regularisation width ε for Heaviside/Dirac in chan-vese.
    #[arg(long, default_value = "1.0", value_name = "FLOAT")]
    pub epsilon: f64,
}

impl Default for SegmentArgs {
    fn default() -> Self {
        Self {
            input: PathBuf::default(),
            output: PathBuf::default(),
            method: String::default(),
            classes: 3,
            lower: None,
            upper: None,
            seed: None,
            multiplier: 2.5,
            max_iterations: 15,
            neighborhood_radius: 1,
            initial_phi: None,
            curvature_weight: 1.0,
            propagation_weight: 1.0,
            advection_weight: 1.0,
            edge_k: 1.0,
            sigma: 1.0,
            dt: 0.05,
            level_set_max_iterations: 200,
            tolerance: 1e-3,
            lower_threshold: None,
            upper_threshold: None,
            connectivity: 6,
            mu: 0.1,
            nu: 0.0,
            lambda1: 1.0,
            lambda2: 1.0,
            epsilon: 1.0,
        }
    }
}

// ── Seed parsing ──────────────────────────────────────────────────────────────

/// Parse a `"Z,Y,X"` string into a `[usize; 3]` seed voxel index.
///
/// # Errors
/// Returns an error when the string does not contain exactly three
/// comma-separated non-negative integer tokens.
fn parse_seed(s: &str) -> Result<[usize; 3]> {
    let parts: Vec<&str> = s.splitn(4, ',').collect();
    if parts.len() != 3 {
        return Err(anyhow!(
            "Seed must be provided as Z,Y,X (three comma-separated integers), got: '{s}'"
        ));
    }
    let z = parts[0]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid Z component '{}' in seed '{s}'", parts[0]))?;
    let y = parts[1]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid Y component '{}' in seed '{s}'", parts[1]))?;
    let x = parts[2]
        .trim()
        .parse::<usize>()
        .with_context(|| format!("Invalid X component '{}' in seed '{s}'", parts[2]))?;
    Ok([z, y, x])
}

// ── Foreground count helper ───────────────────────────────────────────────────

/// Count the number of voxels with value > 0.5 in `image`.
///
/// Suitable for binary (0.0 / 1.0) masks produced by Otsu and
/// connected-threshold segmentation.
///
/// # Panics
/// Panics if the tensor data cannot be extracted as `f32`.
fn count_foreground(image: &ritk_core::image::Image<Backend, 3>) -> usize {
    let td = image.data().clone().into_data();
    let slice = td
        .as_slice::<f32>()
        .expect("segmentation output must contain f32 data");
    slice.iter().filter(|&&v| v > 0.5).count()
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `segment` subcommand.
///
/// Dispatches to the appropriate segmentation algorithm based on `args.method`,
/// writes the output mask / label image, and prints a one-line summary.
///
/// # Errors
/// Returns an error when:
/// - The input image cannot be read.
/// - A required argument for the chosen method is missing or malformed.
/// - The output image cannot be written.
/// - An unknown method name is supplied.
pub fn run(args: SegmentArgs) -> Result<()> {
    info!(
    input = %args.input.display(),
    output = %args.output.display(),
    method = %args.method,
    "segment: starting"
    );
    match args.method.as_str() {
        "otsu" => run_otsu(&args),
        "multi-otsu" => run_multi_otsu(&args),
        "connected-threshold" => run_connected_threshold(&args),
        "li" => run_li(&args),
        "yen" => run_yen(&args),
        "kapur" => run_kapur(&args),
        "triangle" => run_triangle(&args),
        "watershed" => run_watershed(&args),
        "kmeans" => run_kmeans(&args),
        "distance-transform" => run_distance_transform(&args),
        "fill-holes" => run_fill_holes(&args),
        "morphological-gradient" => run_morphological_gradient(&args),
        "confidence-connected" => run_confidence_connected(&args),
        "neighborhood-connected" => run_neighborhood_connected(&args),
        "shape-detection" => run_shape_detection(&args),
        "threshold-level-set" => run_threshold_level_set(&args),
        "laplacian-level-set" => run_laplacian_level_set(&args),
        "skeletonization" => run_skeletonization(&args),
        "connected-components" => run_connected_components(&args),
        "chan-vese" => run_chan_vese(&args),
        "geodesic-active-contour" => run_geodesic_active_contour(&args),
        other => Err(anyhow!(
            "Unknown segmentation method '{}'. \
        Supported methods: otsu, multi-otsu, connected-threshold, \
        li, yen, kapur, triangle, watershed, kmeans, distance-transform, \
        fill-holes, morphological-gradient, confidence-connected, \
        neighborhood-connected, shape-detection, threshold-level-set, \
        laplacian-level-set, skeletonization, connected-components, chan-vese, geodesic-active-contour.",
            other
        )),
    }
}

// ── Otsu thresholding ─────────────────────────────────────────────────────────

/// Apply single-threshold Otsu segmentation.
///
/// Computes the optimal threshold t* that maximises between-class variance,
/// then maps voxels ≥ t* to 1.0 (foreground) and voxels < t* to 0.0
/// (background).
fn run_otsu(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    // Compute threshold first (for reporting) then produce the binary mask.
    let threshold = otsu_threshold::<Backend, 3>(&image);
    let mask = OtsuThreshold::new().apply(&image);

    let n_foreground = count_foreground(&mask);
    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {}: found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        input      = %args.input.display(),
        threshold  = threshold,
        foreground = n_foreground,
        "segment: otsu complete"
    );

    Ok(())
}

// ── Multi-Otsu thresholding ───────────────────────────────────────────────────

/// Apply multi-class Otsu segmentation with `args.classes` intensity classes.
///
/// Computes K−1 optimal thresholds and maps each voxel to the class label
/// (0.0, 1.0, …, K−1.0) whose intensity interval it falls into.
fn run_multi_otsu(args: &SegmentArgs) -> Result<()> {
    if args.classes < 2 {
        return Err(anyhow!(
            "--classes must be ≥ 2 for multi-otsu, got {}",
            args.classes
        ));
    }

    let image = read_image(&args.input)?;

    // Compute thresholds for reporting.
    let thresholds = multi_otsu_threshold::<Backend, 3>(&image, args.classes);
    let labeled = MultiOtsuThreshold::new(args.classes).apply(&image);

    // Count non-background (class > 0) voxels for the summary line.
    let n_labeled = count_foreground(&labeled);

    write_image_inferred(&args.output, &labeled)?;

    // Format threshold list as "[T1, T2, …]".
    let thresh_str: Vec<String> = thresholds.iter().map(|t| format!("{t:.4}")).collect();
    println!(
        "Segmented {}: found {} labeled voxels / thresholds=[{}]",
        args.input.display(),
        n_labeled,
        thresh_str.join(", "),
    );

    info!(
        input      = %args.input.display(),
        classes    = args.classes,
        thresholds = ?thresholds,
        labeled    = n_labeled,
        "segment: multi-otsu complete"
    );

    Ok(())
}

// ── Connected-threshold region growing ───────────────────────────────────────

/// Apply connected-threshold BFS region growing from a user-specified seed.
///
/// Voxels reachable from `seed` whose intensities lie in `[lower, upper]`
/// are set to 1.0 (foreground); all others are set to 0.0 (background).
///
/// # Argument validation
/// `--lower`, `--upper`, and `--seed` are all required for this method.
fn run_connected_threshold(args: &SegmentArgs) -> Result<()> {
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

    // Validate seed bounds against image shape before calling the kernel so
    // that any out-of-bounds error is surfaced as a user-friendly message
    // rather than a panic from the core implementation.
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

    let mask = connected_threshold::<Backend>(&image, seed, lower, upper);
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
        input      = %args.input.display(),
        seed       = ?seed,
        lower      = lower,
        upper      = upper,
        foreground = n_foreground,
        "segment: connected-threshold complete"
    );

    Ok(())
}

// ── Li thresholding ───────────────────────────────────────────────────────────

/// Apply Li minimum cross-entropy thresholding.
///
/// Computes t* that minimises the cross-entropy between the image and its
/// binary thresholded version, then maps voxels ≥ t* to 1.0.
fn run_li(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    let filter = LiThreshold::new();
    let threshold = filter.compute(&image);
    let mask = filter.apply(&image);

    let n_foreground = count_foreground(&mask);
    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {} (li): found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        input      = %args.input.display(),
        threshold  = threshold,
        foreground = n_foreground,
        "segment: li complete"
    );

    Ok(())
}

// ── Yen thresholding ──────────────────────────────────────────────────────────

/// Apply Yen maximum correlation criterion thresholding.
///
/// Computes t* that maximises the correlation criterion, then maps
/// voxels ≥ t* to 1.0.
fn run_yen(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    let filter = YenThreshold::new();
    let threshold = filter.compute(&image);
    let mask = filter.apply(&image);

    let n_foreground = count_foreground(&mask);
    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {} (yen): found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        input      = %args.input.display(),
        threshold  = threshold,
        foreground = n_foreground,
        "segment: yen complete"
    );

    Ok(())
}

// ── Kapur thresholding ────────────────────────────────────────────────────────

/// Apply Kapur maximum entropy thresholding.
///
/// Computes t* that maximises the sum of foreground and background
/// entropies, then maps voxels ≥ t* to 1.0.
fn run_kapur(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    let filter = KapurThreshold::new();
    let threshold = filter.compute(&image);
    let mask = filter.apply(&image);

    let n_foreground = count_foreground(&mask);
    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {} (kapur): found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        input      = %args.input.display(),
        threshold  = threshold,
        foreground = n_foreground,
        "segment: kapur complete"
    );

    Ok(())
}

// ── Triangle thresholding ─────────────────────────────────────────────────────

/// Apply Triangle (geometric) thresholding.
///
/// Constructs a line between the histogram peak and the histogram tail,
/// then selects the bin with maximum perpendicular distance as the
/// threshold.  Maps voxels ≥ t* to 1.0.
fn run_triangle(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    let filter = TriangleThreshold::new();
    let threshold = filter.compute(&image);
    let mask = filter.apply(&image);

    let n_foreground = count_foreground(&mask);
    write_image_inferred(&args.output, &mask)?;

    println!(
        "Segmented {} (triangle): found {} foreground voxels / threshold={:.4}",
        args.input.display(),
        n_foreground,
        threshold,
    );

    info!(
        input      = %args.input.display(),
        threshold  = threshold,
        foreground = n_foreground,
        "segment: triangle complete"
    );

    Ok(())
}

// ── Watershed segmentation ────────────────────────────────────────────────────

/// Apply watershed flooding segmentation.
///
/// The input should be a scalar 3-D image (e.g. gradient magnitude).
/// Returns a label image where label 0 = watershed boundary and
/// labels 1..K = catchment basin indices.
fn run_watershed(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    let ws = WatershedSegmentation::new();
    let labeled = ws.apply(&image)?;

    // Count distinct non-zero labels for summary.
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
        input   = %args.input.display(),
        basins  = n_basins,
        "segment: watershed complete"
    );

    Ok(())
}

// ── K-Means clustering ────────────────────────────────────────────────────────

/// Apply K-Means intensity clustering.
///
/// Each voxel in the output contains its assigned cluster index (0..K−1)
/// as `f32`.  Spatial metadata is preserved.
fn run_kmeans(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;

    let km = KMeansSegmentation::new(args.classes);
    let labeled = km.apply(&image);

    write_image_inferred(&args.output, &labeled)?;

    println!(
        "Segmented {} (kmeans): k={} clusters",
        args.input.display(),
        args.classes,
    );

    info!(
        input   = %args.input.display(),
        k       = args.classes,
        "segment: kmeans complete"
    );

    Ok(())
}

// ── Distance transform ────────────────────────────────────────────────────────

/// Compute the Euclidean distance transform of a binary mask.
///
/// The input is binarised at threshold 0.5 (voxels > 0.5 = foreground).
/// The output is a float image where each foreground voxel contains the
/// Euclidean distance (in voxel units) to the nearest background voxel.
/// Background voxels have value 0.0.
fn run_distance_transform(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::distance_transform;

    let image = read_image(&args.input)?;

    let dt = distance_transform(&image, 0.5);

    write_image_inferred(&args.output, &dt)?;

    println!(
        "Computed distance-transform for {} \u{2192} {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        input  = %args.input.display(),
        output = %args.output.display(),
        "segment: distance-transform complete"
    );

    Ok(())
}

/// Apply binary hole filling.
///
/// The input must be a binary mask (0.0 / 1.0). All background voxels not
/// reachable from the border are converted to foreground.
fn run_fill_holes(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let filled = BinaryFillHoles.apply(&image);

    write_image_inferred(&args.output, &filled)?;

    println!(
        "Segmented {} (fill-holes) -> {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        input  = %args.input.display(),
        output = %args.output.display(),
        "segment: fill-holes complete"
    );

    Ok(())
}

/// Apply binary morphological gradient.
///
/// Produces a boundary mask from the binary input via dilation AND NOT erosion.
fn run_morphological_gradient(args: &SegmentArgs) -> Result<()> {
    let image = read_image(&args.input)?;
    let gradient = MorphologicalGradient::new(1).apply(&image);

    write_image_inferred(&args.output, &gradient)?;

    println!(
        "Segmented {} (morphological-gradient) -> {}",
        args.input.display(),
        args.output.display(),
    );

    info!(
        input  = %args.input.display(),
        output = %args.output.display(),
        "segment: morphological-gradient complete"
    );

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────
// -- Confidence-connected region growing --------------------------------------

fn run_confidence_connected(args: &SegmentArgs) -> Result<()> {
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

    info!(
        input = %args.input.display(), seed = ?seed, lower = lower, upper = upper,
        multiplier = args.multiplier, foreground = n_foreground,
        "segment: confidence-connected complete"
    );
    Ok(())
}

// -- Neighbourhood-connected region growing -----------------------------------

fn run_neighborhood_connected(args: &SegmentArgs) -> Result<()> {
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

    info!(
        input = %args.input.display(), seed = ?seed, lower = lower, upper = upper,
        radius = r, foreground = n_foreground,
        "segment: neighborhood-connected complete"
    );
    Ok(())
}

// -- Shape-detection level set ------------------------------------------

fn run_shape_detection(args: &SegmentArgs) -> Result<()> {
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
        input = %args.input.display(),
        output = %args.output.display(),
        foreground = n_foreground,
        "segment: shape-detection complete"
    );
    Ok(())
}

// -- Threshold level set --------------------------------------------------

fn run_threshold_level_set(args: &SegmentArgs) -> Result<()> {
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
        input = %args.input.display(),
        output = %args.output.display(),
        lower = lower,
        upper = upper,
        foreground = n_foreground,
        "segment: threshold-level-set complete"
    );
    Ok(())
}

// -- Laplacian level set --------------------------------------------------

fn run_laplacian_level_set(args: &SegmentArgs) -> Result<()> {
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

    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        foreground = n_foreground,
        propagation = args.propagation_weight,
        curvature = args.curvature_weight,
        sigma = args.sigma,
        "segment: laplacian-level-set complete"
    );
    Ok(())
}

// -- Connected components -------------------------------------------------

fn run_connected_components(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::ConnectedComponentsFilter;

    let image = read_image(&args.input)?;
    let mut filter = ConnectedComponentsFilter::new();
    filter.connectivity = args.connectivity;
    let (labels, stats) = filter.apply(&image);
    write_image_inferred(&args.output, &labels)?;

    println!(
        "Labeled {}: connected-components found {} components (connectivity={})",
        args.input.display(),
        stats.len(),
        args.connectivity,
    );
    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        n_components = stats.len(),
        connectivity = args.connectivity,
        "segment: connected-components complete"
    );
    Ok(())
}

// -- Chan-Vese active contours --------------------------------------------

fn run_chan_vese(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::ChanVeseSegmentation;

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
    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        foreground = n_foreground,
        mu = args.mu,
        nu = args.nu,
        lambda1 = args.lambda1,
        lambda2 = args.lambda2,
        "segment: chan-vese complete"
    );
    Ok(())
}

// -- Geodesic active contour ----------------------------------------------

fn run_geodesic_active_contour(args: &SegmentArgs) -> Result<()> {
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
    info!(
        input = %args.input.display(),
        output = %args.output.display(),
        foreground = n_foreground,
        propagation = args.propagation_weight,
        curvature = args.curvature_weight,
        advection = args.advection_weight,
        "segment: geodesic-active-contour complete"
    );
    Ok(())
}

// -- Skeletonization ------------------------------------------------------

fn run_skeletonization(args: &SegmentArgs) -> Result<()> {
    use ritk_core::segmentation::Skeletonization;

    let image = read_image(&args.input)?;
    let skeleton = Skeletonization::new().apply::<_, 3>(&image);
    let n_skeleton = count_foreground(&skeleton);

    write_image_inferred(&args.output, &skeleton)?;

    println!(
        "Computed skeleton for {} -> {} ({} skeleton voxels)",
        args.input.display(),
        args.output.display(),
        n_skeleton,
    );

    info!(
        input = %args.input.display(), output = %args.output.display(),
        skeleton = n_skeleton, "segment: skeletonization complete"
    );
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

    // ── Test image factories ──────────────────────────────────────────────────

    /// Build a 4×4×4 bimodal image.
    ///
    /// The first half of voxels (flat indices 0–31) have intensity 20.0;
    /// the second half (32–63) have intensity 200.0.
    /// The analytically correct Otsu threshold lies between 20.0 and 200.0.
    fn make_bimodal_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..64)
            .map(|i| if i < 32 { 20.0_f32 } else { 200.0_f32 })
            .collect();
        let td = TensorData::new(values, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 6×6×6 trimodal image for multi-Otsu tests.
    ///
    /// Voxels are split into three equal groups with intensities 30, 130, 230.
    fn make_trimodal_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let n = 6 * 6 * 6; // 216
        let values: Vec<f32> = (0..n)
            .map(|i| {
                if i < n / 3 {
                    30.0_f32
                } else if i < 2 * n / 3 {
                    130.0_f32
                } else {
                    230.0_f32
                }
            })
            .collect();
        let td = TensorData::new(values, Shape::new([6, 6, 6]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 5×5×5 image with a high-intensity sphere at the centre.
    ///
    /// Centre voxel (2,2,2) and its 6 face-adjacent neighbours have intensity
    /// 200.0; all other voxels have intensity 10.0.
    fn make_sphere_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let (nz, ny, nx) = (5usize, 5usize, 5usize);
        let mut values = vec![10.0_f32; nz * ny * nx];
        let high_indices: &[(usize, usize, usize)] = &[
            (2, 2, 2), // centre
            (1, 2, 2), // −Z
            (3, 2, 2), // +Z
            (2, 1, 2), // −Y
            (2, 3, 2), // +Y
            (2, 2, 1), // −X
            (2, 2, 3), // +X
        ];
        for &(z, y, x) in high_indices {
            values[z * ny * nx + y * nx + x] = 200.0;
        }
        let td = TensorData::new(values, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 4×4×4 binary image: first 32 voxels = 1.0 (foreground),
    /// remaining 32 = 0.0 (background).  Used for distance-transform tests.
    fn make_binary_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..64)
            .map(|i| if i < 32 { 1.0_f32 } else { 0.0_f32 })
            .collect();
        let td = TensorData::new(values, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    /// Build a 4×4×4 image with a smooth ramp 0..63 for watershed / gradient
    /// tests.
    fn make_ramp_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let td = TensorData::new(values, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    // ── Helper: default SegmentArgs ───────────────────────────────────────────

    fn default_args(input: PathBuf, output: PathBuf, method: &str) -> SegmentArgs {
        SegmentArgs {
            input,
            output,
            method: method.to_string(),
            ..Default::default()
        }
    }

    // ── Positive: Otsu creates binary output file ─────────────────────────────

    /// Otsu segmentation must produce a file with the correct shape.
    #[test]
    fn test_segment_otsu_creates_output_file_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "otsu")).unwrap();

        assert!(output.exists(), "output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [4, 4, 4], "output shape must match input");
    }

    // ── Positive: Otsu output is strictly binary ──────────────────────────────

    /// Every voxel in the Otsu output mask must be exactly 0.0 or 1.0.
    #[test]
    fn test_segment_otsu_output_is_strictly_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("mask.mha");

        ritk_io::write_metaimage(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "otsu")).unwrap();

        let mask = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Otsu output must be strictly binary (0.0 or 1.0), got {v}"
            );
        }
    }

    // ── Positive: Otsu threshold is between the two modes ─────────────────────

    /// For a bimodal image with modes at 20 and 200, the Otsu threshold must
    /// lie strictly between 20 and 200.
    #[test]
    fn test_segment_otsu_threshold_between_modes() {
        let image = make_bimodal_image();
        let threshold = otsu_threshold::<Backend, 3>(&image);
        assert!(
            threshold > 20.0 && threshold < 200.0,
            "Otsu threshold {threshold} must lie between the two modes (20, 200)"
        );
    }

    // ── Positive: Otsu foreground count matches high-intensity voxels ─────────

    /// In the bimodal image the high-intensity half (32 voxels at 200.0)
    /// must become the foreground class.
    #[test]
    fn test_segment_otsu_foreground_count_equals_high_mode_voxels() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "otsu")).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let foreground = count_foreground(&mask);
        // The bimodal image has exactly 32 voxels at 200.0.
        assert_eq!(
            foreground, 32,
            "Otsu must label exactly 32 high-intensity voxels as foreground"
        );
    }

    // ── Positive: Multi-Otsu creates labeled output ────────────────────────────

    /// Multi-Otsu with 3 classes must create an output file with the correct shape.
    #[test]
    fn test_segment_multi_otsu_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("labels.nii");

        ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "multi-otsu")).unwrap();

        assert!(output.exists(), "output label image must be created");
        let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(labels.shape(), [6, 6, 6], "label shape must match input");
    }

    // ── Positive: Multi-Otsu labels are in valid set ───────────────────────────

    /// For K=3 classes, every voxel label must be in {0.0, 1.0, 2.0}.
    #[test]
    fn test_segment_multi_otsu_labels_in_valid_set() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("labels.mha");

        ritk_io::write_metaimage(&input, &make_trimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "multi-otsu")).unwrap();

        let labels = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let td = labels.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        let valid = [0.0_f32, 1.0_f32, 2.0_f32];
        for &v in values {
            assert!(
                valid.iter().any(|&vv| (v - vv).abs() < 1e-4),
                "label value {v} is not in the valid set {{0, 1, 2}} for K=3"
            );
        }
    }

    // ── Positive: Multi-Otsu returns K-1 thresholds ───────────────────────────

    /// For K=3 classes, `multi_otsu_threshold` must return exactly 2 thresholds,
    /// both lying within the image's intensity range.
    #[test]
    fn test_segment_multi_otsu_returns_k_minus_1_thresholds() {
        let image = make_trimodal_image();
        let thresholds = multi_otsu_threshold::<Backend, 3>(&image, 3);
        assert_eq!(
            thresholds.len(),
            2,
            "K=3 must produce exactly 2 thresholds, got {:?}",
            thresholds
        );
        for &t in &thresholds {
            assert!(
                t >= 30.0 && t <= 230.0,
                "threshold {t} must lie within the image intensity range [30, 230]"
            );
        }
        // Thresholds must be strictly increasing.
        assert!(
            thresholds[0] < thresholds[1],
            "thresholds must be strictly increasing: {:?}",
            thresholds
        );
    }

    // ── Positive: Connected-threshold grows sphere region ─────────────────────

    /// Seeding at the centre of the sphere must grow exactly the 7 high-intensity
    /// voxels (centre + 6 face-adjacent neighbours).
    #[test]
    fn test_segment_connected_threshold_grows_sphere_from_centre_seed() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: Some("2,2,2".to_string()),
            multiplier: 2.5,
            ..Default::default()
        })
        .unwrap();

        assert!(output.exists(), "output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let foreground = count_foreground(&mask);
        assert_eq!(
            foreground, 7,
            "connected-threshold from centre seed must grow exactly 7 sphere voxels"
        );
    }

    // ── Positive: Connected-threshold output is strictly binary ───────────────

    #[test]
    fn test_segment_connected_threshold_output_is_strictly_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.mha");
        let output = dir.path().join("mask.mha");

        ritk_io::write_metaimage(&input, &make_sphere_image()).unwrap();

        run(SegmentArgs {
            input: input.clone(),
            output: output.clone(),
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: Some("2,2,2".to_string()),
            multiplier: 2.5,
            ..Default::default()
        })
        .unwrap();

        let mask = ritk_io::read_metaimage::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "connected-threshold output must be strictly binary, got {v}"
            );
        }
    }

    // ── Positive: Li threshold creates binary output ──────────────────────────

    /// Li thresholding on a bimodal image must produce a binary mask with the
    /// threshold between the two modes.
    #[test]
    fn test_segment_li_creates_output_and_threshold_between_modes() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "li")).unwrap();

        assert!(output.exists(), "li output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [4, 4, 4], "output shape must match input");

        // Verify binary output.
        let td = mask.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Li output must be strictly binary, got {v}"
            );
        }

        // Verify threshold is between modes.
        let threshold = LiThreshold::new().compute(&make_bimodal_image());
        assert!(
            threshold > 20.0 && threshold < 200.0,
            "Li threshold {threshold} must lie between modes (20, 200)"
        );
    }

    // ── Positive: Yen threshold creates binary output ─────────────────────────

    #[test]
    fn test_segment_yen_creates_output_and_threshold_between_modes() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "yen")).unwrap();

        assert!(output.exists(), "yen output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [4, 4, 4]);

        let td = mask.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Yen output must be strictly binary, got {v}"
            );
        }

        let threshold = YenThreshold::new().compute(&make_bimodal_image());
        assert!(
            threshold > 20.0 && threshold < 200.0,
            "Yen threshold {threshold} must lie between modes (20, 200)"
        );
    }

    // ── Positive: Kapur threshold creates binary output ───────────────────────

    #[test]
    fn test_segment_kapur_creates_output_and_threshold_between_modes() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "kapur")).unwrap();

        assert!(output.exists(), "kapur output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [4, 4, 4]);

        let td = mask.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Kapur output must be strictly binary, got {v}"
            );
        }

        let threshold = KapurThreshold::new().compute(&make_bimodal_image());
        assert!(
            threshold >= 20.0 && threshold <= 200.0,
            "Kapur threshold {threshold} must lie within mode range [20, 200]"
        );
    }

    // ── Positive: Triangle threshold creates binary output ────────────────────

    #[test]
    fn test_segment_triangle_creates_output_and_threshold_between_modes() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "triangle")).unwrap();

        assert!(output.exists(), "triangle output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [4, 4, 4]);

        let td = mask.data().clone().into_data();
        let values = td.as_slice::<f32>().unwrap();
        for &v in values {
            assert!(
                v == 0.0 || v == 1.0,
                "Triangle output must be strictly binary, got {v}"
            );
        }

        let threshold = TriangleThreshold::new().compute(&make_bimodal_image());
        assert!(
            threshold > 20.0 && threshold < 200.0,
            "Triangle threshold {threshold} must lie between modes (20, 200)"
        );
    }

    // ── Positive: Watershed creates output with basin labels ──────────────────

    #[test]
    fn test_segment_watershed_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("labels.nii");

        ritk_io::write_nifti(&input, &make_ramp_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "watershed")).unwrap();

        assert!(output.exists(), "watershed output must be created");
        let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(labels.shape(), [4, 4, 4], "shape must be preserved");

        // All label values must be non-negative (0 = boundary, >=1 = basin).
        let td = labels.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(v >= 0.0, "watershed labels must be non-negative, got {v}");
        }
    }

    // ── Positive: K-Means creates output with cluster labels ──────────────────

    #[test]
    fn test_segment_kmeans_creates_output_with_valid_labels() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("labels.nii");

        ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "kmeans")).unwrap();

        assert!(output.exists(), "kmeans output must be created");
        let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(labels.shape(), [6, 6, 6]);

        // Labels must be in [0, classes-1] = [0, 2].
        let td = labels.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(
                v >= 0.0 && v < 3.0 + 0.5,
                "kmeans label {v} must be in [0, 2]"
            );
        }

        // With 3 clusters on a trimodal image, we expect 3 distinct label values.
        let mut unique: Vec<f32> = vals.to_vec();
        unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique.dedup_by(|a, b| (*a - *b).abs() < 0.01);
        assert_eq!(
            unique.len(),
            3,
            "trimodal image with k=3 must produce exactly 3 distinct labels, got {:?}",
            unique
        );
    }

    // ── Positive: Distance transform creates output ───────────────────────────

    #[test]
    fn test_segment_distance_transform_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("dt.nii");

        ritk_io::write_nifti(&input, &make_binary_image()).unwrap();

        run(default_args(
            input.clone(),
            output.clone(),
            "distance-transform",
        ))
        .unwrap();

        assert!(output.exists(), "distance-transform output must be created");
        let dt = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(dt.shape(), [4, 4, 4], "shape must be preserved");

        // Distance values must be non-negative.
        let td = dt.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(
                v >= 0.0,
                "distance transform values must be non-negative, got {v}"
            );
        }

        // Background voxels (second half, value=0.0 in input) must have EDT = 0.0.
        // Foreground voxels (first half, value=1.0 in input) must have EDT > 0.0
        // except possibly those adjacent to the background boundary — but at
        // minimum the first foreground voxel (index 0) in a 4×4×4 grid with
        // the boundary at z=2 should have a positive distance.
        let has_positive = vals.iter().any(|&v| v > 0.0);
        assert!(
            has_positive,
            "distance-transform must produce at least one positive value for a non-trivial mask"
        );
    }

    // ── Negative: unknown method returns descriptive error ────────────────────

    #[test]
    fn test_segment_unknown_method_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "nonexistent".to_string(),
            classes: 3,
            lower: None,
            upper: None,
            seed: None,
            multiplier: 2.5,
            ..Default::default()
        });

        assert!(result.is_err(), "unknown method must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unknown segmentation method 'nonexistent'"),
            "error must name the unsupported method, got: {msg}"
        );
    }

    // ── Negative: connected-threshold missing --lower ─────────────────────────

    #[test]
    fn test_segment_connected_threshold_missing_lower_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: None, // deliberately omitted
            upper: Some(255.0),
            seed: Some("2,2,2".to_string()),
            multiplier: 2.5,
            ..Default::default()
        });

        assert!(result.is_err(), "missing --lower must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--lower is required"),
            "error must name the missing argument, got: {msg}"
        );
    }

    // ── Negative: connected-threshold missing --upper ─────────────────────────

    #[test]
    fn test_segment_connected_threshold_missing_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: None, // deliberately omitted
            seed: Some("2,2,2".to_string()),
            multiplier: 2.5,
            ..Default::default()
        });

        assert!(result.is_err(), "missing --upper must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--upper is required"),
            "error must name the missing argument, got: {msg}"
        );
    }

    // ── Negative: connected-threshold missing --seed ──────────────────────────

    #[test]
    fn test_segment_connected_threshold_missing_seed_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: None, // deliberately omitted
            multiplier: 2.5,
            ..Default::default()
        });

        assert!(result.is_err(), "missing --seed must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--seed is required"),
            "error must name the missing argument, got: {msg}"
        );
    }

    // ── Negative: connected-threshold lower > upper ───────────────────────────

    #[test]
    fn test_segment_connected_threshold_lower_gt_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(255.0),
            upper: Some(100.0), // lower > upper
            seed: Some("2,2,2".to_string()),
            multiplier: 2.5,
            ..Default::default()
        });

        assert!(result.is_err(), "lower > upper must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("must be \u{2264}")
                || msg.contains("must be <=")
                || msg.contains('\u{2264}'),
            "error must explain the bound constraint, got: {msg}"
        );
    }

    // ── Negative: out-of-bounds seed returns error ────────────────────────────

    #[test]
    fn test_segment_connected_threshold_out_of_bounds_seed_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: Some("99,99,99".to_string()), // far out of [5,5,5] bounds
            multiplier: 2.5,
            ..Default::default()
        });

        assert!(result.is_err(), "out-of-bounds seed must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("out of bounds"),
            "error must explain the bounds problem, got: {msg}"
        );
    }

    // ── Negative: malformed seed string returns error ─────────────────────────

    #[test]
    fn test_segment_malformed_seed_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "connected-threshold".to_string(),
            classes: 3,
            lower: Some(100.0),
            upper: Some(255.0),
            seed: Some("2,2".to_string()), // only two components
            multiplier: 2.5,
            ..Default::default()
        });

        assert!(result.is_err(), "malformed seed must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Z,Y,X"),
            "error must explain the expected format, got: {msg}"
        );
    }

    // ── Negative: multi-otsu classes < 2 returns error ───────────────────────

    #[test]
    fn test_segment_multi_otsu_classes_lt_2_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("out.nii");
        ritk_io::write_nifti(&input, &make_trimodal_image()).unwrap();

        let result = run(SegmentArgs {
            input,
            output,
            method: "multi-otsu".to_string(),
            classes: 1, // invalid
            lower: None,
            upper: None,
            seed: None,
            multiplier: 2.5,
            ..Default::default()
        });

        assert!(result.is_err(), "classes < 2 must yield an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("\u{2265} 2"),
            "error must state the minimum class count, got: {msg}"
        );
    }

    // ── Boundary: parse_seed correct output ───────────────────────────────────

    #[test]
    fn test_parse_seed_valid_input() {
        let seed = parse_seed("4,5,6").unwrap();
        assert_eq!(seed, [4, 5, 6]);
    }

    #[test]
    fn test_parse_seed_with_spaces() {
        let seed = parse_seed("1, 2, 3").unwrap();
        assert_eq!(seed, [1, 2, 3]);
    }

    #[test]
    fn test_parse_seed_too_few_components_returns_error() {
        assert!(parse_seed("1,2").is_err());
    }

    #[test]
    fn test_parse_seed_non_numeric_component_returns_error() {
        assert!(parse_seed("1,two,3").is_err());
    }

    #[test]
    fn test_parse_seed_negative_component_returns_error() {
        // Negative values cannot be parsed as usize.
        assert!(parse_seed("1,-2,3").is_err());
    }

    // ── Positive: Li foreground count matches high-mode voxels ────────────────

    #[test]
    fn test_segment_li_foreground_count() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "li")).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let foreground = count_foreground(&mask);
        assert_eq!(
            foreground, 32,
            "Li must label exactly 32 high-intensity voxels as foreground"
        );
    }

    // ── Positive: Yen foreground count matches high-mode voxels ───────────────

    #[test]
    fn test_segment_yen_foreground_count() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "yen")).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let foreground = count_foreground(&mask);
        assert_eq!(
            foreground, 32,
            "Yen must label exactly 32 high-intensity voxels as foreground"
        );
    }

    // ── Positive: Kapur foreground count matches high-mode voxels ─────────────

    #[test]
    fn test_segment_kapur_foreground_count() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "kapur")).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let foreground = count_foreground(&mask);
        // Kapur maximum-entropy threshold can land on the low-mode boundary
        // (threshold=20) for equal-count bimodal data, labelling all 64 voxels
        // as foreground (≥20).  Both 32 and 64 are valid outcomes.
        assert!(
            foreground == 32 || foreground == 64,
            "Kapur must label either 32 (threshold>20) or 64 (threshold=20) voxels as foreground, got {foreground}"
        );
    }

    // ── Positive: Triangle foreground count matches high-mode voxels ──────────

    #[test]
    fn test_segment_triangle_foreground_count() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_bimodal_image()).unwrap();

        run(default_args(input.clone(), output.clone(), "triangle")).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let foreground = count_foreground(&mask);
        assert_eq!(
            foreground, 32,
            "Triangle must label exactly 32 high-intensity voxels as foreground"
        );
    }

    // ── Positive: Distance-transform background voxels are zero ───────────────

    #[test]
    fn test_segment_distance_transform_background_is_zero() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("dt.nii");

        // All-zero image: every voxel is background.
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values = vec![0.0_f32; 27];
        let td = TensorData::new(values, Shape::new([3, 3, 3]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let img = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );
        ritk_io::write_nifti(&input, &img).unwrap();

        run(default_args(
            input.clone(),
            output.clone(),
            "distance-transform",
        ))
        .unwrap();

        let dt = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = dt.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert_eq!(
                v, 0.0,
                "all-background image must have EDT=0 everywhere, got {v}"
            );
        }
    }

    // -- confidence-connected: positive ---------------------------------------

    #[test]
    fn test_segment_confidence_connected_grows_region() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        // make_sphere_image: 5x5x5, center [2,2,2] and 6 neighbors have value 200.0, rest 10.0
        let mut args = default_args(input.clone(), output.clone(), "confidence-connected");
        args.lower = Some(150.0);
        args.upper = Some(250.0);
        args.seed = Some("2,2,2".to_string());
        run(args).unwrap();

        assert!(output.exists(), "output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5], "shape must match input");
        let n_fg = count_foreground(&mask);
        assert!(n_fg > 0, "must find at least one foreground voxel, got 0");
    }

    #[test]
    fn test_segment_confidence_connected_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "confidence-connected");
        args.lower = Some(150.0);
        args.upper = Some(250.0);
        args.seed = Some("2,2,2".to_string());
        run(args).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(
                v == 0.0 || v == 1.0,
                "all voxels must be 0.0 or 1.0, found {v}"
            );
        }
    }

    #[test]
    fn test_segment_confidence_connected_missing_lower_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        let mut args = default_args(input, output, "confidence-connected");
        args.upper = Some(1.5);
        args.seed = Some("2,2,2".to_string());
        let result = run(args);
        assert!(result.is_err(), "--lower missing must produce an error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--lower"),
            "error must mention --lower, got: {msg}"
        );
    }

    #[test]
    fn test_segment_confidence_connected_missing_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        let mut args = default_args(input, output, "confidence-connected");
        args.lower = Some(0.5);
        args.seed = Some("2,2,2".to_string());
        let result = run(args);
        assert!(result.is_err(), "--upper missing must produce an error");
    }

    #[test]
    fn test_segment_confidence_connected_missing_seed_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        let mut args = default_args(input, output, "confidence-connected");
        args.lower = Some(0.5);
        args.upper = Some(1.5);
        let result = run(args);
        assert!(result.is_err(), "--seed missing must produce an error");
    }

    // -- neighborhood-connected: positive -------------------------------------

    #[test]
    fn test_segment_neighborhood_connected_grows_region() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        // make_sphere_image: 5x5x5, center=200.0, rest=10.0.
        // NeighborhoodConnected requires ALL voxels in the 3x3x3 neighbourhood to
        // satisfy bounds. Use full range [0, 250] so both 10.0 and 200.0 qualify,
        // allowing the region to grow from seed [2,2,2].
        let mut args = default_args(input.clone(), output.clone(), "neighborhood-connected");
        args.lower = Some(0.0);
        args.upper = Some(250.0);
        args.seed = Some("2,2,2".to_string());
        run(args).unwrap();

        assert!(output.exists(), "output mask must be created");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5]);
        let n_fg = count_foreground(&mask);
        assert!(n_fg > 0, "must find foreground voxels, got 0");
    }

    #[test]
    fn test_segment_neighborhood_connected_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "neighborhood-connected");
        args.lower = Some(0.0);
        args.upper = Some(250.0);
        args.seed = Some("2,2,2".to_string());
        run(args).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(
                v == 0.0 || v == 1.0,
                "all voxels must be 0.0 or 1.0, found {v}"
            );
        }
    }

    // -- skeletonization: positive --------------------------------------------

    #[test]
    fn test_segment_skeletonization_creates_output() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("skeleton.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        run(default_args(
            input.clone(),
            output.clone(),
            "skeletonization",
        ))
        .unwrap();

        assert!(output.exists(), "skeleton output must be created");
        let skel = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(skel.shape(), [5, 5, 5], "skeleton shape must match input");
    }

    #[test]
    fn test_segment_skeletonization_strictly_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("skeleton.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        run(default_args(
            input.clone(),
            output.clone(),
            "skeletonization",
        ))
        .unwrap();

        let skel = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = skel.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(
                v == 0.0 || v == 1.0,
                "skeleton voxels must be 0.0 or 1.0, found {v}"
            );
        }
    }

    #[test]
    fn test_segment_fill_holes_fills_enclosed_cavity() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("filled.nii");

        // 7x7x7 hollow sphere: shell at 2 <= dist <= 3 from centre (3,3,3) = 1.0 (foreground).
        // Interior (dist < 2) and exterior (dist > 3) = 0.0.
        // After fill-holes, all interior voxels (dist < 2) must become 1.0.
        let device: <Backend as BurnBackend>::Device = Default::default();
        let (nz, ny, nx) = (7usize, 7usize, 7usize);
        let n = nz * ny * nx;
        let mut vals = vec![0.0_f32; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let d2 = ((iz as i32 - 3).pow(2)
                        + (iy as i32 - 3).pow(2)
                        + (ix as i32 - 3).pow(2)) as f32;
                    if d2 >= 4.0 && d2 <= 9.0 {
                        vals[iz * ny * nx + iy * nx + ix] = 1.0;
                    }
                }
            }
        }
        let td = TensorData::new(vals, Shape::new([nz, ny, nx]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        let hollow_sphere = Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        );

        ritk_io::write_nifti(&input, &hollow_sphere).unwrap();
        run(default_args(input.clone(), output.clone(), "fill-holes")).unwrap();

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = result.data().clone().into_data();
        let out_vals = td.as_slice::<f32>().unwrap();

        // All interior voxels (d2 < 4) must now be foreground.
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let d2 = ((iz as i32 - 3).pow(2)
                        + (iy as i32 - 3).pow(2)
                        + (ix as i32 - 3).pow(2)) as f32;
                    if d2 < 4.0 {
                        assert_eq!(
                            out_vals[iz * ny * nx + iy * nx + ix],
                            1.0,
                            "interior voxel ({},{},{}) at d2={} must be filled",
                            iz,
                            iy,
                            ix,
                            d2
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_segment_morphological_gradient_extracts_boundary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("input.nii");
        let output = dir.path().join("gradient.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        run(default_args(
            input.clone(),
            output.clone(),
            "morphological-gradient",
        ))
        .unwrap();

        let result = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = result.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();

        assert_eq!(vals.len(), 125);
        assert!(
            vals.iter().any(|&v| v == 1.0),
            "morphological gradient must contain boundary voxels"
        );
        assert!(
            vals.iter().all(|&v| v == 0.0 || v == 1.0),
            "morphological gradient must be binary"
        );
    }
    // -- Shape-detection: phi signed-distance helper -------------------------

    fn make_phi_sphere(dims: [usize; 3], center: [f64; 3], radius: f64) -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let mut data = vec![0.0_f32; n];
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let dist = ((iz as f64 - center[0]).powi(2)
                        + (iy as f64 - center[1]).powi(2)
                        + (ix as f64 - center[2]).powi(2))
                    .sqrt();
                    data[iz * ny * nx + iy * nx + ix] = (dist - radius) as f32;
                }
            }
        }
        let td = TensorData::new(data, Shape::new(dims));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    // -- shape-detection tests ----------------------------------------------

    #[test]
    fn test_segment_shape_detection_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        let image = make_sphere_image();
        let phi = make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0);
        ritk_io::write_nifti(&input, &image).unwrap();
        ritk_io::write_nifti(&phi_path, &phi).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "shape-detection");
        args.initial_phi = Some(phi_path);
        args.level_set_max_iterations = 50;
        run(args).unwrap();
        assert!(output.exists(), "output file must exist");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5], "shape must be preserved");
    }

    #[test]
    fn test_segment_shape_detection_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        let image = make_sphere_image();
        let phi = make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0);
        ritk_io::write_nifti(&input, &image).unwrap();
        ritk_io::write_nifti(&phi_path, &phi).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "shape-detection");
        args.initial_phi = Some(phi_path);
        args.level_set_max_iterations = 50;
        run(args).unwrap();
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(v == 0.0 || v == 1.0, "output must be binary, got {v}");
        }
    }

    #[test]
    fn test_segment_shape_detection_missing_phi_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        let args = default_args(input.clone(), output.clone(), "shape-detection");
        let result = run(args);
        assert!(result.is_err(), "--initial-phi missing must produce error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--initial-phi"),
            "error must mention --initial-phi, got: {msg}"
        );
    }

    // -- threshold-level-set tests -----------------------------------------

    #[test]
    fn test_segment_threshold_level_set_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        let image = make_sphere_image();
        let phi = make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0);
        ritk_io::write_nifti(&input, &image).unwrap();
        ritk_io::write_nifti(&phi_path, &phi).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.lower_threshold = Some(5.0);
        args.upper_threshold = Some(250.0);
        args.level_set_max_iterations = 50;
        run(args).unwrap();
        assert!(output.exists(), "output file must exist");
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5], "shape must be preserved");
    }

    #[test]
    fn test_segment_threshold_level_set_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        let image = make_sphere_image();
        let phi = make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0);
        ritk_io::write_nifti(&input, &image).unwrap();
        ritk_io::write_nifti(&phi_path, &phi).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.lower_threshold = Some(5.0);
        args.upper_threshold = Some(250.0);
        args.level_set_max_iterations = 50;
        run(args).unwrap();
        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let td = mask.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for &v in vals {
            assert!(v == 0.0 || v == 1.0, "output must be binary, got {v}");
        }
    }

    #[test]
    fn test_segment_threshold_level_set_missing_phi_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.lower_threshold = Some(5.0);
        args.upper_threshold = Some(250.0);
        let result = run(args);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("--initial-phi"));
    }

    #[test]
    fn test_segment_threshold_level_set_missing_lower_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0)).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.upper_threshold = Some(250.0);
        let result = run(args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("--lower-threshold"));
    }

    #[test]
    fn test_segment_threshold_level_set_missing_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0)).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.lower_threshold = Some(5.0);
        let result = run(args);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("--upper-threshold"));
    }

    #[test]
    fn test_segment_threshold_level_set_lower_gt_upper_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");
        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.0)).unwrap();
        let mut args = default_args(input.clone(), output.clone(), "threshold-level-set");
        args.initial_phi = Some(phi_path);
        args.lower_threshold = Some(250.0);
        args.upper_threshold = Some(5.0);
        let result = run(args);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("must be") && msg.contains("<="),
            "error must explain bound constraint, got: {msg}"
        );
    }

    // -- Connected-components tests -----------------------------------------

    fn make_binary_image_with_components(
        dims: [usize; 3],
        components: &[(usize, usize, usize, usize, usize, usize)],
    ) -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let [nz, ny, nx] = dims;
        let n = nz * ny * nx;
        let mut vals = vec![0.0_f32; n];
        for &(z0, y0, x0, z1, y1, x1) in components {
            for iz in z0..z1 {
                for iy in y0..y1 {
                    for ix in x0..x1 {
                        vals[iz * ny * nx + iy * nx + ix] = 1.0;
                    }
                }
            }
        }
        let td = TensorData::new(vals, Shape::new(dims));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    #[test]
    fn test_segment_connected_components_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("mask.nii");
        let output = dir.path().join("labels.nii");

        // Create two separate foreground regions
        let image =
            make_binary_image_with_components([8, 8, 8], &[(2, 2, 2, 4, 4, 4), (5, 5, 5, 7, 7, 7)]);

        ritk_io::write_nifti(&input, &image).unwrap();

        let args = default_args(input.clone(), output.clone(), "connected-components");
        let result = run(args);
        assert!(result.is_ok(), "connected-components should succeed");

        let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(labels.shape(), [8, 8, 8], "output shape must match input");
    }

    #[test]
    fn test_segment_connected_components_output_labels_are_valid() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("mask.nii");
        let output = dir.path().join("labels.nii");

        // Create two separated components
        let image =
            make_binary_image_with_components([6, 6, 6], &[(1, 1, 1, 3, 3, 3), (4, 4, 4, 6, 6, 6)]);

        ritk_io::write_nifti(&input, &image).unwrap();

        let args = default_args(input.clone(), output.clone(), "connected-components");
        run(args).unwrap();

        let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let vals = labels.data().clone().into_data();
        let slice = vals.as_slice::<f32>().unwrap();

        // Labels must be 0, 1, or 2 (background + two components)
        for &v in slice {
            assert!(
                v == 0.0 || v == 1.0 || v == 2.0,
                "label must be 0, 1, or 2, got {}",
                v
            );
        }

        // Both component labels must be present
        let has_label_1 = slice.iter().any(|&v| v == 1.0);
        let has_label_2 = slice.iter().any(|&v| v == 2.0);
        assert!(has_label_1, "label 1 must be present");
        assert!(has_label_2, "label 2 must be present");
    }

    #[test]
    fn test_segment_connected_components_connectivity_26() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("mask.nii");
        let output = dir.path().join("labels.nii");

        // Create a diagonal pattern: 26-connectivity should merge these
        let image =
            make_binary_image_with_components([4, 4, 4], &[(1, 1, 1, 2, 2, 2), (2, 2, 2, 3, 3, 3)]);

        ritk_io::write_nifti(&input, &image).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "connected-components");
        args.connectivity = 26;
        run(args).unwrap();

        let labels = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let vals = labels.data().clone().into_data();
        let slice = vals.as_slice::<f32>().unwrap();

        // With 26-connectivity, diagonal neighbors are connected
        let has_label_1 = slice.iter().any(|&v| v == 1.0);
        assert!(
            has_label_1,
            "component must be labeled with 26-connectivity"
        );
    }

    // -- Chan-Vese tests ----------------------------------------------------

    #[test]
    fn test_segment_chan_vese_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "chan-vese");
        args.level_set_max_iterations = 10;

        let result = run(args);
        assert!(result.is_ok(), "chan-vese should succeed");

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5], "output shape must match input");
    }

    #[test]
    fn test_segment_chan_vese_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "chan-vese");
        args.level_set_max_iterations = 50;

        run(args).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let vals = mask.data().clone().into_data();
        let slice = vals.as_slice::<f32>().unwrap();

        for (i, &v) in slice.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "output voxel {} must be binary, got {}",
                i,
                v
            );
        }
    }

    // Note: chan-vese does not require --initial-phi, so no missing-phi test needed

    // -- Geodesic active contour tests --------------------------------------

    #[test]
    fn test_segment_geodesic_active_contour_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.5)).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "geodesic-active-contour");
        args.initial_phi = Some(phi_path);
        args.level_set_max_iterations = 10;

        let result = run(args);
        assert!(result.is_ok(), "geodesic-active-contour should succeed");

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5], "output shape must match input");
    }

    #[test]
    fn test_segment_geodesic_active_contour_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.5)).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "geodesic-active-contour");
        args.initial_phi = Some(phi_path);
        args.level_set_max_iterations = 50;

        run(args).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let vals = mask.data().clone().into_data();
        let slice = vals.as_slice::<f32>().unwrap();

        for (i, &v) in slice.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "output voxel {} must be binary, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_segment_geodesic_active_contour_missing_phi_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let args = default_args(input.clone(), output.clone(), "geodesic-active-contour");
        let result = run(args);
        assert!(result.is_err(), "--initial-phi missing must produce error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--initial-phi"),
            "error must mention --initial-phi, got: {msg}"
        );
    }

    // -- Laplacian level set tests -----------------------------------------

    #[test]
    fn test_segment_laplacian_level_set_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.5)).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "laplacian-level-set");
        args.initial_phi = Some(phi_path);
        args.level_set_max_iterations = 10;

        let result = run(args);
        assert!(result.is_ok(), "laplacian-level-set should succeed");

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        assert_eq!(mask.shape(), [5, 5, 5], "output shape must match input");
    }

    #[test]
    fn test_segment_laplacian_level_set_output_is_binary() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let phi_path = dir.path().join("phi.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();
        ritk_io::write_nifti(&phi_path, &make_phi_sphere([5, 5, 5], [2.0, 2.0, 2.0], 1.5)).unwrap();

        let mut args = default_args(input.clone(), output.clone(), "laplacian-level-set");
        args.initial_phi = Some(phi_path);
        args.level_set_max_iterations = 50;

        run(args).unwrap();

        let mask = ritk_io::read_nifti::<Backend, _>(&output, &Default::default()).unwrap();
        let vals = mask.data().clone().into_data();
        let slice = vals.as_slice::<f32>().unwrap();

        for (i, &v) in slice.iter().enumerate() {
            assert!(
                v == 0.0 || v == 1.0,
                "output voxel {} must be binary, got {}",
                i,
                v
            );
        }
    }

    #[test]
    fn test_segment_laplacian_level_set_missing_phi_returns_error() {
        let dir = tempdir().unwrap();
        let input = dir.path().join("image.nii");
        let output = dir.path().join("mask.nii");

        ritk_io::write_nifti(&input, &make_sphere_image()).unwrap();

        let args = default_args(input.clone(), output.clone(), "laplacian-level-set");
        let result = run(args);
        assert!(result.is_err(), "--initial-phi missing must produce error");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("--initial-phi"),
            "error must mention --initial-phi, got: {msg}"
        );
    }
}
