use clap::Args;
use std::path::PathBuf;

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

    // ── K-Means extended parameters ────────────────────────────────────────
    /// Maximum Lloyd iterations for `kmeans`. Default: 100.
    #[arg(long, value_name = "INT")]
    pub kmeans_max_iterations: Option<usize>,

    /// Centroid-displacement convergence tolerance for `kmeans`. Default: 1e-6.
    #[arg(long, value_name = "FLOAT")]
    pub kmeans_tolerance: Option<f64>,

    /// Deterministic seed for k-means++ initialization. Default: 42.
    #[arg(long, value_name = "INT")]
    pub kmeans_seed: Option<u64>,

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
    // -- Marker-watershed --------------------------------------------------
    /// Path to marker label image (for marker-watershed method).
    #[arg(long, value_name = "PATH")]
    pub markers: Option<String>,
}

impl Default for SegmentArgs {
    fn default() -> Self {
        Self {
            input: PathBuf::default(),
            output: PathBuf::default(),
            method: String::default(),
            classes: 3,
            kmeans_max_iterations: None,
            kmeans_tolerance: None,
            kmeans_seed: None,
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
            markers: None,
        }
    }
}
