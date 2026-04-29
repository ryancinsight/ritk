//! `ritk register` — image registration command.
//!
//! Registers a moving image to a fixed reference image.
//!
//! # Supported methods
//!
//! | Method            | DOF   | Algorithm                                        |
//! |-------------------|-------|--------------------------------------------------|
//! | `rigid-mi`        | 6     | Rigid hill-climbing (MI)                         |
//! | `affine-mi`       | 9     | Affine hill-climbing (MI)                        |
//! | `demons`          | dense | Thirion Demons deformable                        |
//! | `multires-demons` | dense | Multi-resolution Thirion/Diffeomorphic Demons    |
//! | `ic-demons`       | dense | Inverse-consistent diffeomorphic Demons          |
//! | `syn`             | dense | Greedy SyN diffeomorphic                         |
//! | `bspline-ffd`     | dense | B-Spline FFD deformable (Rueckert 1999)          |
//! | `multires-syn`    | dense | Multi-resolution SyN (coarse-to-fine pyramid)    |
//! | `bspline-syn`     | dense | BSpline SyN (B-spline velocity fields)           |
//! | `lddmm`           | dense | LDDMM geodesic shooting (EPDiff, Gaussian RKHS)  |
//!
//! # Pipeline
//!
//! 1. Read fixed and moving images (format inferred from extension).
//! 2. Optionally apply isotropic Gaussian smoothing (`--sigma-fixed`).
//! 3. Convert both images to `ndarray::Array3<f64>` (for MI methods) or
//!    flat `Vec<f32>` (for deformable methods).
//! 4. Run the selected registration method; initial transform is identity.
//! 5. Apply the estimated 4×4 homogeneous transform to the moving image
//!    (MI methods) or reconstruct from warped output (deformable methods).
//! 6. Write the warped image to `--output`.
//! 7. Optionally serialise the 4×4 transform matrix to JSON at
//!    `--output-transform` (MI methods only).
//! 8. Print a registration summary (iterations, metric value, convergence status).

use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use ritk_core::filter::GaussianFilter;
use ritk_core::image::Image;
use ritk_registration::classical::engine::{ClassicalConfig, MutualInformationMetric};
use ritk_registration::classical::spatial;
use ritk_registration::ImageRegistration;

use super::Backend;

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `register` subcommand.
#[derive(Args, Debug)]
pub struct RegisterArgs {
    /// Fixed (reference) image path.  Format is inferred from the file extension.
    #[arg(long)]
    pub fixed: PathBuf,

    /// Moving image path.  Format is inferred from the file extension.
    #[arg(long)]
    pub moving: PathBuf,

    /// Output path for the warped moving image.
    #[arg(short, long)]
    pub output: PathBuf,

    /// Registration method.
    ///
    /// Accepted values: `rigid-mi`, `affine-mi`, `demons`, `multires-demons`,
    /// `ic-demons`, `syn`, `bspline-ffd`, `multires-syn`, `bspline-syn`, `lddmm`.
    #[arg(long, value_name = "METHOD")]
    pub method: String,

    /// Output path for the estimated transform (JSON array of 16 floats
    /// representing a row-major 4×4 homogeneous matrix).  Optional.
    /// Only produced by `rigid-mi` and `affine-mi`.
    #[arg(long, value_name = "PATH")]
    pub output_transform: Option<PathBuf>,

    /// Maximum number of hill-climbing iterations.
    #[arg(long, default_value = "100", value_name = "INT")]
    pub iterations: usize,

    /// Standard deviation (mm) of the isotropic Gaussian filter applied to
    /// both images before registration.  Set to 0.0 to disable smoothing.
    #[arg(long, default_value = "1.5", value_name = "FLOAT")]
    pub sigma_fixed: f64,

    /// Number of pyramid levels for multi-resolution Demons (default 3).
    #[arg(long, default_value = "3", value_name = "INT")]
    pub levels: usize,

    /// Use Diffeomorphic Demons at each pyramid level (default: Thirion Demons).
    #[arg(long, default_value = "false", value_name = "BOOL")]
    pub use_diffeomorphic: bool,

    /// Regularization weight for BSpline FFD and BSpline SyN bending energy (default 0.001).
    #[arg(long, default_value = "0.001", value_name = "FLOAT")]
    pub regularization_weight: f64,

    /// Convergence threshold for BSpline FFD and BSpline SyN: optimization stops when the
    /// relative NCC metric change between consecutive iterations falls below this value (default 1e-5).
    #[arg(long, default_value = "0.00001", value_name = "FLOAT")]
    pub convergence_threshold: f64,

    /// Control point spacing in voxels for BSpline FFD and BSpline SyN (default 8).
    #[arg(long, default_value = "8", value_name = "INT")]
    pub control_spacing: usize,

    /// NCC window radius in voxels for SyN-family methods (default 2).
    #[arg(long, default_value = "2", value_name = "INT")]
    pub cc_radius: usize,

    /// Enforce inverse consistency in Multi-Resolution SyN (default false).
    #[arg(long, default_value = "false", value_name = "BOOL")]
    pub inverse_consistency: bool,

    /// Number of LDDMM time-discretization steps (default 10).
    #[arg(long, default_value = "10", value_name = "INT")]
    pub num_time_steps: usize,

    /// RKHS Gaussian kernel sigma for LDDMM regularization (default 3.0).
    #[arg(long, default_value = "3.0", value_name = "FLOAT")]
    pub kernel_sigma: f64,

    /// Learning rate for BSpline FFD and LDDMM gradient descent (default 0.01).
    #[arg(long, default_value = "0.01", value_name = "FLOAT")]
    pub learning_rate: f64,

    /// Backward-force weight for inverse-consistent Demons in [0, 1] (default 0.5).
    #[arg(long, default_value = "0.5", value_name = "FLOAT")]
    pub inverse_consistency_weight: f64,

    /// Scaling-and-squaring steps for diffeomorphic Demons variants (default 6).
    #[arg(long, default_value = "6", value_name = "INT")]
    pub n_squarings: usize,
}

// ── Image ↔ ndarray conversion ────────────────────────────────────────────────

/// Convert a 3-D `Image<Backend, 3>` to an `ndarray::Array3<f64>`.
///
/// Data is extracted in the image's native [Z, Y, X] layout (C-order) and
/// cast element-wise from `f32` to `f64`.
///
/// # Panics
/// Panics if the tensor data cannot be extracted as `f32`.
fn image_to_array3(image: &Image<Backend, 3>) -> ndarray::Array3<f64> {
    let shape = image.shape();
    let td = image.data().clone().into_data();
    let slice = td
        .as_slice::<f32>()
        .expect("image tensor must contain f32 data");
    let f64_vec: Vec<f64> = slice.iter().map(|&v| v as f64).collect();

    ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), f64_vec)
        .expect("shape derived from image must be consistent with data length")
}

/// Convert a warped `ndarray::Array3<f64>` back to `Image<Backend, 3>`.
///
/// The spatial metadata (origin, spacing, direction) is copied from
/// `reference` so the output image lives in the fixed image's frame.
fn array3_to_image(arr: ndarray::Array3<f64>, reference: &Image<Backend, 3>) -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let (nz, ny, nx) = arr.dim();

    let f32_vec: Vec<f32> = arr.iter().map(|&v| v as f32).collect();
    let td = TensorData::new(f32_vec, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);

    Image::new(
        tensor,
        reference.origin().clone(),
        reference.spacing().clone(),
        reference.direction().clone(),
    )
}

// ── Image ↔ flat Vec<f32> conversion (for demons / SyN) ──────────────────────

/// Extract a flat `Vec<f32>` in Z-major order and the `[nz, ny, nx]` shape
/// from a 3-D image.
///
/// # Panics
/// Panics if the tensor data cannot be extracted as `f32`.
fn image_to_flat_vec(image: &Image<Backend, 3>) -> (Vec<f32>, [usize; 3]) {
    let shape = image.shape();
    let td = image.data().clone().into_data();
    let data: Vec<f32> = td
        .as_slice::<f32>()
        .expect("image tensor must contain f32 data")
        .to_vec();
    (data, [shape[0], shape[1], shape[2]])
}

/// Reconstruct an `Image<Backend, 3>` from flat `Vec<f32>` data and a
/// `[nz, ny, nx]` shape, copying spatial metadata from `reference`.
fn flat_vec_to_image(
    data: Vec<f32>,
    shape: [usize; 3],
    reference: &Image<Backend, 3>,
) -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let td = TensorData::new(data, Shape::new(shape));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);

    Image::new(
        tensor,
        reference.origin().clone(),
        reference.spacing().clone(),
        reference.direction().clone(),
    )
}

// ── Command handler ───────────────────────────────────────────────────────────

/// Execute the `register` subcommand.
///
/// All errors are propagated as `anyhow::Error`; no panics occur in the
/// production path.
///
/// # Errors
/// Returns an error when:
/// - The fixed or moving image cannot be read.
/// - An unknown method name is supplied.
/// - The registration engine reports an error.
/// - The output image or transform cannot be written.
pub fn run(args: RegisterArgs) -> Result<()> {
    info!(
        "register: starting fixed={} moving={} output={} method={} iterations={} sigma_fixed={}",
        args.fixed.display(),
        args.moving.display(),
        args.output.display(),
        args.method,
        args.iterations,
        args.sigma_fixed
    );

    match args.method.as_str() {
        "rigid-mi" | "affine-mi" => run_mi_registration(&args),
        "demons" => run_demons(&args),
        "multires-demons" => run_multires_demons(&args),
        "ic-demons" => run_inverse_consistent_demons(&args),
        "syn" => run_syn(&args),
        "bspline-ffd" => run_bspline_ffd(&args),
        "multires-syn" => run_multires_syn(&args),
        "bspline-syn" => run_bspline_syn(&args),
        "lddmm" => run_lddmm(&args),
        other => Err(anyhow!(
            "Unknown registration method '{other}'. \
             Supported methods: rigid-mi, affine-mi, demons, multires-demons, ic-demons, syn, \n             bspline-ffd, multires-syn, bspline-syn, lddmm."
        )),
    }
}

// ── MI-based registration (rigid / affine) ────────────────────────────────────

/// Run rigid-mi or affine-mi registration via the classical MI engine.
fn run_mi_registration(args: &RegisterArgs) -> Result<()> {
    // ── 1. Read images ─────────────────────────────────────────────────────
    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    // ── 2. Optional pre-registration Gaussian smoothing ───────────────────
    // GaussianFilter skips any dimension whose sigma ≤ 1e-6, so sigma=0.0
    // is a safe no-op.
    let (fixed_img, moving_img) = if args.sigma_fixed > 1e-12 {
        let filter: GaussianFilter<Backend> = GaussianFilter::new(vec![args.sigma_fixed; 3]);
        (filter.apply(&fixed_img), filter.apply(&moving_img))
    } else {
        (fixed_img, moving_img)
    };

    // ── 3. Convert images to ndarray::Array3<f64> ─────────────────────────
    let fixed_arr = image_to_array3(&fixed_img);
    let moving_arr = image_to_array3(&moving_img);

    // ── 4. Build registration engine with user-supplied iteration budget ───
    let config = ClassicalConfig {
        max_iterations: args.iterations,
        tolerance: 1e-6,
        step_multiplier: 1.0,
    };
    let metric = MutualInformationMetric::default();
    let reg = ImageRegistration::with_config(config, metric);

    // Identity 4×4 homogeneous matrix as the initial transform.
    let identity: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ];

    // ── 5. Run registration ────────────────────────────────────────────────
    let result = match args.method.as_str() {
        "rigid-mi" => reg
            .rigid_registration_mutual_info(&moving_arr, &fixed_arr, &identity)
            .with_context(|| "rigid MI registration failed")?,
        "affine-mi" => reg
            .affine_registration_mutual_info(&moving_arr, &fixed_arr, &identity)
            .with_context(|| "affine MI registration failed")?,
        _ => unreachable!("run_mi_registration called with non-MI method"),
    };

    // ── 6. Warp moving image with estimated transform ──────────────────────
    // Re-read the original (un-smoothed) moving image for warping so the
    // output preserves the full-resolution signal.
    let moving_orig = super::read_image(&args.moving)?;
    let moving_orig_arr = image_to_array3(&moving_orig);
    let warped_arr = spatial::apply_transform(&moving_orig_arr, &result.transform);

    // ── 7. Convert warped array back to Image and write output ─────────────
    // Spatial metadata comes from the fixed image (the output lives in the
    // fixed image's coordinate frame).
    let warped_img = array3_to_image(warped_arr, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    // ── 8. Optionally write transform JSON ─────────────────────────────────
    if let Some(ref tx_path) = args.output_transform {
        let json = serde_json::to_string_pretty(&result.transform)
            .context("Failed to serialise transform to JSON")?;
        std::fs::write(tx_path, &json)
            .with_context(|| format!("Failed to write transform to {}", tx_path.display()))?;
        info!("register: transform written path={}", tx_path.display());
    }

    // ── 9. Print summary ───────────────────────────────────────────────────
    let q = &result.quality;
    println!(
        "Registered {} \u{2192} {} (method={}, iterations={}, converged={}, MI={:.6}, cost={:.6})",
        args.moving.display(),
        args.output.display(),
        args.method,
        q.iterations,
        q.converged,
        q.mutual_information,
        q.final_cost,
    );

    info!(
        "register: MI registration complete method={} iterations={} converged={} mi={} final_cost={}",
        args.method, q.iterations, q.converged, q.mutual_information, q.final_cost
    );

    Ok(())
}

// ── Thirion Demons registration ───────────────────────────────────────────────

/// Run Thirion Demons deformable registration.
///
/// Converts both images to flat `Vec<f32>`, runs the Thirion Demons
/// algorithm, and reconstructs the output image from `result.warped`.
fn run_demons(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::demons::{DemonsConfig, ThirionDemonsRegistration};

    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = DemonsConfig {
        max_iterations: args.iterations,
        sigma_diffusion: 1.5,
        sigma_fluid: 0.0,
        max_step_length: 2.0,
    };

    let reg = ThirionDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Thirion Demons registration failed")?;

    let warped_img = flat_vec_to_image(result.warped, fixed_shape, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=demons, iterations={}, final_mse={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        result.final_mse,
    );

    info!(
        "register: demons complete method={} iterations={} final_mse={}",
        "demons", result.num_iterations, result.final_mse
    );

    Ok(())
}

// ── Multi-resolution Demons registration ─────────────────────────────────────────

/// Run multi-resolution Demons deformable registration.
///
/// Converts both images to flat `Vec<f32>`, runs the coarse-to-fine Demons pyramid,
/// and reconstructs the output image from `result.warped`.
fn run_multires_demons(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::demons::{
        DemonsConfig, MultiResDemonsConfig, MultiResDemonsRegistration,
    };

    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = MultiResDemonsConfig {
        base_config: DemonsConfig {
            max_iterations: args.iterations,
            sigma_diffusion: 1.5,
            sigma_fluid: 0.0,
            max_step_length: 2.0,
        },
        levels: args.levels,
        use_diffeomorphic: args.use_diffeomorphic,
        n_squarings: 6,
    };

    let reg = MultiResDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Multi-resolution Demons registration failed")?;

    let warped_img = flat_vec_to_image(result.warped, fixed_shape, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} → {} (method=multires-demons, iterations={}, levels={}, final_mse={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        args.levels,
        result.final_mse,
    );

    info!(
        "register: multires-demons complete method={} iterations={} levels={} final_mse={}",
        "multires-demons", result.num_iterations, args.levels, result.final_mse
    );

    Ok(())
}

// ── SyN diffeomorphic registration ────────────────────────────────────────────

/// Run greedy SyN diffeomorphic registration.
///
/// Converts both images to flat `Vec<f32>`, runs SyN with local CC metric,
/// and reconstructs the output image from `result.warped_moving` (the moving
/// image warped towards the fixed image's midpoint).
/// Run inverse-consistent diffeomorphic Demons registration.
///
/// Maintains forward and exact inverse transforms through SVF negation and
/// writes the warped moving image in the fixed image frame.
fn run_inverse_consistent_demons(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::demons::{
        DemonsConfig, InverseConsistentDemonsConfig,
        InverseConsistentDiffeomorphicDemonsRegistration,
    };

    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = InverseConsistentDemonsConfig {
        demons: DemonsConfig {
            max_iterations: args.iterations,
            sigma_diffusion: 1.5,
            sigma_fluid: 0.0,
            max_step_length: 2.0,
        },
        inverse_consistency_weight: args.inverse_consistency_weight,
        n_squarings: args.n_squarings,
    };

    let reg = InverseConsistentDiffeomorphicDemonsRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Inverse-consistent Demons registration failed")?;

    let warped_img = flat_vec_to_image(result.warped, fixed_shape, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} → {} (method=ic-demons, iterations={}, final_mse={:.6}, ic_residual={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        result.final_mse,
        result.inverse_consistency_residual,
    );

    info!(
        "register: inverse-consistent demons complete method={} iterations={} final_mse={} inverse_consistency_residual={}",
        "ic-demons", result.num_iterations, result.final_mse, result.inverse_consistency_residual
    );

    Ok(())
}

fn run_syn(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::diffeomorphic::{SyNConfig, SyNRegistration};

    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = SyNConfig {
        max_iterations: args.iterations,
        sigma_smooth: 3.0,
        cc_window_radius: 2,
        gradient_step: 0.25,
        ..Default::default()
    };

    let reg = SyNRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "SyN registration failed")?;

    // SyN produces warped_moving: moving image warped to the midpoint.
    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=syn, iterations={}, final_cc={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        result.final_cc,
    );

    info!(
        "register: syn complete method={} iterations={} final_cc={}",
        "syn", result.num_iterations, result.final_cc
    );

    Ok(())
}

// ── BSpline FFD registration ───────────────────────────────────────────────

/// Run B-Spline Free-Form Deformation registration.
///
/// Rueckert et al. (1999): Multi-resolution control-lattice FFD with NCC metric
/// and bending-energy regularization.  Control-point spacing halves at each
/// successive level.
fn run_bspline_ffd(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::bspline_ffd::{BSplineFFDConfig, BSplineFFDRegistration};

    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = BSplineFFDConfig {
        initial_control_spacing: [
            args.control_spacing,
            args.control_spacing,
            args.control_spacing,
        ],
        num_levels: args.levels,
        max_iterations_per_level: args.iterations,
        learning_rate: args.learning_rate,
        regularization_weight: args.regularization_weight,
        convergence_threshold: args.convergence_threshold,
    };

    let result = BSplineFFDRegistration::register(
        &fixed_vals,
        &moving_vals,
        fixed_shape,
        [1.0, 1.0, 1.0],
        &config,
    )
    .with_context(|| "BSpline FFD registration failed")?;

    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=bspline-ffd, levels={}, iterations={}, final_metric={:.6})",
        args.moving.display(),
        args.output.display(),
        args.levels,
        result.num_iterations,
        result.final_metric,
    );

    info!(
        "register: bspline-ffd complete method={} levels={} iterations={} final_metric={}",
        "bspline-ffd", args.levels, result.num_iterations, result.final_metric
    );

    Ok(())
}

// ── Multi-resolution SyN registration ────────────────────────────────────────────

/// Run Multi-Resolution SyN diffeomorphic registration.
///
/// Avants & Gee coarse-to-fine pyramid SyN: Gaussian downsampling with
/// level-doubling velocity fields and optional inverse consistency enforcement.
fn run_multires_syn(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::diffeomorphic::multires_syn::{
        MultiResSyNConfig, MultiResSyNRegistration,
    };

    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = MultiResSyNConfig {
        num_levels: args.levels,
        iterations_per_level: vec![args.iterations; args.levels],
        sigma_smooth: args.sigma_fixed,
        convergence_threshold: args.convergence_threshold,
        convergence_window: 10,
        n_squarings: 6,
        cc_window_radius: args.cc_radius,
        enforce_inverse_consistency: args.inverse_consistency,
        gradient_step: 0.25,
    };

    let reg = MultiResSyNRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "Multi-resolution SyN registration failed")?;

    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=multires-syn, levels={}, iterations={}, final_cc={:.6})",
        args.moving.display(),
        args.output.display(),
        args.levels,
        result.num_iterations,
        result.final_cc,
    );

    info!(
        "register: multires-syn complete method={} levels={} iterations={} final_cc={}",
        "multires-syn", args.levels, result.num_iterations, result.final_cc
    );

    Ok(())
}

// ── BSpline SyN registration ──────────────────────────────────────────────

/// Run BSpline SyN diffeomorphic registration.
///
/// Symmetric diffeomorphic registration with B-spline-parameterized velocity
/// fields.  Provides intrinsic smoothness from B-spline basis and bending-energy
/// regularization.
fn run_bspline_syn(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::diffeomorphic::bspline_syn::{BSplineSyNConfig, BSplineSyNRegistration};

    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = BSplineSyNConfig {
        max_iterations: args.iterations,
        control_spacing: [
            args.control_spacing,
            args.control_spacing,
            args.control_spacing,
        ],
        sigma_smooth: args.sigma_fixed,
        convergence_threshold: args.convergence_threshold,
        convergence_window: 10,
        n_squarings: 6,
        cc_window_radius: args.cc_radius,
        regularization_weight: args.regularization_weight,
        gradient_step: 0.25,
    };

    let reg = BSplineSyNRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "BSpline SyN registration failed")?;

    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=bspline-syn, iterations={}, final_cc={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        result.final_cc,
    );

    info!(
        "register: bspline-syn complete method={} iterations={} final_cc={}",
        "bspline-syn", result.num_iterations, result.final_cc
    );

    Ok(())
}

// ── LDDMM registration ────────────────────────────────────────────────────

/// Run LDDMM (Large Deformation Diffeomorphic Metric Mapping) registration.
///
/// Geodesic shooting via EPDiff with a Gaussian RKHS kernel (Miller 2006).
/// The initial velocity field parameterizes the geodesic; the deformation
/// phi_1 at t=1 warps the moving image.
fn run_lddmm(args: &RegisterArgs) -> Result<()> {
    use ritk_registration::lddmm::{LddmmConfig, LddmmRegistration};

    let fixed_img = super::read_image(&args.fixed)?;
    let moving_img = super::read_image(&args.moving)?;

    let (fixed_vals, fixed_shape) = image_to_flat_vec(&fixed_img);
    let (moving_vals, _) = image_to_flat_vec(&moving_img);

    let config = LddmmConfig {
        max_iterations: args.iterations,
        num_time_steps: args.num_time_steps,
        kernel_sigma: args.kernel_sigma,
        learning_rate: args.learning_rate,
        ..Default::default()
    };

    let reg = LddmmRegistration::new(config);
    let result = reg
        .register(&fixed_vals, &moving_vals, fixed_shape, [1.0, 1.0, 1.0])
        .with_context(|| "LDDMM registration failed")?;

    let warped_img = flat_vec_to_image(result.warped_moving, fixed_shape, &fixed_img);
    super::write_image_inferred(&args.output, &warped_img)?;

    println!(
        "Registered {} \u{2192} {} (method=lddmm, iterations={}, time_steps={}, final_metric={:.6})",
        args.moving.display(),
        args.output.display(),
        result.num_iterations,
        args.num_time_steps,
        result.final_metric,
    );

    info!(
        "register: lddmm complete method={} iterations={} time_steps={} final_metric={}",
        "lddmm", result.num_iterations, args.num_time_steps, result.final_metric
    );

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use tempfile::tempdir;

    /// Build a deterministic 4×4×4 image from a ramp of intensities.
    ///
    /// Using a ramp (not constant) so the MI metric has a non-degenerate
    /// joint histogram to work with.
    fn make_ramp_image() -> Image<Backend, 3> {
        let device: <Backend as BurnBackend>::Device = Default::default();
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 4.0).collect();
        let td = TensorData::new(values, Shape::new([4, 4, 4]));
        let tensor = Tensor::<Backend, 3>::from_data(td, &device);
        Image::new(
            tensor,
            Point::new([0.0; 3]),
            Spacing::new([1.0; 3]),
            Direction::identity(),
        )
    }

    // ── Positive: rigid-mi creates output file ────────────────────────────

    /// Running `rigid-mi` on identical fixed/moving images must produce a
    /// warped output file whose shape matches the input.
    #[test]
    fn test_register_rigid_mi_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "rigid-mi".to_string(),
            output_transform: None,
            // Use very few iterations so the test completes quickly.
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        assert!(output_path.exists(), "warped output file must be created");
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(
            warped.shape(),
            [4, 4, 4],
            "warped image shape must match fixed image shape"
        );
    }

    // ── Positive: affine-mi creates output file ───────────────────────────

    /// Running `affine-mi` must produce a warped output file.
    #[test]
    fn test_register_affine_mi_creates_output() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "affine-mi".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        assert!(
            output_path.exists(),
            "affine warped output file must be created"
        );
    }

    // ── Positive: --output-transform writes valid JSON ────────────────────

    /// When `--output-transform` is supplied the file must exist and parse
    /// as a JSON array of exactly 16 finite float values.
    #[test]
    fn test_register_writes_transform_json_with_16_elements() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");
        let tx_path = dir.path().join("transform.json");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path,
            method: "rigid-mi".to_string(),
            output_transform: Some(tx_path.clone()),
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        assert!(tx_path.exists(), "transform JSON file must be created");

        let json_str = std::fs::read_to_string(&tx_path).unwrap();
        let values: Vec<f64> =
            serde_json::from_str(&json_str).expect("transform file must be valid JSON");
        assert_eq!(
            values.len(),
            16,
            "transform must contain exactly 16 elements (row-major 4\u{d7}4 matrix)"
        );
        for (i, &v) in values.iter().enumerate() {
            assert!(
                v.is_finite(),
                "transform element [{i}] must be finite, got {v}"
            );
        }
    }

    // ── Positive: identity on identical images preserves voxel sum ────────

    /// When fixed == moving and the registration converges close to identity,
    /// the voxel sum of the warped image must be close to the input sum.
    /// (Exact equality is not required because nearest-neighbour warp may
    /// drop boundary voxels.)
    #[test]
    fn test_register_identity_moving_preserves_voxel_sum_approximately() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        let original_sum: f32 = image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();

        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "rigid-mi".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let warped_sum: f32 = warped
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();

        // Allow up to 5 % relative deviation to account for boundary voxels
        // that may fall outside the source volume after warping.
        let rel_err = ((original_sum - warped_sum).abs()) / original_sum.abs().max(1.0);
        assert!(
            rel_err < 0.05,
            "warped voxel sum {warped_sum:.1} must be within 5% of original {original_sum:.1}"
        );
    }

    // ── Positive: demons creates output file ──────────────────────────────

    /// Running `demons` on identical fixed/moving images must produce a
    /// warped output file whose shape matches the input.
    #[test]
    fn test_register_demons_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "demons".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        assert!(
            output_path.exists(),
            "demons warped output file must be created"
        );
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(
            warped.shape(),
            [4, 4, 4],
            "demons warped image shape must match fixed image shape"
        );
    }

    // ── Positive: demons identity registration has low MSE ────────────────

    /// When fixed == moving, the Thirion Demons final MSE must be near zero.
    #[test]
    fn test_register_demons_identity_low_mse() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "demons".to_string(),
            output_transform: None,
            iterations: 5,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        // Verify the warped image has finite voxel values.
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let td = warped.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "demons output voxel [{i}] must be finite, got {v}"
            );
        }
    }

    // ── Positive: syn creates output file ─────────────────────────────────

    /// Running `syn` on identical fixed/moving images must produce a warped
    /// output file whose shape matches the input.
    #[test]
    fn test_register_syn_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "syn".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        assert!(
            output_path.exists(),
            "syn warped output file must be created"
        );
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(
            warped.shape(),
            [4, 4, 4],
            "syn warped image shape must match fixed image shape"
        );
    }

    // ── Positive: syn identity registration produces finite voxels ────────

    /// When fixed == moving, the SyN output voxels must all be finite.
    #[test]
    fn test_register_syn_identity_finite_voxels() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "syn".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let td = warped.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "syn output voxel [{i}] must be finite, got {v}"
            );
        }
    }

    // ── Negative: unknown method returns descriptive error ────────────────

    /// Supplying an unknown method name must return `Err`, not panic.
    #[test]
    fn test_register_unknown_method_returns_error() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        let result = run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path,
            method: "nonexistent".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        });

        assert!(result.is_err(), "unknown method must return Err");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Unknown registration method 'nonexistent'"),
            "error must name the unsupported method, got: {msg}"
        );
    }

    // ── Negative: missing fixed image returns error ───────────────────────

    #[test]
    fn test_register_missing_fixed_returns_error() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("does_not_exist.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        let result = run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path,
            method: "rigid-mi".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 3,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        });

        assert!(result.is_err(), "missing fixed image must yield an error");
    }

    // ── Boundary: image_to_array3 round-trip preserves values ────────────

    /// Converting an image to `Array3<f64>` and back must preserve voxel
    /// values within f32 precision.
    #[test]
    fn test_image_to_array3_and_back_preserves_values() {
        let image = make_ramp_image();
        let arr = image_to_array3(&image);

        // Verify shape.
        assert_eq!(arr.dim(), (4, 4, 4), "array shape must match image shape");

        // Verify values: flat index i → value i * 4.0.
        let flat: Vec<f64> = arr.iter().copied().collect();
        for (i, &v) in flat.iter().enumerate() {
            let expected = i as f64 * 4.0;
            assert!(
                (v - expected).abs() < 1e-5,
                "element [{i}]: expected {expected}, got {v}"
            );
        }

        // Convert back and verify sum is preserved.
        let reconstructed = array3_to_image(arr, &image);
        let orig_sum: f32 = image
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();
        let recon_sum: f32 = reconstructed
            .data()
            .clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .sum();
        assert!(
            (orig_sum - recon_sum).abs() < 1e-3,
            "voxel sum must be preserved: orig={orig_sum}, recon={recon_sum}"
        );
    }

    // ── Boundary: image_to_flat_vec round-trip preserves values ───────────

    /// Converting an image to flat vec and back must preserve voxel values.
    #[test]
    fn test_image_to_flat_vec_and_back_preserves_values() {
        let image = make_ramp_image();
        let (data, shape) = image_to_flat_vec(&image);

        assert_eq!(shape, [4, 4, 4], "flat_vec shape must match image shape");
        assert_eq!(data.len(), 64, "flat_vec length must equal total voxels");

        // Verify individual values.
        for (i, &v) in data.iter().enumerate() {
            let expected = i as f32 * 4.0;
            assert!(
                (v - expected).abs() < 1e-5,
                "flat_vec element [{i}]: expected {expected}, got {v}"
            );
        }

        // Round-trip.
        let reconstructed = flat_vec_to_image(data, shape, &image);
        let recon_td = reconstructed.data().clone().into_data();
        let recon_vals = recon_td.as_slice::<f32>().unwrap();
        let orig_td = image.data().clone().into_data();
        let orig_vals = orig_td.as_slice::<f32>().unwrap();

        for (i, (&o, &r)) in orig_vals.iter().zip(recon_vals.iter()).enumerate() {
            assert!(
                (o - r).abs() < 1e-6,
                "round-trip voxel [{i}]: orig={o}, recon={r}"
            );
        }
    }
    // -- Positive: multires-demons creates output file -------------------------

    /// Running `multires-demons` with levels=1 on identical images must produce a
    /// warped output file whose shape matches the input.
    #[test]
    fn test_register_multires_demons_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "multires-demons".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        assert!(
            output_path.exists(),
            "multires-demons warped output file must be created"
        );
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(
            warped.shape(),
            [4, 4, 4],
            "multires-demons warped image shape must match fixed image shape"
        );
    }

    // -- Positive: multires-demons identity registration has low MSE ----------

    /// When fixed == moving, multires-demons final MSE must be near zero (levels=1).
    #[test]
    fn test_register_multires_demons_identity_low_mse() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "multires-demons".to_string(),
            output_transform: None,
            iterations: 5,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        // Verify the warped image has finite voxel values (identity => MSE near 0).
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let td = warped.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "multires-demons output voxel [{i}] must be finite, got {v}"
            );
        }
    }

    // ── Inverse-consistent Demons: output shape ──────────────────────────────

    #[test]
    fn test_register_ic_demons_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "ic-demons".to_string(),
            output_transform: None,
            iterations: 3,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        assert!(
            output_path.exists(),
            "ic-demons warped output file must be created"
        );
        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(
            warped.shape(),
            [4, 4, 4],
            "ic-demons warped image shape must match fixed image shape"
        );
    }

    #[test]
    fn test_register_ic_demons_identity_finite_voxels() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("warped.nii");

        let image = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &image).unwrap();
        ritk_io::write_nifti(&moving_path, &image).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "ic-demons".to_string(),
            output_transform: None,
            iterations: 5,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .unwrap();

        let warped = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let td = warped.data().clone().into_data();
        let vals = td.as_slice::<f32>().unwrap();
        for (i, &v) in vals.iter().enumerate() {
            assert!(
                v.is_finite(),
                "ic-demons output voxel [{i}] must be finite, got {v}"
            );
        }
    }

    // ── BSpline FFD: output shape ──────────────────────────────────────────────────────

    #[test]
    fn test_register_bspline_ffd_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "bspline-ffd".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("bspline-ffd must succeed");

        assert!(output_path.exists(), "output must exist");
        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(out.shape(), [4, 4, 4], "output shape must match fixed");
    }

    #[test]
    fn test_register_bspline_ffd_identity_finite_voxels() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "bspline-ffd".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("bspline-ffd must succeed");

        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let td = out.data().clone().into_data();
        let vals: &[f32] = td.as_slice().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "all output voxels must be finite"
        );
    }

    // ── Multi-resolution SyN: output shape ─────────────────────────────────────────────

    #[test]
    fn test_register_multires_syn_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "multires-syn".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("multires-syn must succeed");

        assert!(output_path.exists(), "output must exist");
        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(out.shape(), [4, 4, 4], "output shape must match fixed");
    }

    #[test]
    fn test_register_multires_syn_identity_finite_voxels() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "multires-syn".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("multires-syn must succeed");

        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let td = out.data().clone().into_data();
        let vals: &[f32] = td.as_slice().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "all output voxels must be finite"
        );
    }

    // ── BSpline SyN: output shape ─────────────────────────────────────────────────────

    #[test]
    fn test_register_bspline_syn_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "bspline-syn".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("bspline-syn must succeed");

        assert!(output_path.exists(), "output must exist");
        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(out.shape(), [4, 4, 4], "output shape must match fixed");
    }

    #[test]
    fn test_register_bspline_syn_identity_finite_voxels() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "bspline-syn".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("bspline-syn must succeed");

        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let td = out.data().clone().into_data();
        let vals: &[f32] = td.as_slice().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "all output voxels must be finite"
        );
    }

    // ── LDDMM: output shape ────────────────────────────────────────────────────────────

    #[test]
    fn test_register_lddmm_creates_output_with_correct_shape() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "lddmm".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("lddmm must succeed");

        assert!(output_path.exists(), "output must exist");
        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        assert_eq!(out.shape(), [4, 4, 4], "output shape must match fixed");
    }

    #[test]
    fn test_register_lddmm_identity_finite_voxels() {
        let dir = tempdir().unwrap();
        let fixed_path = dir.path().join("fixed.nii");
        let moving_path = dir.path().join("moving.nii");
        let output_path = dir.path().join("output.nii");

        let img = make_ramp_image();
        ritk_io::write_nifti(&fixed_path, &img).unwrap();
        ritk_io::write_nifti(&moving_path, &img).unwrap();

        run(RegisterArgs {
            fixed: fixed_path,
            moving: moving_path,
            output: output_path.clone(),
            method: "lddmm".to_string(),
            output_transform: None,
            iterations: 2,
            sigma_fixed: 0.0,
            levels: 1,
            use_diffeomorphic: false,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: false,
            num_time_steps: 2,
            kernel_sigma: 3.0,
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        })
        .expect("lddmm must succeed");

        let out = ritk_io::read_nifti::<Backend, _>(&output_path, &Default::default()).unwrap();
        let td = out.data().clone().into_data();
        let vals: &[f32] = td.as_slice().unwrap();
        assert!(
            vals.iter().all(|v| v.is_finite()),
            "all output voxels must be finite"
        );
    }
}
