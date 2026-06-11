//! `ritk register` — image registration command.
//!
//! Registers a moving image to a fixed reference image.
//!
//! # Supported methods
//!
//! | Method | DOF | Algorithm |
//! |-------------------|-------|--------------------------------------------------|
//! | `rigid-mi` | 6 | Rigid hill-climbing (MI) |
//! | `affine-mi` | 9 | Affine hill-climbing (MI) |
//! | `demons` | dense | Thirion Demons deformable |
//! | `multires-demons` | dense | Multi-resolution Thirion/Diffeomorphic Demons |
//! | `ic-demons` | dense | Inverse-consistent diffeomorphic Demons |
//! | `syn` | dense | Greedy SyN diffeomorphic |
//! | `bspline-ffd` | dense | B-Spline FFD deformable (Rueckert 1999) |
//! | `multires-syn` | dense | Multi-resolution SyN (coarse-to-fine pyramid) |
//! | `bspline-syn` | dense | BSpline SyN (B-spline velocity fields) |
//! | `lddmm` | dense | LDDMM geodesic shooting (EPDiff, Gaussian RKHS) |
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

mod demons;
mod diffeomorphic;
mod lddmm;
mod mi;

use anyhow::{anyhow, Context, Result};
use burn::tensor::backend::Backend as BurnBackend;
use burn::tensor::{Shape, Tensor, TensorData};
use clap::Args;
use std::path::PathBuf;
use tracing::info;

use super::Backend;
use ritk_core::filter::{GaussianFilter, GaussianSigma};
use ritk_core::image::Image;
use ritk_registration::classical::engine::{ClassicalConfig, MutualInformationMetric};
use ritk_registration::classical::spatial;
use ritk_registration::ImageRegistration;

/// CLI-visible Demons variant string, parsed into [`DemonsVariant`] at dispatch.
fn parse_demons_variant(s: &str) -> Result<ritk_registration::demons::DemonsVariant, String> {
    match s.to_lowercase().as_str() {
        "thirion" | "classic" => Ok(ritk_registration::demons::DemonsVariant::Classic),
        "diffeomorphic" => Ok(ritk_registration::demons::DemonsVariant::Diffeomorphic),
        other => Err(format!(
            "Invalid Demons variant '{other}'. Expected 'thirion' or 'diffeomorphic'."
        )),
    }
}

// ── CLI arguments ─────────────────────────────────────────────────────────────

/// Arguments for the `register` subcommand.
#[derive(Args, Debug)]
pub struct RegisterArgs {
    /// Fixed (reference) image path. Format is inferred from the file extension.
    #[arg(long)]
    pub fixed: PathBuf,

    /// Moving image path. Format is inferred from the file extension.
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
    /// representing a row-major 4×4 homogeneous matrix). Optional.
    /// Only produced by `rigid-mi` and `affine-mi`.
    #[arg(long, value_name = "PATH")]
    pub output_transform: Option<PathBuf>,

    /// Maximum number of hill-climbing iterations.
    #[arg(long, default_value = "100", value_name = "INT")]
    pub iterations: usize,

    /// Standard deviation (mm) of the isotropic Gaussian filter applied to
    /// both images before registration. Set to 0.0 to disable smoothing.
    #[arg(long, default_value = "1.5", value_name = "FLOAT")]
    pub sigma_fixed: f64,

    /// Number of pyramid levels for multi-resolution Demons (default 3).
    #[arg(long, default_value = "3", value_name = "INT")]
    pub levels: usize,

    /// Demons variant for multi-resolution registration (default: thirion).
    ///
    /// Accepted values: `thirion`, `diffeomorphic`.
    #[arg(long, default_value = "thirion", value_name = "VARIANT", value_parser = parse_demons_variant)]
    pub variant: ritk_registration::demons::DemonsVariant,

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
pub(super) fn image_to_array3(image: &Image<Backend, 3>) -> ndarray::Array3<f64> {
    let shape = image.shape();
    let slice = image.data_slice();
    let f64_vec: Vec<f64> = slice.iter().map(|&v| v as f64).collect();
    ndarray::Array3::from_shape_vec((shape[0], shape[1], shape[2]), f64_vec)
        .expect("shape derived from image must be consistent with data length")
}

/// Convert a warped `ndarray::Array3<f64>` back to `Image<Backend, 3>`.
///
/// The spatial metadata (origin, spacing, direction) is copied from
/// `reference` so the output image lives in the fixed image's frame.
pub(super) fn array3_to_image(
    arr: ndarray::Array3<f64>,
    reference: &Image<Backend, 3>,
) -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let (nz, ny, nx) = arr.dim();
    let f32_vec: Vec<f32> = arr.iter().map(|&v| v as f32).collect();
    let td = TensorData::new(f32_vec, Shape::new([nz, ny, nx]));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        *reference.origin(),
        *reference.spacing(),
        *reference.direction(),
    )
}

// ── Image ↔ flat Vec<f32> conversion (for demons / SyN) ──────────────────────

/// Extract a flat `Vec<f32>` in Z-major order and the `[nz, ny, nx]` shape
/// from a 3-D image.
///
/// # Panics
/// Panics if the tensor data cannot be extracted as `f32`.
pub(super) fn image_to_flat_vec(image: &Image<Backend, 3>) -> (Vec<f32>, [usize; 3]) {
    let shape = image.shape();
    let data: Vec<f32> = image.data_slice().into_owned();
    (data, [shape[0], shape[1], shape[2]])
}

/// Reconstruct an `Image<Backend, 3>` from flat `Vec<f32>` data and a
/// `[nz, ny, nx]` shape, copying spatial metadata from `reference`.
pub(super) fn flat_vec_to_image(
    data: Vec<f32>,
    shape: [usize; 3],
    reference: &Image<Backend, 3>,
) -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let td = TensorData::new(data, Shape::new(shape));
    let tensor = Tensor::<Backend, 3>::from_data(td, &device);
    Image::new(
        tensor,
        *reference.origin(),
        *reference.spacing(),
        *reference.direction(),
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
        "rigid-mi" | "affine-mi" => mi::run_mi_registration(&args),
        "demons" => demons::run_demons(&args),
        "multires-demons" => demons::run_multires_demons(&args),
        "ic-demons" => demons::run_inverse_consistent_demons(&args),
        "syn" => diffeomorphic::run_syn(&args),
        "bspline-ffd" => diffeomorphic::run_bspline_ffd(&args),
        "multires-syn" => diffeomorphic::run_multires_syn(&args),
        "bspline-syn" => diffeomorphic::run_bspline_syn(&args),
        "lddmm" => lddmm::run_lddmm(&args),
        other => Err(anyhow!(
            "Unknown registration method '{other}'. \
            Supported methods: rigid-mi, affine-mi, demons, multires-demons, ic-demons, syn, \
            bspline-ffd, multires-syn, bspline-syn, lddmm."
        )),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BurnBackend;
    use burn::tensor::{Shape, Tensor, TensorData};
    use ritk_core::image::Image;
    use ritk_core::spatial::{Direction, Point, Spacing};
    use ritk_registration::demons::DemonsVariant;
    use tempfile::tempdir;

    /// Build a deterministic 4×4×4 image from a ramp of intensities.
    ///
    /// Using a ramp (not constant) so the MI metric has a non-degenerate
    /// joint histogram to work with.
    pub(crate) fn make_ramp_image() -> Image<Backend, 3> {
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
            variant: DemonsVariant::Classic,
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
            variant: DemonsVariant::Classic,
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
}
