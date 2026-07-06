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
//! 3. Convert both images to `leto::Array3<f64>` (for MI methods) or
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

use anyhow::{Context, Result};
use clap::Args;
use leto::Array3;
use ritk_image::tensor::Backend as BurnBackend;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use std::path::PathBuf;
use tracing::info;

use super::Backend;
use ritk_core::image::Image;
use ritk_filter::{GaussianFilter, GaussianSigma};
use ritk_registration::classical::engine::{ClassicalConfig, MutualInformationMetric};
use ritk_registration::classical::spatial;
use ritk_registration::ImageRegistration;

/// CLI-visible Demons variant string, parsed into `DemonsVariant` at dispatch.
fn parse_demons_variant(s: &str) -> Result<ritk_registration::demons::DemonsVariant, String> {
    match s.to_lowercase().as_str() {
        "thirion" | "classic" => Ok(ritk_registration::demons::DemonsVariant::Classic),
        "diffeomorphic" => Ok(ritk_registration::demons::DemonsVariant::Diffeomorphic),
        other => Err(format!(
            "Invalid Demons variant '{other}'. Expected 'thirion' or 'diffeomorphic'."
        )),
    }
}

/// Parse a clap string argument into a validated [`GaussianSigma`].
///
/// Rejects any value that is not a finite positive float.
fn parse_gaussian_sigma(s: &str) -> Result<GaussianSigma, String> {
    let v: f64 = s.parse().map_err(|e| format!("invalid float: {e}"))?;
    GaussianSigma::new(v).ok_or_else(|| format!("sigma must be > 0, got {v}"))
}

// ── CLI types ────────────────────────────────────────────────────────────────

/// CLI representation of the `--inverse-consistency` flag for Multi-Resolution SyN.
///
/// Maps to [`ritk_registration::diffeomorphic::multires_syn::InverseConsistency`]
/// at dispatch time via `From<CliInverseConsistency> for InverseConsistency`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum CliInverseConsistency {
    Relaxed,
    Enforced,
}

/// Registration algorithm to use.
#[derive(clap::ValueEnum, Clone, Debug)]
pub enum RegistrationMethod {
    #[value(name = "rigid-mi")]
    RigidMi,
    #[value(name = "affine-mi")]
    AffineMi,
    Demons,
    #[value(name = "multires-demons")]
    MultiResDemons,
    #[value(name = "ic-demons")]
    IcDemons,
    Syn,
    #[value(name = "bspline-ffd")]
    BsplineFfd,
    #[value(name = "multires-syn")]
    MultiResSyn,
    #[value(name = "bspline-syn")]
    BsplineSyn,
    Lddmm,
}

impl std::fmt::Display for RegistrationMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::RigidMi => "rigid-mi",
            Self::AffineMi => "affine-mi",
            Self::Demons => "demons",
            Self::MultiResDemons => "multires-demons",
            Self::IcDemons => "ic-demons",
            Self::Syn => "syn",
            Self::BsplineFfd => "bspline-ffd",
            Self::MultiResSyn => "multires-syn",
            Self::BsplineSyn => "bspline-syn",
            Self::Lddmm => "lddmm",
        })
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
    #[arg(long)]
    pub method: RegistrationMethod,

    /// Output path for the estimated transform (JSON array of 16 floats
    /// representing a row-major 4×4 homogeneous matrix). Optional.
    /// Only produced by `rigid-mi` and `affine-mi`.
    #[arg(long, value_name = "PATH")]
    pub output_transform: Option<PathBuf>,

    /// Maximum number of hill-climbing iterations.
    #[arg(long, default_value = "100", value_name = "INT")]
    pub iterations: usize,

    /// Standard deviation (mm) of the isotropic Gaussian filter applied to
    /// both images before registration. Must be > 0.
    #[arg(long, default_value = "1.5", value_name = "FLOAT", value_parser = parse_gaussian_sigma)]
    pub sigma_fixed: GaussianSigma,

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

    /// Enforce inverse consistency in Multi-Resolution SyN (default: relaxed).
    #[arg(long, value_enum, default_value_t = CliInverseConsistency::Relaxed)]
    pub inverse_consistency: CliInverseConsistency,

    /// Number of LDDMM time-discretization steps (default 10).
    #[arg(long, default_value = "10", value_name = "INT")]
    pub num_time_steps: usize,

    /// RKHS Gaussian kernel sigma for LDDMM regularization (default 3.0).
    #[arg(long, default_value = "3.0", value_name = "FLOAT", value_parser = parse_gaussian_sigma)]
    pub kernel_sigma: GaussianSigma,

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

// ── Image ↔ Leto volume conversion ────────────────────────────────────────────

/// Convert a 3-D `Image<Backend, 3>` to a `leto::Array3<f64>`.
///
/// Data is extracted in the image's native [Z, Y, X] layout (C-order) and
/// cast element-wise from `f32` to `f64`.
///
/// # Panics
/// Panics if the tensor data cannot be extracted as `f32`.
pub(super) fn image_to_leto_volume(image: &Image<Backend, 3>) -> Array3<f64> {
    let shape = image.shape();
    let slice = image.data_slice();
    let f64_vec: Vec<f64> = slice.iter().map(|&v| v as f64).collect();
    Array3::from_shape_vec([shape[0], shape[1], shape[2]], f64_vec)
        .expect("shape derived from image must be consistent with data length")
}

/// Convert a warped `leto::Array3<f64>` back to `Image<Backend, 3>`.
///
/// The spatial metadata (origin, spacing, direction) is copied from
/// `reference` so the output image lives in the fixed image's frame.
pub(super) fn leto_volume_to_image(
    volume: Array3<f64>,
    reference: &Image<Backend, 3>,
) -> Image<Backend, 3> {
    let device: <Backend as BurnBackend>::Device = Default::default();
    let [nz, ny, nx] = volume.shape();
    let f32_vec: Vec<f32> = volume.iter().map(|&v| v as f32).collect();
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
        args.sigma_fixed.get()
    );

    match &args.method {
        RegistrationMethod::RigidMi | RegistrationMethod::AffineMi => {
            mi::run_mi_registration(&args)
        }
        RegistrationMethod::Demons => demons::run_demons(&args),
        RegistrationMethod::MultiResDemons => demons::run_multires_demons(&args),
        RegistrationMethod::IcDemons => demons::run_inverse_consistent_demons(&args),
        RegistrationMethod::Syn => diffeomorphic::run_syn(&args),
        RegistrationMethod::BsplineFfd => diffeomorphic::run_bspline_ffd(&args),
        RegistrationMethod::MultiResSyn => diffeomorphic::run_multires_syn(&args),
        RegistrationMethod::BsplineSyn => diffeomorphic::run_bspline_syn(&args),
        RegistrationMethod::Lddmm => lddmm::run_lddmm(&args),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use ritk_core::image::Image;
    use ritk_image::tensor::Backend as BurnBackend;
    use ritk_image::tensor::{Shape, Tensor, TensorData};
    use ritk_registration::demons::DemonsVariant;
    use ritk_spatial::{Direction, Point, Spacing};
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

    // ── Negative: invalid method names are rejected by clap at parse time;
    //    `run()` is exhaustive over `RegistrationMethod` and cannot receive
    //    an unknown variant. ─────────────────────────────────────────────

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
            method: RegistrationMethod::RigidMi,
            output_transform: None,
            iterations: 3,
            sigma_fixed: GaussianSigma::default(),
            levels: 3,
            variant: DemonsVariant::Classic,
            regularization_weight: 0.001,
            control_spacing: 4,
            cc_radius: 2,
            inverse_consistency: CliInverseConsistency::Relaxed,
            num_time_steps: 2,
            kernel_sigma: GaussianSigma::new_unchecked(3.0),
            learning_rate: 0.01,
            inverse_consistency_weight: 0.5,
            n_squarings: 6,
            convergence_threshold: 1e-5,
        });

        assert!(result.is_err(), "missing fixed image must yield an error");
    }
}
