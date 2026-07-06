//! Python-exposed global Mutual Information registration.
//!
//! Multi-resolution Mattes MI + RegularStepGradientDescent (RSGD) registration,
//! supporting translation, rigid, and affine transform types. Converts PyImage
//! data to Burn autodiff-backed tensors, runs registration via
//! `ritk-registration`, and returns the 4×4 homogeneous matrix, final MI
//! value, and convergence diagnostics.
//!
//! The CMA-ES registration binding is in the `cma_es` submodule.

mod cma_es;

use crate::errors::{RitkPyError, RitkResult};
use crate::image::{image_to_vec, PyImage};
use burn_ndarray::NdArray;
use pyo3::prelude::*;
use ritk_core::image::Image;
use ritk_filter::GaussianSigma;
use ritk_image::burn::backend::Autodiff;
use ritk_image::tensor::{Shape, Tensor, TensorData};
use ritk_registration::classical::global_mi::{
    ConvergenceStatus, GlobalMiConfig, GlobalMiRegistration, GlobalMiTransformType,
};
use ritk_registration::optimizer::regular_step_gd::RegularStepGdConfig;
use ritk_transform::{AffineTransform, RigidTransform, TranslationTransform};

// ─── Backend aliases ─────────────────────────────────────────────────────────
// NdArray<f32> is the concrete inner backend (matching crate::image::Backend).
// Autodiff<NdArray<f32>> wraps it to provide gradient computation required by
// the RSGD optimizer.
pub(crate) type InnerBackend = NdArray<f32>;
pub(crate) type AutodiffBackend = Autodiff<InnerBackend>;

// ─── Image conversion ────────────────────────────────────────────────────────

/// Convert a `PyImage` (which wraps `Image<NdArray<f32>, 3>`) into an
/// `Image<Autodiff<NdArray<f32>>, 3>` by extracting the flat f32 data and
/// re-creating the tensor on the autodiff backend.
///
/// This is necessary because `GlobalMiRegistration` requires `B: AutodiffBackend`
/// for gradient computation, while `PyImage` stores data on the non-autodiff
/// `NdArray<f32>` backend.
pub(crate) fn py_image_to_autodiff_image(py_image: &PyImage) -> Image<AutodiffBackend, 3> {
    let (values, shape) = image_to_vec(py_image.inner.as_ref());
    let device = Default::default();
    let tensor = Tensor::<AutodiffBackend, 3>::from_data(
        TensorData::new(values, Shape::new(shape)),
        &device,
    );
    Image::new(
        tensor,
        *py_image.inner.origin(),
        *py_image.inner.spacing(),
        *py_image.inner.direction(),
    )
}

/// Compute the physical center of a 3D image from its metadata.
///
/// The center is defined as: `center[d] = origin[d] + (shape[d] - 1) * spacing[d] / 2`
/// for each spatial dimension `d ∈ {0, 1, 2}`.
fn compute_image_center(image: &Image<AutodiffBackend, 3>) -> Tensor<AutodiffBackend, 1> {
    let origin = image.origin();
    let spacing = image.spacing();
    let shape = image.shape();
    let center_vals: [f32; 3] = std::array::from_fn(|d| {
        (origin[d] as f32) + ((shape[d] - 1) as f32) * (spacing[d] as f32) / 2.0
    });
    Tensor::from_data(
        TensorData::from(center_vals.as_slice()),
        &image.data().device(),
    )
}

/// Per-optimizer scalar parameters for RSGD, grouped to stay within the
/// function argument limit.
struct RsgdParams {
    initial_step_length: f64,
    relaxation_factor: f64,
    minimum_step_length: f64,
    maximum_iterations: usize,
}

/// Build a `GlobalMiConfig` from Python parameters.
///
/// Creates per-level RSGD configurations from the single set of optimizer
/// parameters provided by the caller, scaling defaults per ITK conventions:
/// - Coarse levels use larger step lengths and fewer iterations.
/// - Fine levels use smaller step lengths and more iterations.
fn build_config(
    transform_type: GlobalMiTransformType,
    num_levels: usize,
    shrink_factors: Vec<usize>,
    smoothing_sigmas: Vec<f64>,
    num_mi_bins: usize,
    sampling_percentage: f32,
    rsgd: RsgdParams,
) -> GlobalMiConfig {
    let RsgdParams {
        initial_step_length,
        relaxation_factor,
        minimum_step_length,
        maximum_iterations,
    } = rsgd;

    // Build per-level RSGD configs with step-length decay across pyramid levels.
    // Coarse levels: larger step, fewer iterations. Fine levels: smaller step, more iterations.
    let rsgd_configs: Vec<RegularStepGdConfig> = (0..num_levels)
        .map(|level| {
            let step_decay = 0.5f64.powi(level as i32);
            let iter_scale = 1.0 + 0.5 * level as f64;
            RegularStepGdConfig {
                initial_step_length: initial_step_length * step_decay,
                relaxation_factor,
                minimum_step_length,
                maximum_step_length: initial_step_length * 2.0,
                gradient_tolerance: minimum_step_length,
                maximum_iterations: ((maximum_iterations as f64) * iter_scale) as usize,
                learning_rate_decay: 0.0, // classic fixed-step RSGD (ITK-compatible)
            }
        })
        .collect();

    GlobalMiConfig {
        num_levels,
        shrink_factors,
        smoothing_sigmas: smoothing_sigmas
            .into_iter()
            .map(GaussianSigma::new)
            .collect(),
        num_mi_bins,
        sampling_percentage,
        rsgd_configs,
        transform_type,
        center: None,
    }
}

// ─── Python function ─────────────────────────────────────────────────────────

/// All scalar tuning knobs for `global_mi_register`, grouped to stay within
/// the function argument limit.
pub(crate) struct GlobalMiOptions {
    transform_type: String,
    num_levels: usize,
    shrink_factors: Option<Vec<usize>>,
    smoothing_sigmas: Option<Vec<f64>>,
    num_mi_bins: usize,
    sampling_percentage: f32,
    rsgd: RsgdParams,
}

/// Core registration logic, separated from the `#[pyfunction]` entry point so
/// the public Rust function signature stays within the argument limit.
fn run_global_mi_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: GlobalMiOptions,
) -> RitkResult<(Vec<f32>, f64, PyObject)> {
    // ── 1. Parse transform type ──────────────────────────────────────────
    let transform_type_enum = match opts.transform_type.as_str() {
        "translation" => GlobalMiTransformType::Translation,
        "rigid" => GlobalMiTransformType::Rigid,
        "affine" => GlobalMiTransformType::Affine,
        other => {
            return Err(RitkPyError::value(format!(
                "transform_type must be \"translation\", \"rigid\", or \"affine\", got \"{}\"",
                other
            )));
        }
    };
    let num_levels = opts.num_levels;

    // ── 2. Apply default lists when not provided ─────────────────────────
    let shrink_factors = opts.shrink_factors.unwrap_or_else(|| match num_levels {
        0 => vec![],
        1 => vec![1],
        2 => vec![2, 1],
        _ => vec![4, 2, 1],
    });
    let smoothing_sigmas = opts.smoothing_sigmas.unwrap_or_else(|| match num_levels {
        0 => vec![],
        1 => vec![0.0],
        2 => vec![2.0, 0.0],
        _ => vec![4.0, 2.0, 0.0],
    });

    // ── 3. Build configuration ───────────────────────────────────────────
    let config = build_config(
        transform_type_enum,
        num_levels,
        shrink_factors,
        smoothing_sigmas,
        opts.num_mi_bins,
        opts.sampling_percentage,
        opts.rsgd,
    );

    // ── 4. Convert PyImages to autodiff-backend Images ───────────────────
    let fixed_ad = py_image_to_autodiff_image(fixed);
    let moving_ad = py_image_to_autodiff_image(moving);

    // ── 5. Run registration inside allow_threads ─────────────────────────
    let (matrix, final_mi, convergence_history, iterations_per_level, loss_history, convergence) =
        py.allow_threads(|| {
            let device = Default::default();
            match transform_type_enum {
                GlobalMiTransformType::Translation => {
                    let initial = TranslationTransform::<AutodiffBackend, 3>::new(Tensor::zeros(
                        [3],
                        &device,
                    ));
                    let (_transform, result) = GlobalMiRegistration::register_translation_full(
                        &fixed_ad, &moving_ad, initial, &config,
                    );
                    (
                        result.matrix,
                        result.final_mi,
                        result.convergence_history,
                        result.iterations_per_level,
                        result.loss_history,
                        result.convergence,
                    )
                }
                GlobalMiTransformType::Rigid => {
                    let center = compute_image_center(&fixed_ad);
                    let initial =
                        RigidTransform::<AutodiffBackend, 3>::identity(Some(center), &device);
                    let (_transform, result) = GlobalMiRegistration::register_rigid_full(
                        &fixed_ad, &moving_ad, initial, &config,
                    );
                    (
                        result.matrix,
                        result.final_mi,
                        result.convergence_history,
                        result.iterations_per_level,
                        result.loss_history,
                        result.convergence,
                    )
                }
                GlobalMiTransformType::Affine => {
                    let center = compute_image_center(&fixed_ad);
                    let initial =
                        AffineTransform::<AutodiffBackend, 3>::identity(Some(center), &device);
                    let (_transform, result) = GlobalMiRegistration::register_affine_full(
                        &fixed_ad, &moving_ad, initial, &config,
                    );
                    (
                        result.matrix,
                        result.final_mi,
                        result.convergence_history,
                        result.iterations_per_level,
                        result.loss_history,
                        result.convergence,
                    )
                }
            }
        });

    // ── 6. Convert result matrix to Vec<f32> ─────────────────────────────
    let matrix_vec: Vec<f32> = matrix.0.iter().map(|&v| v as f32).collect();

    // ── 7. Build convergence info dict ───────────────────────────────────
    let convergence_strs: Vec<String> = convergence_history
        .iter()
        .map(|reason| match reason {
            ritk_registration::optimizer::regular_step_gd::ConvergenceReason::GradientConvergence => {
                "GradientConvergence".to_string()
            }
            ritk_registration::optimizer::regular_step_gd::ConvergenceReason::StepConvergence => {
                "StepConvergence".to_string()
            }
            ritk_registration::optimizer::regular_step_gd::ConvergenceReason::MaximumIterations => {
                "MaximumIterations".to_string()
            }
        })
        .collect();

    let info = pyo3::types::PyDict::new_bound(py);
    info.set_item("convergence_history", convergence_strs)?;
    info.set_item("iterations_per_level", iterations_per_level)?;
    info.set_item("converged", convergence == ConvergenceStatus::Converged)?;
    info.set_item("loss_history", loss_history)?;

    Ok((matrix_vec, final_mi, info.unbind().into()))
}

// ─── Python-visible options class ────────────────────────────────────────────

/// Registration options for `global_mi_register`.
///
/// Groups all tuning parameters so `global_mi_register` stays within the
/// function argument limit while exposing every knob to Python callers.
///
/// Args:
///     transform_type: `"translation"`, `"rigid"`, or `"affine"` (default `"rigid"`).
///     num_levels: Multi-resolution pyramid levels (default 3).
///     shrink_factors: Shrink factors per level (default `[4, 2, 1]` for 3 levels).
///     smoothing_sigmas: Gaussian smoothing sigmas per level in physical units
///         (default `[4.0, 2.0, 0.0]` for 3 levels).
///     num_mi_bins: Mattes MI histogram bins (default 50).
///     sampling_percentage: Fraction of voxels sampled for MI (default 0.20).
///     initial_step_length: RSGD initial step (default 1.0).
///     relaxation_factor: RSGD step shrinkage on rejected steps (default 0.5).
///     minimum_step_length: RSGD convergence threshold (default 1e-6).
///     maximum_iterations: RSGD max steps at finest level (default 200).
#[pyclass(name = "GlobalMiOptions")]
#[derive(Clone)]
pub struct PyGlobalMiOptions {
    #[pyo3(get, set)]
    pub transform_type: String,
    #[pyo3(get, set)]
    pub num_levels: usize,
    #[pyo3(get, set)]
    pub shrink_factors: Option<Vec<usize>>,
    #[pyo3(get, set)]
    pub smoothing_sigmas: Option<Vec<f64>>,
    #[pyo3(get, set)]
    pub num_mi_bins: usize,
    #[pyo3(get, set)]
    pub sampling_percentage: f32,
    #[pyo3(get, set)]
    pub initial_step_length: f64,
    #[pyo3(get, set)]
    pub relaxation_factor: f64,
    #[pyo3(get, set)]
    pub minimum_step_length: f64,
    #[pyo3(get, set)]
    pub maximum_iterations: usize,
}

impl Default for PyGlobalMiOptions {
    fn default() -> Self {
        Self {
            transform_type: "rigid".to_owned(),
            num_levels: 3,
            shrink_factors: None,
            smoothing_sigmas: None,
            num_mi_bins: 50,
            sampling_percentage: 0.20,
            initial_step_length: 1.0,
            relaxation_factor: 0.5,
            minimum_step_length: 1e-6,
            maximum_iterations: 200,
        }
    }
}

#[pymethods]
impl PyGlobalMiOptions {
    #[new]
    #[pyo3(signature = (
        transform_type = "rigid".to_owned(),
        num_levels = 3,
        shrink_factors = None,
        smoothing_sigmas = None,
        num_mi_bins = 50,
        sampling_percentage = 0.20,
        initial_step_length = 1.0,
        relaxation_factor = 0.5,
        minimum_step_length = 1e-6,
        maximum_iterations = 200,
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transform_type: String,
        num_levels: usize,
        shrink_factors: Option<Vec<usize>>,
        smoothing_sigmas: Option<Vec<f64>>,
        num_mi_bins: usize,
        sampling_percentage: f32,
        initial_step_length: f64,
        relaxation_factor: f64,
        minimum_step_length: f64,
        maximum_iterations: usize,
    ) -> Self {
        Self {
            transform_type,
            num_levels,
            shrink_factors,
            smoothing_sigmas,
            num_mi_bins,
            sampling_percentage,
            initial_step_length,
            relaxation_factor,
            minimum_step_length,
            maximum_iterations,
        }
    }
}

impl PyGlobalMiOptions {
    pub(crate) fn into_options(self) -> GlobalMiOptions {
        GlobalMiOptions {
            transform_type: self.transform_type,
            num_levels: self.num_levels,
            shrink_factors: self.shrink_factors,
            smoothing_sigmas: self.smoothing_sigmas,
            num_mi_bins: self.num_mi_bins,
            sampling_percentage: self.sampling_percentage,
            rsgd: RsgdParams {
                initial_step_length: self.initial_step_length,
                relaxation_factor: self.relaxation_factor,
                minimum_step_length: self.minimum_step_length,
                maximum_iterations: self.maximum_iterations,
            },
        }
    }
}

// ─── Python function ─────────────────────────────────────────────────────────

/// Register a moving image to a fixed image using global Mutual Information.
///
/// Multi-resolution Mattes MI + RegularStepGradientDescent (RSGD) registration,
/// following ITK's `ImageRegistrationMethod` architecture.
///
/// Args:
///     fixed: Fixed (reference) image.
///     moving: Moving image to register to the fixed image.
///     opts: `GlobalMiOptions` instance controlling transform type, pyramid
///         levels, MI bins, and RSGD optimizer parameters.
///
/// Returns:
///     (matrix, final_mi, info):
///         - matrix: 4×4 homogeneous transform as 16 floats (row-major).
///         - final_mi: Final Mattes MI value (positive; negated from the loss).
///         - info: Dict with keys `convergence_history`, `iterations_per_level`,
///           `converged`, `loss_history`.
///
/// Raises:
///     RuntimeError: If configuration validation fails or registration produces an error.
#[pyfunction]
pub fn global_mi_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: PyGlobalMiOptions,
) -> RitkResult<(Vec<f32>, f64, PyObject)> {
    run_global_mi_register(py, fixed, moving, opts.into_options())
}

// ─── Re-exports ──────────────────────────────────────────────────────────────

pub use cma_es::{cma_mi_register, PyCmaMiOptions};
