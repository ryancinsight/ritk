//! Python-exposed global Mutual Information registration.
//!
//! Multi-resolution Mattes MI + RegularStepGradientDescent (RSGD) registration,
//! supporting translation, rigid, and affine transform types. Converts PyImage
//! data to Burn autodiff-backed tensors, runs registration via
//! `ritk-registration`, and returns the 4×4 homogeneous matrix, final MI
//! value, and convergence diagnostics.

use crate::image::{image_to_vec, PyImage};
use burn::backend::Autodiff;
use burn::tensor::{Shape, Tensor, TensorData};
use burn_ndarray::NdArray;
use pyo3::prelude::*;
use ritk_core::image::Image;
use ritk_core::transform::{AffineTransform, RigidTransform, TranslationTransform};
use ritk_registration::classical::global_mi::{
    GlobalMiConfig, GlobalMiRegistration, GlobalMiTransformType,
};
use ritk_registration::optimizer::regular_step_gd::RegularStepGdConfig;

// ─── Backend aliases ─────────────────────────────────────────────────────────
// NdArray<f32> is the concrete inner backend (matching crate::image::Backend).
// Autodiff<NdArray<f32>> wraps it to provide gradient computation required by
// the RSGD optimizer.
type InnerBackend = NdArray<f32>;
type AutodiffBackend = Autodiff<InnerBackend>;

// ─── Image conversion ────────────────────────────────────────────────────────

/// Convert a `PyImage` (which wraps `Image<NdArray<f32>, 3>`) into an
/// `Image<Autodiff<NdArray<f32>>, 3>` by extracting the flat f32 data and
/// re-creating the tensor on the autodiff backend.
///
/// This is necessary because `GlobalMiRegistration` requires `B: AutodiffBackend`
/// for gradient computation, while `PyImage` stores data on the non-autodiff
/// `NdArray<f32>` backend.
fn py_image_to_autodiff_image(py_image: &PyImage) -> PyResult<Image<AutodiffBackend, 3>> {
    let (values, shape) = image_to_vec(py_image.inner.as_ref())?;
    let device = Default::default();
    let tensor = Tensor::<AutodiffBackend, 3>::from_data(
        TensorData::new(values, Shape::new(shape)),
        &device,
    );
    Ok(Image::new(
        tensor,
        py_image.inner.origin().clone(),
        py_image.inner.spacing().clone(),
        py_image.inner.direction().clone(),
    ))
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
    initial_step_length: f64,
    relaxation_factor: f64,
    minimum_step_length: f64,
    maximum_iterations: usize,
) -> GlobalMiConfig {
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
            }
        })
        .collect();

    GlobalMiConfig {
        num_levels,
        shrink_factors,
        smoothing_sigmas,
        num_mi_bins,
        sampling_percentage,
        rsgd_configs,
        transform_type,
        center: None,
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
///     transform_type: Transform type — `"translation"`, `"rigid"`, or `"affine"`.
///         Default: `"rigid"`.
///     num_levels: Number of multi-resolution pyramid levels (default: 3).
///     shrink_factors: Shrink factors per level. Length must equal `num_levels`.
///         Default: `[4, 2, 1]`.
///     smoothing_sigmas: Gaussian smoothing sigmas per level in physical units.
///         Length must equal `num_levels`. Default: `[4.0, 2.0, 0.0]`.
///     num_mi_bins: Number of Mattes MI histogram bins (default: 50).
///     sampling_percentage: Fraction of voxels sampled for MI estimation (default: 0.20).
///     initial_step_length: RSGD initial step length (default: 1.0).
///     relaxation_factor: RSGD step shrinkage factor on rejected steps (default: 0.5).
///     minimum_step_length: RSGD step convergence threshold (default: 1e-6).
///     maximum_iterations: RSGD maximum accepted steps at the finest level (default: 200).
///
/// Returns:
///     (matrix, final_mi, info):
///         - matrix: 4×4 homogeneous transform as 16 floats (row-major).
///         - final_mi: Final Mattes MI value (positive; negated from the loss).
///         - info: Dict with keys:
///             - convergence_history: list of per-level convergence reason strings.
///             - iterations_per_level: list of iteration counts per level.
///             - converged: bool — True if every level converged.
///             - loss_history: list of loss values from the final level.
///
/// Raises:
///     RuntimeError: If configuration validation fails or registration produces an error.
#[pyfunction]
#[pyo3(signature = (
    fixed,
    moving,
    transform_type="rigid",
    num_levels=3,
    shrink_factors=None,
    smoothing_sigmas=None,
    num_mi_bins=50,
    sampling_percentage=0.20,
    initial_step_length=1.0,
    relaxation_factor=0.5,
    minimum_step_length=1e-6,
    maximum_iterations=200,
))]
pub fn global_mi_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    transform_type: &str,
    num_levels: usize,
    shrink_factors: Option<Vec<usize>>,
    smoothing_sigmas: Option<Vec<f64>>,
    num_mi_bins: usize,
    sampling_percentage: f32,
    initial_step_length: f64,
    relaxation_factor: f64,
    minimum_step_length: f64,
    maximum_iterations: usize,
) -> PyResult<(Vec<f32>, f64, PyObject)> {
    // ── 1. Parse transform type ──────────────────────────────────────────
    let transform_type_enum = match transform_type {
        "translation" => GlobalMiTransformType::Translation,
        "rigid" => GlobalMiTransformType::Rigid,
        "affine" => GlobalMiTransformType::Affine,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "transform_type must be \"translation\", \"rigid\", or \"affine\", got \"{}\"",
                other
            )));
        }
    };

    // ── 2. Apply default lists when not provided ─────────────────────────
    let shrink_factors = shrink_factors.unwrap_or_else(|| match num_levels {
        0 => vec![],
        1 => vec![1],
        2 => vec![2, 1],
        _ => vec![4, 2, 1],
    });
    let smoothing_sigmas = smoothing_sigmas.unwrap_or_else(|| match num_levels {
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
        num_mi_bins,
        sampling_percentage,
        initial_step_length,
        relaxation_factor,
        minimum_step_length,
        maximum_iterations,
    );

    // ── 4. Convert PyImages to autodiff-backend Images ───────────────────
    let fixed_ad = py_image_to_autodiff_image(fixed)?;
    let moving_ad = py_image_to_autodiff_image(moving)?;

    // ── 5. Run registration inside allow_threads ─────────────────────────
    let (matrix, final_mi, convergence_history, iterations_per_level, loss_history, converged) = py
        .allow_threads(|| {
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
                        result.converged,
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
                        result.converged,
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
                        result.converged,
                    )
                }
            }
        });

    // ── 6. Convert result matrix to Vec<f32> ─────────────────────────────
    let matrix_vec: Vec<f32> = matrix.iter().map(|&v| v as f32).collect();

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
    info.set_item("converged", converged)?;
    info.set_item("loss_history", loss_history)?;

    Ok((matrix_vec, final_mi, info.unbind().into()))
}
