// ─── CMA-ES Registration Python binding ──────────────────────────────────────

use pyo3::prelude::*;

use crate::errors::{RitkPyError, RitkResult};
use crate::image::PyImage;
use ritk_core::image::Image;
use ritk_registration::{CmaMiConfig, CmaMiRegistration, InitStrategy};

use super::{py_image_to_autodiff_image, AutodiffBackend};

/// Options for `cma_mi_register` (CMA-ES + optional RSGD cascade).
///
/// Use a `preset` to select a standard configuration, or set `preset = "custom"`
/// and adjust individual fields.
///
/// Presets:
/// - `"brain_default"` — Single-level CMA-ES, shrink=8, NMI, sigma0=0.7, 200 gen.
/// - `"brain_multiscale"` — 3-level cascade [16→8→4], NMI, recommended for typical brain CT.
/// - `"brain_multiscale_thin_slab"` — 3-level anisotropic cascade [\[1,16,16\]→...], NMI,
/// recommended for RIRE-style thin CT (≤50 z-slices at ≥2 mm spacing).
/// - `"fast_exploratory"` — Single-level, shrink=16, Mattes MI, fast but coarse.
/// - `"custom"` — Build config from individual fields below.
///
/// Custom fields (used when `preset == "custom"`):
/// - `coarse_shrink`: Isotropic shrink factor for CMA-ES level (default 8).
/// - `num_mi_bins`: Histogram bins (default 32).
/// - `sampling_percentage`: Fraction of voxels sampled (default 0.25).
/// - `translation_range_mm`: Half-range for translation search in mm (default 60.0).
/// - `rotation_range_rad`: Half-range for rotation search in radians (default π/4).
/// - `sigma0`: CMA-ES initial step size in normalised space (default 0.7).
/// - `max_generations`: Maximum CMA-ES generations (default 200).
/// - `init_strategy`: Initialization strategy: "manual" (default) or "center_of_mass".
///   "center_of_mass" automatically computes a pre-alignment translation from
///   the image centers of mass (unreliable for CT↔MRI T1 cross-modal).
///   "manual" uses the provided initial transform or identity.
#[pyclass(name = "CmaMiOptions")]
#[derive(Clone)]
pub struct PyCmaMiOptions {
    #[pyo3(get, set)]
    pub preset: String,
    #[pyo3(get, set)]
    pub coarse_shrink: usize,
    #[pyo3(get, set)]
    pub num_mi_bins: usize,
    #[pyo3(get, set)]
    pub sampling_percentage: f32,
    #[pyo3(get, set)]
    pub translation_range_mm: f64,
    #[pyo3(get, set)]
    pub rotation_range_rad: f64,
    #[pyo3(get, set)]
    pub sigma0: f64,
    #[pyo3(get, set)]
    pub max_generations: usize,
    #[pyo3(get, set)]
    pub init_strategy: PyInitStrategy,
}

/// Initialization strategy for CMA-ES registration, replacing `use_com_init: bool`.
///
/// Eliminates boolean blindness: `init_strategy="center_of_mass"` vs
/// `init_strategy="manual"` is self-documenting compared to `use_com_init=True/False`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PyInitStrategy {
    /// Skip automatic pre-alignment; use the provided initial transform or identity (default).
    #[default]
    Manual,
    /// Use center-of-mass of images as initial translation.
    /// Note: unreliable for CT↔MRI T1 cross-modal registration.
    CenterOfMass,
}

impl<'py> FromPyObject<'py> for PyInitStrategy {
    fn extract_bound(ob: &pyo3::Bound<'py, PyAny>) -> PyResult<Self> {
        let s: String = ob.extract()?;
        match s.to_lowercase().as_str() {
            "manual" => Ok(Self::Manual),
            "center_of_mass" | "com" => Ok(Self::CenterOfMass),
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown init strategy '{}'. Choices: manual, center_of_mass",
                other
            ))),
        }
    }
}

impl IntoPy<PyObject> for PyInitStrategy {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Self::Manual => "manual".into_py(py),
            Self::CenterOfMass => "center_of_mass".into_py(py),
        }
    }
}

impl From<PyInitStrategy> for InitStrategy {
    fn from(val: PyInitStrategy) -> Self {
        match val {
            PyInitStrategy::Manual => InitStrategy::Manual,
            PyInitStrategy::CenterOfMass => InitStrategy::CenterOfMass,
        }
    }
}

impl Default for PyCmaMiOptions {
    fn default() -> Self {
        Self {
            preset: "brain_default".to_owned(),
            coarse_shrink: 8,
            num_mi_bins: 32,
            sampling_percentage: 0.25,
            translation_range_mm: 60.0,
            rotation_range_rad: std::f64::consts::FRAC_PI_4,
            sigma0: 0.7,
            max_generations: 200,
            init_strategy: PyInitStrategy::Manual,
        }
    }
}

#[pymethods]
impl PyCmaMiOptions {
    #[new]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Build a `CmaMiConfig` from Python options.
pub(crate) fn build_cma_config(opts: &PyCmaMiOptions) -> RitkResult<CmaMiConfig> {
    match opts.preset.as_str() {
        "brain_default" => Ok(CmaMiConfig::brain_rigid_default()),
        "brain_multiscale" => Ok(CmaMiConfig::brain_rigid_multiscale()),
        "brain_multiscale_thin_slab" => Ok(CmaMiConfig::brain_rigid_multiscale_thin_slab()),
        "fast_exploratory" => Ok(CmaMiConfig::fast_exploratory()),
        "custom" => {
            let defaults = CmaMiConfig::default();
            Ok(CmaMiConfig {
                coarse_shrink: opts.coarse_shrink,
                num_mi_bins: opts.num_mi_bins,
                sampling_percentage: opts.sampling_percentage,
                translation_range_mm: opts.translation_range_mm,
                rotation_range_rad: opts.rotation_range_rad,
                init_strategy: InitStrategy::from(opts.init_strategy),
                cma_config: ritk_registration::optimizer::CmaEsConfig {
                    sigma0: opts.sigma0,
                    max_generations: opts.max_generations,
                    ..defaults.cma_config
                },
                ..defaults
            })
        }
        other => Err(RitkPyError::value(format!(
            "CmaMiOptions.preset must be one of \
             \"brain_default\", \"brain_multiscale\", \
             \"brain_multiscale_thin_slab\", \"fast_exploratory\", \"custom\"; \
             got \"{}\"",
            other
        ))),
    }
}

/// Register a moving image to a fixed image using CMA-ES global search + optional RSGD refinement.
///
/// Runs a CMA-ES (Covariance Matrix Adaptation Evolution Strategy) global search
/// over 6-DOF rigid parameters to maximise Mutual Information, followed by an
/// optional RSGD fine refinement. CMA-ES is derivative-free and can escape local
/// MI maxima that trap gradient-based methods.
///
/// Args:
///     fixed: Fixed (reference) image.
///     moving: Moving image to register to the fixed image.
///     opts: `CmaMiOptions` instance controlling the preset and optional custom parameters.
///     fixed_mask: Optional binary mask in fixed-image space (same shape and
///         spacing as `fixed`). When provided, only voxels where the mask
///         value exceeds 0.5 contribute to MI estimation at each pyramid
///         level (ANTs/ITK masking strategy). ``None`` → uniform stochastic
///         sampling over all voxels (default).
///
/// Returns:
///     (matrix, final_mi, info):
///         - matrix: 4×4 homogeneous transform as 16 floats (row-major).
///         - final_mi: Final MI value (positive; negated from the minimisation objective).
///         - info: Dict with keys `cma_generations`, `stop_reason`, `final_sigma`,
///           `rsgd_iterations`.
///
/// Raises:
///     ValueError: If `opts.preset` is not a recognised preset name.
///     RuntimeError: If registration produces an error.
#[pyfunction]
#[pyo3(signature = (fixed, moving, opts, fixed_mask=None))]
pub fn cma_mi_register(
    py: Python<'_>,
    fixed: &PyImage,
    moving: &PyImage,
    opts: PyCmaMiOptions,
    fixed_mask: Option<&PyImage>,
) -> RitkResult<(Vec<f32>, f64, PyObject)> {
    let config = build_cma_config(&opts)?;

    // Convert PyImages to autodiff-backend Images
    let fixed_ad = py_image_to_autodiff_image(fixed);
    let moving_ad = py_image_to_autodiff_image(moving);
    let mask_ad: Option<Image<AutodiffBackend, 3>> = fixed_mask.map(py_image_to_autodiff_image);

    // Run CMA-ES + optional RSGD inside allow_threads to release the GIL
    let (
        matrix,
        final_mi,
        cma_generations,
        stop_reason,
        final_sigma,
        rsgd_iterations,
        best_tz_mm,
        best_ty_mm,
        best_tx_mm,
        best_alpha_rad,
        best_beta_rad,
        best_gamma_rad,
    ) = py.allow_threads(|| {
        let (_transform, result) = CmaMiRegistration::register_rigid_with_mask(
            &fixed_ad,
            &moving_ad,
            [0.0, 0.0, 0.0],
            None,
            &config,
            mask_ad.as_ref(),
        );
        // Extract the denormalized best params for diagnostics
        let rot_scale = config.rotation_range_rad;
        let trans_scale = config.translation_range_mm;
        let bx = &result.cma_best_params;
        let (alpha, beta, gamma, tz, ty, tx) = if bx.len() >= 6 {
            (
                bx[0].clamp(-1.0, 1.0) * rot_scale,
                bx[1].clamp(-1.0, 1.0) * rot_scale,
                bx[2].clamp(-1.0, 1.0) * rot_scale,
                bx[3].clamp(-1.0, 1.0) * trans_scale,
                bx[4].clamp(-1.0, 1.0) * trans_scale,
                bx[5].clamp(-1.0, 1.0) * trans_scale,
            )
        } else {
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        };
        (
            result.matrix,
            result.final_mi,
            result.cma_generations,
            result.cma_stop_reason,
            result.cma_final_sigma,
            result.rsgd_iterations,
            tz,
            ty,
            tx,
            alpha,
            beta,
            gamma,
        )
    });

    // Convert result matrix to Vec<f32>
    let matrix_vec: Vec<f32> = matrix.0.iter().map(|&v| v as f32).collect();

    // Build info dict
    let info = pyo3::types::PyDict::new_bound(py);
    info.set_item("cma_generations", cma_generations)?;
    info.set_item("stop_reason", format!("{:?}", stop_reason))?;
    info.set_item("final_sigma", final_sigma)?;
    info.set_item("rsgd_iterations", rsgd_iterations)?;
    // Best transform parameters (denormalized, in physical units) for diagnostics
    info.set_item("best_tz_mm", best_tz_mm)?;
    info.set_item("best_ty_mm", best_ty_mm)?;
    info.set_item("best_tx_mm", best_tx_mm)?;
    info.set_item("best_alpha_rad", best_alpha_rad)?;
    info.set_item("best_beta_rad", best_beta_rad)?;
    info.set_item("best_gamma_rad", best_gamma_rad)?;

    Ok((matrix_vec, final_mi, info.unbind().into()))
}
